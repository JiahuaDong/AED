"""Preserve the post-model-load RNG boundary of single-process evaluation."""
import random
import os

import numpy as np
import torch


def capture_rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state().clone(),
                cuda=[x.clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else None)


def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def bind_physical_egl(gpu):
    """Make robosuite render on the physical GPU this worker is pinned to."""
    from robosuite.renderers.context import egl_context
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu):
        raise ValueError('EGL binding must match the worker physical CUDA device')
    if egl_context.EGL_DISPLAY is not None:
        raise RuntimeError('Cannot rebind an existing EGL display')

    def create_display(device_id=0):
        if device_id not in (-1, gpu):
            raise ValueError(f'Unexpected EGL device request {device_id}, expected {gpu}')
        egl = egl_context.EGL
        devices = egl.eglQueryDevicesEXT()
        if not 0 <= gpu < len(devices):
            raise RuntimeError(f'Physical GPU{gpu} absent from EGL device enumeration')
        display = egl.eglGetPlatformDisplayEXT(egl.EGL_PLATFORM_DEVICE_EXT, devices[gpu], None)
        if display == egl.EGL_NO_DISPLAY or egl.eglGetError() != egl.EGL_SUCCESS:
            raise RuntimeError(f'Failed to create EGL display for GPU{gpu}')
        initialized = egl.eglInitialize(display, None, None)
        if initialized != egl.EGL_TRUE or egl.eglGetError() != egl.EGL_SUCCESS:
            raise RuntimeError(f'Failed to initialize EGL display for GPU{gpu}')
        return display

    egl_context.create_initialized_egl_device_display = create_display
