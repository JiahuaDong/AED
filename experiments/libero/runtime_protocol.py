"""Check that the installed MuJoCo version matches the one used for LIBERO evaluation."""
import importlib.metadata as metadata

REQUIRED_MUJOCO = "3.3.2"


def assert_mujoco_pin(required=REQUIRED_MUJOCO):
    """Raise if the installed MuJoCo version differs from `required`."""
    if required not in ("3.3.2", "3.3.5"):
        raise ValueError(f"Unsupported explicit MuJoCo protocol: {required!r}")
    import mujoco

    imported = getattr(mujoco, "__version__", None)
    try:
        declared = metadata.version("mujoco")
    except metadata.PackageNotFoundError:
        declared = None
    path = str(getattr(mujoco, "__file__", "unknown"))
    if imported != required or declared != required:
        raise RuntimeError(
            f"LIBERO evaluation requires MuJoCo {required}; "
            f"imported={imported}, distribution={declared}, file={path}. "
            "Check that the right interpreter/PYTHONPATH is active and that no other "
            "MuJoCo install shadows it."
        )
    return {"version": imported, "distribution_version": declared, "module_path": path}
