"""Ground-truth episode diagnostics for LIBERO closed-loop evaluation.

This module reads simulator state.  The optional oracle retry works as follows:
after a confirmed empty gripper closure it asks the caller to open the gripper,
discard the stale action chunk, and replan from the resulting observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


# LIBERO's final BDDL goal for this task only contains the bowl-in-drawer
# predicate, even though opening the drawer is an explicit instruction phase.
_EXTRA_SUBTASK_PREDICATES: dict[tuple[str, int], list[tuple[str, list[str]]]] = {
    ("libero_goal", 3): [
        (
            "open_top_drawer",
            ["open", "wooden_cabinet_1_top_region"],
        ),
    ],
}


def _unwrap_libero_env(env: Any) -> Any:
    """Return the underlying BDDLBaseDomain, raising if it cannot be found."""
    current = env
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        if all(
            hasattr(current, attr)
            for attr in (
                "parsed_problem",
                "objects_dict",
                "fixtures_dict",
                "_eval_predicate",
                "_check_grasp",
                "robots",
            )
        ):
            return current
        if not hasattr(current, "env"):
            break
        current = current.env
    raise RuntimeError(
        "LIBERO diagnostics require access to the underlying BDDLBaseDomain; "
        f"could not unwrap environment type {type(env)!r}."
    )


def _predicate_label(state: Iterable[Any]) -> str:
    values = [str(value) for value in state]
    if len(values) < 2:
        raise ValueError(f"Invalid LIBERO predicate: {values!r}")
    return f"{values[0]}({', '.join(values[1:])})"


@dataclass(frozen=True)
class OracleRetryDecision:
    should_retry: bool
    reason: str | None = None


class LiberoEpisodeDiagnostics:
    """Track BDDL subgoals, true grasp state, and optional oracle retries."""

    def __init__(
        self,
        env: Any,
        *,
        task_suite_name: str,
        task_id: int,
        oracle_grasp_retry: bool,
        oracle_grasp_confirm_steps: int,
        oracle_gripper_closed_threshold: float,
        oracle_grasp_max_retries: int,
    ) -> None:
        self.base_env = _unwrap_libero_env(env)
        self.task_suite_name = str(task_suite_name)
        self.task_id = int(task_id)
        self.oracle_grasp_retry = bool(oracle_grasp_retry)
        self.oracle_grasp_confirm_steps = int(oracle_grasp_confirm_steps)
        self.oracle_gripper_closed_threshold = float(
            oracle_gripper_closed_threshold
        )
        self.oracle_grasp_max_retries = int(oracle_grasp_max_retries)

        if self.oracle_grasp_confirm_steps <= 0:
            raise ValueError("oracle_grasp_confirm_steps must be positive.")
        if self.oracle_gripper_closed_threshold <= 0:
            raise ValueError("oracle_gripper_closed_threshold must be positive.")
        if self.oracle_grasp_max_retries < 0:
            raise ValueError("oracle_grasp_max_retries must be non-negative.")

        goal_states = self.base_env.parsed_problem.get("goal_state")
        if not isinstance(goal_states, list) or len(goal_states) == 0:
            raise RuntimeError("LIBERO diagnostics found no BDDL goal predicates.")
        self.goal_states = [list(state) for state in goal_states]

        extras = _EXTRA_SUBTASK_PREDICATES.get(
            (self.task_suite_name, self.task_id), []
        )
        self.subtasks: list[dict[str, Any]] = [
            {
                "name": name,
                "predicate": list(predicate),
                "source": "instruction_intermediate",
            }
            for name, predicate in extras
        ]
        self.subtasks.extend(
            {
                "name": f"bddl_goal_{idx + 1}",
                "predicate": state,
                "source": "bddl_goal",
            }
            for idx, state in enumerate(self.goal_states)
        )
        for subtask in self.subtasks:
            subtask.update(
                {
                    "label": _predicate_label(subtask["predicate"]),
                    "ever_true": False,
                    "first_true_step": None,
                    "final_true": False,
                    "transitions": [],
                }
            )

        self.movable_models = dict(self.base_env.objects_dict)
        self.fixture_models = {
            name: model
            for name, model in self.base_env.fixtures_dict.items()
            if name not in {"main_table", "floor", "countertop", "coffee_table"}
        }
        self.grasp_target_names = self._resolve_grasp_targets()
        if len(self.grasp_target_names) == 0:
            raise RuntimeError(
                "LIBERO diagnostics could not resolve a movable grasp target "
                f"from goal predicates {self.goal_states!r}."
            )

        self.grasp_success_by_target = {
            name: {"ever_grasped": False, "first_grasp_step": None}
            for name in self.grasp_target_names
        }
        self.grasp_events: list[dict[str, Any]] = []
        self.grasp_attempts = 0
        self.missed_grasp_attempts = 0
        self.oracle_retry_count = 0
        self.oracle_retry_steps: list[int] = []
        self._previous_grasped_targets: set[str] = set()
        self._previous_close_command = False
        self._active_close_attempt = False
        self._close_command_steps = 0
        self._retry_latched_until_open = False
        self._last_subtask_values = [False] * len(self.subtasks)
        self.steps_observed = 0

    def _resolve_grasp_targets(self) -> list[str]:
        targets: list[str] = []
        for state in self.goal_states:
            if len(state) >= 2 and state[1] in self.movable_models:
                name = str(state[1])
                if name not in targets:
                    targets.append(name)
        if targets:
            return targets
        for name in self.base_env.parsed_problem.get("obj_of_interest", []):
            if name in self.movable_models and name not in targets:
                targets.append(str(name))
        return targets

    def _eval_predicate(self, state: list[str]) -> bool:
        value = self.base_env._eval_predicate(state)
        if value is None:
            raise RuntimeError(f"Predicate evaluation returned None for {state!r}.")
        return bool(value)

    def _grasped_names(self, models: dict[str, Any]) -> set[str]:
        gripper = self.base_env.robots[0].gripper
        grasped: set[str] = set()
        for name, model in models.items():
            if bool(self.base_env._check_grasp(gripper=gripper, object_geoms=model)):
                grasped.add(str(name))
        return grasped

    def _record_subtasks(self, step: int) -> None:
        for idx, subtask in enumerate(self.subtasks):
            current = self._eval_predicate(subtask["predicate"])
            if current and not subtask["ever_true"]:
                subtask["ever_true"] = True
                subtask["first_true_step"] = int(step)
            if current != self._last_subtask_values[idx]:
                subtask["transitions"].append(
                    {"step": int(step), "value": bool(current)}
                )
            subtask["final_true"] = bool(current)
            self._last_subtask_values[idx] = bool(current)

    def observe_initial_state(self) -> None:
        self._record_subtasks(step=0)

    def observe_step(
        self,
        *,
        step: int,
        executed_action: np.ndarray,
        obs: dict[str, Any],
        allow_oracle_retry: bool,
    ) -> OracleRetryDecision:
        """Record post-action state and decide whether an empty grasp must retry."""
        action = np.asarray(executed_action, dtype=np.float32).reshape(-1)
        if action.size == 0:
            raise ValueError("Executed action is empty.")
        if "robot0_gripper_qpos" not in obs:
            raise RuntimeError(
                "LIBERO diagnostics require obs['robot0_gripper_qpos']."
            )

        self.steps_observed = max(self.steps_observed, int(step))
        self._record_subtasks(step=int(step))

        grasped_movable = self._grasped_names(self.movable_models)
        grasped_fixtures = self._grasped_names(self.fixture_models)
        grasped_targets = grasped_movable.intersection(self.grasp_target_names)
        newly_grasped_targets = grasped_targets - self._previous_grasped_targets
        for name in sorted(newly_grasped_targets):
            target_state = self.grasp_success_by_target[name]
            if not target_state["ever_grasped"]:
                target_state["ever_grasped"] = True
                target_state["first_grasp_step"] = int(step)
            self.grasp_events.append(
                {"step": int(step), "event": "target_grasped", "object": name}
            )
        self._previous_grasped_targets = set(grasped_targets)

        close_command = bool(float(action[-1]) > 0.0)
        if close_command and not self._previous_close_command:
            self.grasp_attempts += 1
            self._active_close_attempt = True
            self._close_command_steps = 0
            self.grasp_events.append(
                {"step": int(step), "event": "close_attempt_started"}
            )
        if close_command:
            self._close_command_steps += 1
        else:
            if self._active_close_attempt and not grasped_movable:
                self.missed_grasp_attempts += 1
                self.grasp_events.append(
                    {"step": int(step), "event": "close_attempt_ended_empty"}
                )
            self._active_close_attempt = False
            self._close_command_steps = 0
            self._retry_latched_until_open = False

        occupied = bool(grasped_movable or grasped_fixtures)
        gripper_qpos = np.asarray(
            obs["robot0_gripper_qpos"], dtype=np.float32
        ).reshape(-1)
        physically_closed = bool(
            gripper_qpos.size > 0
            and float(np.max(np.abs(gripper_qpos)))
            <= self.oracle_gripper_closed_threshold
        )

        should_retry = bool(
            allow_oracle_retry
            and self.oracle_grasp_retry
            and self._active_close_attempt
            and close_command
            and self._close_command_steps >= self.oracle_grasp_confirm_steps
            and physically_closed
            and not occupied
            and not self._retry_latched_until_open
            and self.oracle_retry_count < self.oracle_grasp_max_retries
        )
        if should_retry:
            self.missed_grasp_attempts += 1
            self.oracle_retry_count += 1
            self.oracle_retry_steps.append(int(step))
            self._active_close_attempt = False
            self._retry_latched_until_open = True
            self.grasp_events.append(
                {
                    "step": int(step),
                    "event": "oracle_empty_grasp_retry",
                    "gripper_qpos": gripper_qpos.tolist(),
                }
            )

        self._previous_close_command = close_command
        if should_retry:
            return OracleRetryDecision(
                should_retry=True,
                reason=(
                    "gripper physically closed without grasping a movable "
                    "object or articulated fixture"
                ),
            )
        return OracleRetryDecision(should_retry=False)

    def summary(self) -> dict[str, Any]:
        grasped_target_count = sum(
            int(state["ever_grasped"])
            for state in self.grasp_success_by_target.values()
        )
        return {
            "task_suite": self.task_suite_name,
            "task_id": self.task_id,
            "grasp_targets": list(self.grasp_target_names),
            "grasp_success_any": bool(grasped_target_count > 0),
            "grasp_success_all": bool(
                grasped_target_count == len(self.grasp_target_names)
            ),
            "grasp_success_by_target": self.grasp_success_by_target,
            "grasp_attempts": int(self.grasp_attempts),
            "missed_grasp_attempts": int(self.missed_grasp_attempts),
            "grasp_events": list(self.grasp_events),
            "subtasks": self.subtasks,
            "oracle_enabled": bool(self.oracle_grasp_retry),
            "oracle_retry_count": int(self.oracle_retry_count),
            "oracle_retry_steps": list(self.oracle_retry_steps),
            "steps_observed": int(self.steps_observed),
        }


def aggregate_episode_diagnostics(
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(episodes) == 0:
        raise ValueError("Cannot aggregate an empty diagnostics list.")

    total = len(episodes)
    reference_subtasks = episodes[0]["subtasks"]
    subtask_aggregate: list[dict[str, Any]] = []
    for idx, reference in enumerate(reference_subtasks):
        for episode in episodes:
            if len(episode["subtasks"]) != len(reference_subtasks):
                raise RuntimeError("Inconsistent subtask count across episodes.")
            if episode["subtasks"][idx]["label"] != reference["label"]:
                raise RuntimeError("Inconsistent subtask labels across episodes.")
        ever_count = sum(
            int(episode["subtasks"][idx]["ever_true"]) for episode in episodes
        )
        final_count = sum(
            int(episode["subtasks"][idx]["final_true"]) for episode in episodes
        )
        subtask_aggregate.append(
            {
                "index": idx + 1,
                "name": reference["name"],
                "label": reference["label"],
                "source": reference["source"],
                "ever_successes": ever_count,
                "ever_success_rate": ever_count / total,
                "final_successes": final_count,
                "final_success_rate": final_count / total,
            }
        )

    grasp_any = sum(int(episode["grasp_success_any"]) for episode in episodes)
    grasp_all = sum(int(episode["grasp_success_all"]) for episode in episodes)
    return {
        "episodes": total,
        "grasp_success_any_episodes": grasp_any,
        "grasp_success_any_rate": grasp_any / total,
        "grasp_success_all_episodes": grasp_all,
        "grasp_success_all_rate": grasp_all / total,
        "grasp_attempts_total": sum(
            int(episode["grasp_attempts"]) for episode in episodes
        ),
        "missed_grasp_attempts_total": sum(
            int(episode["missed_grasp_attempts"]) for episode in episodes
        ),
        "oracle_retry_count_total": sum(
            int(episode["oracle_retry_count"]) for episode in episodes
        ),
        "subtasks": subtask_aggregate,
    }
