"""Tools that load the teacher and run episodes for experiment.py. You can run the experiment
without reading this file.

To match how the template runs a submission, each unit gets its own teacher with its own private
state. RecordingPlayer then records what happens at each activation.
"""

from collections.abc import Mapping
from copy import deepcopy
from importlib import import_module
from numbers import Integral
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
from sandbox.env import META, default_action, extract_overlay, extract_overlay_static, make_env
from sandbox.env.skirmish_crane.naive import Agent as Naive
from sandbox.harness.environment import EnvironmentEntry, resolve_layout, resolve_parameters
from sandbox.harness.manifest import _evict_modules_from_roots
from sandbox.harness.session import REASON_TERMINATED, REASON_TRUNCATED, AgentPlayer, run_episode
from sandbox.season import load_season_settings

# The project root holds teacher.py, agent.py, imitation-policy.pt, and season.json.
ROOT = Path(__file__).resolve().parent.parent
# Recorded data, training reports, and evaluations go here. .gitignore keeps it out of the repository.
RUNS = ROOT / "runs"


def action_labels(observation, action):
    """Return the teacher's path and target, and raise an error if either one is illegal."""
    if not isinstance(action, Mapping):
        raise ValueError("Teacher act() must return a dict with path and target")
    labels = []
    for name in ("path", "target"):
        legal = [index for index, allowed in enumerate(observation["action_mask"][name]) if allowed]
        value = action.get(name)
        if isinstance(value, bool) or not isinstance(value, Integral) or value not in legal:
            raise ValueError(f"Teacher returned illegal {name}={value!r}; fix the teacher before collecting")
        labels.append(int(value))
    return labels


def load_teacher(spec):
    module, separator, name = spec.partition(":")
    if not separator or not module or not name:
        raise ValueError("Use --teacher module:Class, for example teacher:Agent")

    def load_class():
        # Like the template loader, give each unit a fresh class, module, and helper state.
        _evict_modules_from_roots((ROOT,))
        try:
            factory = getattr(import_module(module), name)
        except ModuleNotFoundError as exc:
            if exc.name == module:
                raise ValueError(
                    f"Cannot find teacher module {module!r}. Copy your agent's agent.py into this "
                    "folder as teacher.py, together with its helper files, or name another module "
                    "with --teacher module:Class."
                ) from exc
            raise
        if not callable(factory) or not all(
            callable(getattr(factory, method, None)) for method in ("reset", "act")
        ):
            raise ValueError("The teacher class must provide reset(seed, observation) and act(observation)")
        return factory

    load_class()  # Check for a missing teacher before starting any episodes.
    return lambda: load_class()()


class RecordingPlayer:
    """Give one unit its own agent and record the teacher's choice at each activation.

    In teacher-driven collection, the teacher plays and we record its action. In learner-driven
    collection, the learner plays and a separate teacher, the labeler, receives a copy of the
    same observation. We record the teacher's answer and execute the learner's action. The
    record function turns the observation and teacher's action into one demonstration.
    """

    def __init__(self, agent, seed, seat, rows, timings, labeler=None, record=None):
        self.agent, self.seed, self.seat = agent, seed, seat
        self.rows, self.timings, self.labeler, self.record = rows, timings, labeler, record
        self.first_act_ms = None

    def reset(self, seed, observation):
        if self.labeler is not None:
            self.labeler.reset(seed, deepcopy(observation))
        self.agent.reset(seed, observation)

    def __getattr__(self, name):
        return getattr(self.agent, name)

    def act(self, observation):
        if self.rows is not None:
            # Copy the situation before either agent has a chance to modify the observation.
            before = deepcopy(observation)
            label = None if self.labeler is None else self.labeler.act(deepcopy(observation))

        start = perf_counter()
        action = self.agent.act(observation)
        elapsed = 1000 * (perf_counter() - start)
        self.timings.append(elapsed)
        if self.first_act_ms is None:
            self.first_act_ms = elapsed

        if self.rows is not None and self.record is not None:
            row = {
                "seed": self.seed,
                "seat": self.seat,
                "driver": "teacher" if self.labeler is None else "learner",
            }
            row.update(self.record(before, action if self.labeler is None else label))
            self.rows.append(row)
        return action


def episode(kind, seed, seat, teacher_factory, rows=None, record=None):
    """Play one episode with kind in seat against Naive. If rows is given, record teacher choices."""
    season = load_season_settings(ROOT, META)
    parameters = resolve_parameters(META, {} if season is None else season.parameters)
    layout = resolve_layout(META, parameters)
    own = layout.seats[seat].players
    timings = []
    if kind == "teacher":
        factory = teacher_factory
    else:
        # We load the learner only for learner episodes. Teacher collection needs no saved model.
        from agent import Agent

        factory = Agent

    # When the learner drives collection, a teacher labels each situation without acting.
    labeling = rows is not None and kind == "learner"
    recorders = {
        player: RecordingPlayer(
            factory(), seed, seat, rows, timings, teacher_factory() if labeling else None, record
        )
        for player in own
    }
    players = {player: recorders[player] if player in recorders else Naive() for player in layout.players}
    # The season's time limits, or the environment's own where season.json sets none.
    step_limit = season.decision_limit_ms if season and season.decision_limit_ms else META.step_limit_ms
    game_limit = season.game_limit_ms if season and season.game_limit_ms else META.episode_limit_ms
    if labeling:
        # The teacher labels inside the learner's timed activation, so give each a full budget.
        step_limit, game_limit = 2 * step_limit, 2 * game_limit
    env = make_env(parameters)
    try:
        entry = EnvironmentEntry(
            META, lambda parameters: env, default_action, extract_overlay, extract_overlay_static
        )
        result = run_episode(
            entry,
            {player: AgentPlayer(agent) for player, agent in players.items()},
            seed=seed,
            parameters=parameters,
            step_limit_ms=step_limit,
            episode_limit_ms=game_limit,
        )
    finally:
        env.close()
    own_timeouts = sum(result.step_timeouts[player] for player in own)
    if rows is not None and (
        own_timeouts
        or result.failed_player is not None
        or result.reason not in (REASON_TERMINATED, REASON_TRUNCATED)
    ):
        raise ValueError(
            f"Collection aborted for seed {seed}, seat {seat}: "
            f"{own_timeouts} {kind} timeouts, reason={result.reason}, "
            f"failed_player={result.failed_player}. No data was saved."
        )
    first = [unit.first_act_ms for unit in recorders.values() if unit.first_act_ms is not None]
    # Exclude each unit's first call when reporting warmed-up action latency.
    warm = list(timings)
    for elapsed in first:
        warm.remove(elapsed)
    return {
        "agent": kind,
        "seed": seed,
        "seat": seat,
        "act_calls": len(timings),
        "p95_act_ms": float(np.percentile(timings, 95)) if timings else None,
        "warm_mean_act_ms": mean(warm) if warm else None,
        "warm_p95_act_ms": float(np.percentile(warm, 95)) if warm else None,
        "first_act_ms": first,
        "score": mean(result.scores[player] for player in own),
        "mean_act_ms": mean(timings) if timings else None,
        "max_act_ms": max(timings) if timings else None,
        "own_timeouts": own_timeouts,
        "opponent_timeouts": sum(
            count for player, count in result.step_timeouts.items() if player not in own
        ),
        "reason": result.reason,
        "failed_player": result.failed_player,
    }
