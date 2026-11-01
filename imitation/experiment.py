"""Record the teacher agent's choices, then compare it with the learner in complete episodes.

Each recorded decision is a demonstration. During collection, we describe each reachable tile
and possible target with features, then mark the teacher's choices as labels. With --driver
learner, the learner plays while the teacher labels each situation it reaches. Evaluation
plays both against Naive on the same seeds and seats, recording scores and act times.

Start with demonstration(), which builds one record. experiment_tools.py loads the teacher and
runs the episodes. Run this script from the project root, for example:

    python -m imitation.experiment collect --episodes 10
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

import encoder
from sandbox.crane import tile

from imitation.experiment_tools import ROOT, RUNS, action_labels, episode, load_teacher


def demonstration(observation, action):
    """Record one demonstration, describing each legal option and labeling the teacher's choices.

    A record holds the inputs and the labels for one unit's activation:
    - entities describes the acting unit, visible units, and zones. target_entity_indices
      connects each target option to its enemy's row, with -1 for no target.
    - move_options has one row of MOVE_FEATURES numbers per reachable tile, and move_choice
      marks the teacher's tile. Different paths to the same tile count as the same choice.
    - target_options has one row of TARGET_FEATURES numbers per legal target, measured from the
      teacher's tile, and target_choice marks the teacher's target.
    - To help you inspect disagreements later, paths and targets keep the IDs behind the option
      rows, while path and target keep the teacher's action.
    """
    # To label the teacher's chosen tile and target, we first check that its action is legal.
    path, target = action_labels(observation, action)

    # Find the tile where the teacher's path ends. Its position among the options is the tile label.
    unit = observation["observation"]["self"]
    moves = encoder.move_options(observation)
    ends = [end for _, end in moves]
    chosen_end = tile.at_path_end(unit["position"], path)

    # For the inputs, we describe every option with the same encoder the learner uses in play.
    targets = encoder.target_options(observation)
    entities, enemy_indices = encoder.encode_entities(observation)
    return {
        "unit": unit["unit_id"],
        "unit_type": unit["type"],
        "encoder_version": encoder.VERSION,
        "path": path,
        "target": target,
        "entities": entities,
        "target_entity_indices": encoder.target_entity_indices(observation, targets, enemy_indices),
        "paths": [path_id for path_id, _ in moves],
        "move_options": [encoder.describe_move(observation, end) for end in ends],
        "move_choice": ends.index(chosen_end),
        # We describe targets from the teacher's tile, where it chose its target.
        "targets": targets,
        "target_options": [
            encoder.describe_target(observation, chosen_end, target_id) for target_id in targets
        ],
        "target_choice": targets.index(target),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("collect", "evaluate"))
    parser.add_argument("--start", type=int, help="first seed; default 0 for collection, 1000 for evaluation")
    parser.add_argument("--episodes", type=int, default=10, help="number of seeds; each runs both seats")
    parser.add_argument(
        "--teacher",
        default="teacher:Agent",
        help="teacher module:Class, e.g. teacher:Agent; its choices are what the learner learns",
    )
    parser.add_argument(
        "--driver",
        choices=("teacher", "learner"),
        default="teacher",
        help="collection only: who plays while the teacher labels; learner needs imitation-policy.pt",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.start is None:
        args.start = 0 if args.command == "collect" else 1000
    try:
        teacher_factory = load_teacher(args.teacher)
    except (ImportError, AttributeError, ValueError) as exc:
        parser.error(str(exc))
    if args.command == "collect" and args.driver == "learner" and not (ROOT / "imitation-policy.pt").exists():
        parser.error("--driver learner needs imitation-policy.pt; train on teacher demonstrations first")
    output = args.output or RUNS / (
        "demonstrations.jsonl" if args.command == "collect" else "evaluation.json"
    )
    if output.exists():
        parser.error(f"{output} already exists; choose another --output or move the old file")

    # Collection records teacher choices while the selected driver plays.
    # For comparison, evaluation plays both agents on the same seeds and seats.
    rows, results = [], []
    kinds = (args.driver,) if args.command == "collect" else ("teacher", "learner")
    for seed in range(args.start, args.start + args.episodes):
        for seat in (0, 1):
            for kind in kinds:
                if args.command == "collect":
                    result = episode(kind, seed, seat, teacher_factory, rows, record=demonstration)
                else:
                    result = episode(kind, seed, seat, teacher_factory)
                result["teacher"] = args.teacher
                results.append(result)
                print(json.dumps(result), flush=True)

    # To preserve any file created while the experiment ran, we use "x" to require a new file.
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        if args.command == "collect":
            for row in rows:
                row["teacher"] = args.teacher
                stream.write(json.dumps(row) + "\n")
            print(
                f"Saved {len(rows)} decisions to {output}: {dict(Counter(row['unit_type'] for row in rows))}"
            )
        else:
            stream.write(json.dumps(results, indent=2) + "\n")
            for kind in kinds:
                selected = [row for row in results if row["agent"] == kind]
                print(f"{kind}: mean score {mean(row['score'] for row in selected):.2f}")


if __name__ == "__main__":
    main()
