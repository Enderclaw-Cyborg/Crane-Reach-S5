"""Tools that read demonstrations and report training results for train.py.

You can train without reading this file.
"""

import json

import numpy as np
import torch
from encoder import ENTITY_FEATURES, MOVE_FEATURES, TARGET_FEATURES, VERSION
from sandbox.crane import tile
from torch.nn import functional as F

# The entity columns that mark the acting unit's token and an enemy's token.
SELF_COLUMN, ENEMY_COLUMN = ENTITY_FEATURES.index("self"), ENTITY_FEATURES.index("enemy")


def validate_row(row, heads, index):
    required = {
        "seed",
        "unit_type",
        "driver",
        "encoder_version",
        "paths",
        "targets",
        "entities",
        "target_entity_indices",
        *(name for options, choice, _ in heads.values() for name in (options, choice)),
    }
    if not required <= row.keys() or row.get("encoder_version") != VERSION:
        raise ValueError(f"Row {index} has missing fields or a different encoder; recollect it.")
    entities = np.asarray(row["entities"], dtype=np.float32)
    if entities.ndim != 2 or entities.shape[1] != len(ENTITY_FEATURES) or len(entities) == 0:
        raise ValueError(f"Row {index} needs entity tokens, including the acting unit.")
    if not np.isfinite(entities).all() or entities[0, SELF_COLUMN] != 1:
        raise ValueError(f"Row {index} has invalid entity features or no acting-unit token.")
    for head, (options, choice, features) in heads.items():
        values = np.asarray(row[options], dtype=np.float32)
        label = row[choice]
        ids = row["paths" if head == "move" else "targets"]
        if (
            values.ndim != 2
            or values.shape[1] != len(features)
            or len(values) == 0
            or not np.isfinite(values).all()
            or type(label) is not int
            or not 0 <= label < len(values)
            or len(ids) != len(values)
        ):
            raise ValueError(f"Row {index} has invalid {options} or {choice}; recollect it.")
    pointers = row["target_entity_indices"]
    if len(pointers) != len(row["targets"]):
        raise ValueError(f"Row {index} has the wrong number of target pointers.")
    for target, pointer in zip(row["targets"], pointers, strict=True):
        # No target (0) points at -1. Every other target points at an enemy token after self.
        if type(pointer) is not int:
            valid = False
        elif target == 0:
            valid = pointer == -1
        else:
            valid = 0 < pointer < len(entities) and entities[pointer, ENEMY_COLUMN] == 1
        if not valid:
            raise ValueError(f"Row {index} has an invalid target pointer; targets must name enemy tokens.")


def load_rows(paths, heads):
    """Read demonstrations, preserving source lines and rejecting stale or malformed data."""
    rows = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.strip():
                row = json.loads(line)
                try:
                    validate_row(row, heads, line_number)
                except (ValueError, TypeError, KeyError, AttributeError) as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
                rows.append({**row, "_source": {"file": str(path), "line": line_number}})
    if not rows:
        raise ValueError("No decisions found. Collect demonstrations first.")
    return rows


def collate_rows(rows):
    """Pad variable-sized observations and options; True marks a real token or candidate."""
    batch = {}
    for name, width, mask_name in (
        ("entities", len(ENTITY_FEATURES), "entity_mask"),
        ("move_options", len(MOVE_FEATURES), "move_mask"),
        ("target_options", len(TARGET_FEATURES), "target_mask"),
    ):
        lengths = [len(row[name]) for row in rows]
        if not lengths or min(lengths) == 0:
            raise ValueError(f"Every decision needs at least one {name} row")
        # Each tensor has three dimensions: decision, entity or option, feature.
        # Fill unused rows with zeros and leave their mask False so the model ignores them.
        batch[name] = torch.zeros(len(rows), max(lengths), width)
        batch[mask_name] = torch.zeros(len(rows), max(lengths), dtype=torch.bool)
        for i, row in enumerate(rows):
            batch[name][i, : lengths[i]] = torch.as_tensor(row[name], dtype=torch.float32)
            batch[mask_name][i, : lengths[i]] = True
    # Padding adds rows at the end, so real target pointers still name the same entity rows.
    batch["target_entity_indices"] = torch.full(batch["target_mask"].shape, -1, dtype=torch.long)
    for i, row in enumerate(rows):
        batch["target_entity_indices"][i, : len(row["target_options"])] = torch.as_tensor(
            row["target_entity_indices"]
        )
    for head in ("move", "target"):
        batch[f"{head}_choice"] = torch.tensor([row[f"{head}_choice"] for row in rows], dtype=torch.long)
    return batch


def minibatch(data, indices):
    """Select decisions and trim padding to the largest observation in this batch."""
    batch = {key: values[indices] for key, values in data.items()}
    for mask, names in (
        ("entity_mask", ("entities", "entity_mask")),
        ("move_mask", ("move_options", "move_mask")),
        ("target_mask", ("target_options", "target_mask", "target_entity_indices")),
    ):
        count = int(batch[mask].sum(dim=1).max())
        for name in names:
            batch[name] = batch[name][:, :count]
    return batch


def model_inputs(batch):
    return {key: value for key, value in batch.items() if not key.endswith("_choice")}


def decision_loss(logits, labels, mask):
    """Average cross entropy over decisions, excluding padded candidates from softmax."""
    if not mask.any(dim=1).all() or not mask.gather(1, labels[:, None]).all():
        raise ValueError("Each decision must have a legal labeled candidate")
    # A label is the teacher's option index. Cross entropy rewards assigning probability
    # to that option among the real choices. A decision with just one option has zero loss.
    return F.cross_entropy(logits.masked_fill(~mask, -torch.inf), labels)


def agreement(rows, picks):
    """Return how often the learner agrees with the teacher: overall, by unit type, and by driver."""
    pairs = list(zip(rows, picks, strict=True))
    groups = {"all": pairs}
    for kind in ("footman", "archer", "cavalry"):
        groups[kind] = [(row, chosen) for row, chosen in pairs if row["unit_type"] == kind]
    if len({row["driver"] for row in rows}) > 1:
        for driver in ("teacher", "learner"):
            groups[f"{driver}_driven"] = [(row, chosen) for row, chosen in pairs if row["driver"] == driver]
    result = {}
    for name, selected in groups.items():
        if not selected:
            continue
        # Both predictions use the recorded inputs. Target inputs describe the teacher's
        # destination, even when the learner predicts a different tile.
        move = np.asarray([chosen["move"] == row["move_choice"] for row, chosen in selected])
        target = np.asarray([chosen["target"] == row["target_choice"] for row, chosen in selected])
        result[name] = {
            "rows": len(selected),
            "tile": float(move.mean()),
            "target": float(target.mean()),
            "whole": float((move & target).mean()),
        }
    return result


def random_baseline(rows):
    """Return the agreement we would expect from picking a legal option at random."""
    move = np.asarray([1 / len(row["move_options"]) for row in rows])
    target = np.asarray([1 / len(row["target_options"]) for row in rows])
    return {
        "tile": float(move.mean()),
        "target": float(target.mean()),
        "whole": float((move * target).mean()),
    }


def describe(row, move, target):
    """Show a choice as its tile offset from the unit, its path, and its target."""
    path = row["paths"][move]
    end = tile.at_path_end({"q": 0, "r": 0}, path)
    return {"tile_offset": [end["q"], end["r"]], "path": path, "target": row["targets"][target]}


def disagreements(rows, picks, per_kind=3):
    """List a few validation decisions per unit type where the learner and the teacher differ."""
    found, counts = [], {}
    for row, chosen in zip(rows, picks, strict=True):
        kind = row["unit_type"]
        if (chosen["move"], chosen["target"]) == (row["move_choice"], row["target_choice"]) or counts.get(
            kind, 0
        ) >= per_kind:
            continue
        counts[kind] = counts.get(kind, 0) + 1
        found.append(
            {
                **{key: row.get(key) for key in ("seed", "seat", "unit", "unit_type", "driver")},
                "teacher": describe(row, row["move_choice"], row["target_choice"]),
                "learner": describe(row, chosen["move"], chosen["target"]),
                **row["_source"],
            }
        )
    return found


def print_summary(report, output, report_path):
    print(f"Joint training: {report['epochs']} epochs, best epoch {report['best_epoch']}.")
    for head, name in (("move", "Tiles"), ("target", "Targets")):
        print(f"{name}: {report['heads'][head]['decisions']:,} decisions with choices to learn")
    print(
        f"Model: {report['parameters']:,} parameters, {report['model_bytes'] / 1000:.0f} KB, "
        f"trained in {report['fit_seconds']:.1f} s"
    )
    print("Validation agreement means how often the learner matches the teacher on reserved seeds.")
    print("Target agreement uses the teacher's tile. Whole-action agreement means both choices match.")
    print("Validation agreement     tile  target   whole  demonstrations")
    for name, values in report["validation_agreement"].items():
        print(
            f"  {name:<20} {values['tile']:6.1%}  {values['target']:6.1%}  {values['whole']:6.1%}"
            f"  {values['rows']:6,}"
        )
    baseline = report["random_choice_baseline"]
    print(
        f"  {'random legal choice':<20} {baseline['tile']:6.1%}  {baseline['target']:6.1%}"
        f"  {baseline['whole']:6.1%}"
    )
    print(f"Saved {output} and {report_path}. Agreement uses the same scoring as Agent.act.")
