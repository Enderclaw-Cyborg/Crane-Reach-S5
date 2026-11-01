"""Train a small attention policy on the teacher agent's decisions, on a GPU when one is available.

Each decision contributes a movement loss and a target loss. We reserve entire seeds for
validation, stop when validation loss stops improving, and restore the best epoch.
Run from the project root: python -m imitation.train runs/demonstrations.jsonl
"""

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import torch
from encoder import MOVE_FEATURES, TARGET_FEATURES, VERSION
from imitation_model import Policy, save_policy
from tqdm import tqdm

from imitation.experiment_tools import ROOT, RUNS
from imitation.train_tools import (
    agreement,
    collate_rows,
    decision_loss,
    disagreements,
    load_rows,
    minibatch,
    model_inputs,
    print_summary,
    random_baseline,
)

HEADS = {
    "move": ("move_options", "move_choice", MOVE_FEATURES),
    "target": ("target_options", "target_choice", TARGET_FEATURES),
}


def default_device():
    """Use a CUDA GPU, then Apple's MPS GPU on macOS, then the CPU."""
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def batch_loss(model, batch, device="cpu"):
    """Train both decisions together so they also improve the shared entity context."""
    batch = {key: values.to(device) for key, values in batch.items()}
    logits = model(**model_inputs(batch))
    losses = [decision_loss(logits[head], batch[f"{head}_choice"], batch[f"{head}_mask"]) for head in HEADS]
    return torch.stack(losses).sum()


def fit_model(
    training_rows,
    validation_rows,
    *,
    width=64,
    layers=2,
    heads=4,
    epochs=200,
    batch_size=64,
    threads=None,
    seed=0,
    device="cpu",
):
    """Fit joint decision losses and return the policy with the lowest validation loss."""
    threads = min(4, os.cpu_count() or 1) if threads is None else threads
    if min(epochs, batch_size, threads) < 1 or not training_rows or not validation_rows:
        raise ValueError("Need positive epochs, batch size, threads, and nonempty training/validation data")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    model = Policy(width, layers, heads)
    # Decisions have different numbers of entities and options. Padding lets us batch them;
    # masks distinguish real rows from the extra rows that only fill space.
    training, validation = collate_rows(training_rows), collate_rows(validation_rows)
    head_info = {}
    for head, (options, _, _) in HEADS.items():
        # Fit scaling only on real training candidates, never on validation data or padding.
        values = training[options][training[f"{head}_mask"]]
        scale = values.std(dim=0, correction=0)
        scale[scale == 0] = 1
        getattr(model, f"{head}_mean").copy_(values.mean(dim=0))
        getattr(model, f"{head}_scale").copy_(scale)
        choices = sum(len(row[options]) > 1 for row in training_rows)
        model.head_available[head] = bool(choices)
        head_info[head] = {"decisions": choices, "option_rows": len(values)}
    model.to(device)

    @torch.inference_mode()
    def validation_loss():
        # Measure held-out decisions without changing weights or recording gradients.
        model.eval()
        total = 0.0
        for start in range(0, len(validation_rows), batch_size):
            batch = minibatch(validation, slice(start, start + batch_size))
            total += float(batch_loss(model, batch, device)) * len(batch["move_choice"])
        return total / len(validation_rows)

    best_loss, patience_reference, best_epoch, unchanged = float("inf"), float("inf"), 0, 0
    best_state = deepcopy(model.state_dict())
    history = []
    if any(model.head_available.values()):
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        generator = torch.Generator().manual_seed(seed)
        with tqdm(
            range(1, epochs + 1), desc="Joint policy", unit="epoch", file=sys.stdout, dynamic_ncols=True
        ) as progress:
            for epoch in progress:
                model.train()
                # Shuffle whole decisions, keeping each decision's options and labels together.
                order = torch.randperm(len(training_rows), generator=generator)
                total, counted = 0.0, 0
                for start in range(0, len(training_rows), batch_size):
                    batch = minibatch(training, order[start : start + batch_size])
                    # A forced decision has zero loss. An entirely forced batch needs no update.
                    if not any((batch[f"{head}_mask"].sum(dim=1) > 1).any() for head in HEADS):
                        continue
                    optimizer.zero_grad(set_to_none=True)
                    loss = batch_loss(model, batch, device)
                    # backward computes how weights affect loss; step adjusts the weights.
                    loss.backward()
                    # Cap the combined gradient size before the optimizer uses it.
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    total += float(loss.detach()) * len(batch["move_choice"])
                    counted += len(batch["move_choice"])
                current = validation_loss()
                history.append({"epoch": epoch, "training_loss": total / counted, "validation_loss": current})
                progress.set_postfix(train=f"{history[-1]['training_loss']:.4f}", val=f"{current:.4f}")
                # Restart the 15-epoch wait after validation loss improves by more than 0.0001.
                # Keep every new minimum, including smaller improvements, for the final model.
                if current < patience_reference - 1e-4:
                    patience_reference, unchanged = current, 0
                else:
                    unchanged += 1
                if current < best_loss:
                    best_loss, best_epoch = current, epoch
                    best_state = deepcopy(model.state_dict())
                if unchanged >= 15:
                    break
    else:
        best_loss = validation_loss()
    # The last epoch may be worse than an earlier one. Restore the best validation result.
    model.load_state_dict(best_state)
    model.cpu().eval()
    return model, {
        "epochs": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "stop_reason": "no choices to learn"
        if not history
        else "validation loss stopped improving"
        if unchanged >= 15
        else "epoch limit reached",
        "history": history,
        "heads": head_info,
        "threads": threads,
        "device": str(device),
        "seed": seed,
        "batch_size": batch_size,
        "architecture": model.config,
    }


@torch.inference_mode()
def predict(model, rows, batch_size=64):
    model.eval()
    data = collate_rows(rows)
    picks = []
    for start in range(0, len(rows), batch_size):
        logits = model(**model_inputs(minibatch(data, slice(start, start + batch_size))))
        choices = {head: values.argmax(dim=1).tolist() for head, values in logits.items()}
        picks.extend(dict(zip(choices, pair, strict=True)) for pair in zip(*choices.values(), strict=True))
    return picks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "imitation-policy.pt")
    parser.add_argument("--validation-seeds", nargs="+", type=int, default=[8, 9])
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=default_device())
    args = parser.parse_args()
    if (
        min(args.width, args.layers, args.heads, args.epochs, args.batch_size, args.threads) < 1
        or args.width % args.heads
    ):
        parser.error("Sizes and threads must be positive; --width must be divisible by --heads")
    if args.output.suffix != ".pt":
        parser.error("--output must end in .pt")
    try:
        rows = load_rows(args.data, HEADS)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    # Both seats, all units, and both drivers of one seed stay on the same side of the split.
    training = [row for row in rows if row["seed"] not in args.validation_seeds]
    validation = [row for row in rows if row["seed"] in args.validation_seeds]
    if not training or not validation:
        parser.error("Need both training seeds and validation seeds in the data")
    print(f"Loaded {len(rows):,} demonstrations: {len(training):,} training, {len(validation):,} validation.")
    started = perf_counter()
    model, info = fit_model(
        training,
        validation,
        **{
            key: getattr(args, key)
            for key in ("width", "layers", "heads", "epochs", "batch_size", "threads", "seed", "device")
        },
    )
    picks = predict(model, validation, args.batch_size)
    save_policy(model, args.output)
    report = {
        **info,
        "encoder_version": VERSION,
        "training_seeds": sorted({row["seed"] for row in training}),
        "validation_seeds": sorted({row["seed"] for row in validation}),
        "training_rows": len(training),
        "validation_rows": len(validation),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "validation_agreement": agreement(validation, picks),
        "random_choice_baseline": random_baseline(validation),
        "fit_seconds": perf_counter() - started,
        "model_bytes": args.output.stat().st_size,
        "disagreements": disagreements(validation, picks),
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    report_path = RUNS / f"{args.output.stem}.training.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print_summary(report, args.output, report_path)


if __name__ == "__main__":
    main()
