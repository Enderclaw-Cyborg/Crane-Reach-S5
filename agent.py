"""Choose a reachable tile, then a target, with the trained PyTorch policy.

We encode the acting unit, visible units, and zones once. Movement candidates read that
context through attention. Each target candidate points directly at its enemy's token.
Both decisions use legal options, with the first option breaking equal-score ties.
"""

from pathlib import Path

import torch
from encoder import (
    describe_move,
    describe_target,
    encode_entities,
    move_options,
    target_entity_indices,
    target_options,
)
from imitation_model import load_policy


def option_tensor(rows):
    """Add a batch dimension: one decision, several rows, several features per row."""
    return torch.tensor(rows, dtype=torch.float32).unsqueeze(0)


@torch.inference_mode()
def choose_action(model, observation):
    # Read the battlefield once. Both scorers use this same context.
    entities, enemy_indices = encode_entities(observation)
    # A mask marks real rows. During play every row is real; training also has padding.
    entity_mask = torch.ones((1, len(entities)), dtype=torch.bool)
    context = model.encode(option_tensor(entities), entity_mask)

    moves = move_options(observation)
    targets = target_options(observation)
    if not moves or not targets:
        raise ValueError("The observation must contain legal movement and target options")
    # Score destinations, keeping one legal path to each. argmax keeps the first tie.
    move_rows = option_tensor([describe_move(observation, end) for _, end in moves])
    move_mask = torch.ones((1, len(moves)), dtype=torch.bool)
    scores = model.score_moves(context, entity_mask, move_rows, move_mask)
    path, end = moves[int(scores[0].argmax())]

    # Describe targets from the tile we actually chose. The pointers connect target IDs
    # to the enemy tokens in context; -1 selects the model's learned no-target value.
    target_rows = option_tensor([describe_target(observation, end, target) for target in targets])
    pointers = torch.tensor([target_entity_indices(observation, targets, enemy_indices)])
    target_mask = torch.ones((1, len(targets)), dtype=torch.bool)
    scores = model.score_targets(context, target_rows, pointers, target_mask)
    return {"path": path, "target": targets[int(scores[0].argmax())]}


class Agent:
    def __init__(self):
        # A model this small runs fastest on one CPU thread. The setting applies to the whole
        # process, so we set it before the first inference.
        torch.set_num_threads(1)
        self.model = load_policy(Path(__file__).with_name("imitation-policy.pt"))

    def reset(self, seed, observation):
        # This learner carries no private memory between activations. Its weights stay fixed.
        pass

    def act(self, observation):
        return choose_action(self.model, observation)
