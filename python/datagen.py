from dataclasses import dataclass

import torch
from cube_set import CubeSet

def get_obs_size():
    return CubeSet(1, "cpu").get_obs().size(-1)

@dataclass
class DataBatch:
    inputs: torch.Tensor
    last_moves: torch.Tensor
    solve_dists: torch.Tensor

def gen_batches(batch_size: int, min_moves: int, max_moves: int, device) -> list[DataBatch]:
    assert 0 < min_moves <= max_moves
    num_batches = (max_moves + 1) - min_moves

    all_inputs = []
    all_last_moves = []
    all_solve_dists = []
    for num_moves in range(min_moves, max_moves + 1): # We gotta make a batch per move
        cube_set = CubeSet(batch_size, device)

        random_turns = cube_set.scramble_all(num_moves)

        inputs = cube_set.get_obs()
        last_moves = (
            # Calculate the flattened 0-18 move index
            random_turns[-1, :, 0] * 3 + random_turns[-1, :, 1]
        ).squeeze(-1)

        all_inputs.append(inputs)
        all_last_moves.append(last_moves)
        all_solve_dists.append(num_moves / 20)

    all_inputs = torch.cat(all_inputs, dim=0)
    all_last_moves = torch.cat(all_last_moves, dim=0)
    all_solve_dists = (
        torch.tensor(all_solve_dists, dtype=torch.float32, device=device)
        .repeat_interleave(batch_size, dim=0) # Repeat for every batch
    )


    assert all_inputs.size(0) == all_last_moves.size(0)
    assert all_solve_dists.size(0) == all_solve_dists.size(0)
    assert all_inputs.size(0) == batch_size * num_batches

    # Shuffle so that the batches don't each have their own move count
    shuffle_idc = torch.randperm(all_inputs.size(0))
    all_inputs = all_inputs[shuffle_idc]
    all_last_moves = all_last_moves[shuffle_idc]
    all_solve_dists = all_solve_dists[shuffle_idc]

    all_batches = []
    for i in range(num_batches):
        start = i * batch_size
        end = (i+1) * batch_size
        batch_inputs = all_inputs[start:end]
        batch_last_moves = all_last_moves[start:end]
        batch_solve_dists = all_solve_dists[start:end]

        all_batches.append(
            DataBatch(batch_inputs, batch_last_moves, batch_solve_dists)
        )

    return all_batches

if __name__ == "__main__":
    batches = gen_batches(100, 5, 10, "cuda")
    for batch in batches:
        print(batch.inputs.shape, batch.last_moves.shape, batch.solve_dists.shape)