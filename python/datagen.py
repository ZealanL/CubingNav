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

# NOTE: Scramble exp controls the distribution between the number of samples with different scramble move counts
#   Example: A scramble_exp of 2 will make scrambles with 1 more move twice as prevalent
def gen_batch(batch_size: int, device, min_moves: int, max_moves: int, scramble_exp: float) -> DataBatch:
    assert 0 < min_moves <= max_moves

    if 1:
        # Create a distribution of move count probabilities and sample from it to create our move counts for each scramble
        move_count_scales = (
            torch.arange(min_moves, max_moves + 1, device=device)
            .to(torch.float32).pow(scramble_exp)
        )
        move_count_distrib = move_count_scales / move_count_scales.sum()
        move_counts = torch.multinomial(move_count_distrib, batch_size, replacement=True) + min_moves

    cube_set = CubeSet(batch_size, device)
    scramble_moves = cube_set.scramble_all(move_counts)

    # Pull the last moves in every scramble
    last_scramble_moves = scramble_moves[torch.arange(batch_size, device=device), move_counts - 1]
    last_scramble_move_indices = last_scramble_moves[..., 0] * 3 + last_scramble_moves[..., 1]
    solve_dists = move_counts.to(torch.float32) / 20 # TODO: Ugly constant for god's number

    return DataBatch(
        inputs=cube_set.get_obs(),
        last_moves=last_scramble_move_indices,
        solve_dists=solve_dists,
    )

if __name__ == "__main__":
    batch = gen_batch(1000, "cuda", 1, 20, 2.0)
    print(batch.inputs.shape, batch.last_moves.shape, batch.solve_dists.shape)