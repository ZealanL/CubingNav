import copy
import line_profiler
from dataclasses import dataclass

import torch
from line_profiler import profile

from cube_set import CubeSet, NUM_UNIQUE_MOVES

def get_obs_size() -> int:
    return CubeSet(1, "cpu").get_obs().size(-1)

@dataclass
class DataBatch:
    cubes_obs: torch.Tensor
    next_cubes_obs: torch.Tensor
    solved_mask: torch.Tensor
    move_counts: torch.Tensor

# NOTE: Scramble exp controls the distribution between the number of samples with different scramble move counts
#   Example: A scramble_exp of 2 will make scrambles with 1 more move twice as prevalent
@profile
def gen_batch(batch_size: int, device, min_moves: int, max_moves: int, scramble_exp: float) -> DataBatch:
    assert 0 <= min_moves <= max_moves

    # Create a distribution of move count probabilities and sample from it to create our move counts for each scramble
    scramble_move_range_size = (max_moves - min_moves) + 1
    move_count_scales = (
        # (exp^n) where n is (cur_moves - min_moves)
        torch.full((scramble_move_range_size,), scramble_exp, device=device).pow(
            torch.arange(min_moves, max_moves + 1, dtype=torch.float32, device=device)
        )
    )
    move_count_distrib = move_count_scales / move_count_scales.sum()
    move_counts = torch.multinomial(move_count_distrib, batch_size, replacement=True) + min_moves

    cube_set = CubeSet(batch_size, device)
    cube_set.scramble_all(move_counts)

    solved_mask = cube_set.get_solved_mask()

    cur_obs = cube_set.get_obs()

    # Generate repeating indices, equals "[([i] * batch_size) for i in range(NUM_UNIQUE_MOVES)]" flattened
    # Pattern style: [0, 0, 0, 1, 1, 1, 2, 2, 2, ...]
    move_indices = torch.arange(NUM_UNIQUE_MOVES, device=device).unsqueeze(-1).repeat(1, batch_size).flatten()

    # Tiles all the cubes NUM_UNIQUE_MOVES times
    # Pattern style: [0, 1, 2, 0, 1, 2, 0, 1, 2, ...]
    # This will allow us to apply each move to each original cube
    next_cube_set = copy.deepcopy(cube_set)
    next_cube_set.tile(NUM_UNIQUE_MOVES)

    next_cube_set.do_turn(move_indices)
    next_obs = next_cube_set.get_obs().reshape(NUM_UNIQUE_MOVES, batch_size, -1)
    next_obs = next_obs.transpose(0, 1) # to (batch_size, NUM_UNIQUE_MOVES, obs_size)

    assert next_obs.shape == (batch_size, NUM_UNIQUE_MOVES, cur_obs.size(-1))

    return DataBatch(
        cubes_obs=cur_obs,
        next_cubes_obs=next_obs,
        solved_mask=solved_mask,
        move_counts=move_counts
    )

if __name__ == "__main__":
    batch = gen_batch(1000, "cuda", 1, 20, 2.0)
    print(batch.inputs.shape, batch.last_moves.shape, batch.solve_dists.shape)