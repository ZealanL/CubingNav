from dataclasses import dataclass

import torch
from cube_set import CubeSet

def get_obs_size():
    return CubeSet(1, "cpu").get_obs().size(-1)

@dataclass
class DataBatch:
    inputs: torch.Tensor
    target_outputs: torch.Tensor

def gen_batches(batch_size: int, min_moves: int, max_moves: int, device) -> list[DataBatch]:
    assert 0 < min_moves <= max_moves
    num_batches = (max_moves + 1) - min_moves

    all_inputs = []
    all_target_outputs = []
    for num_moves in range(min_moves, max_moves + 1): # We gotta make a batch per move
        cube_set = CubeSet(batch_size, device)

        # Make random turns
        if 1:
            random_turn_faces = torch.randint(0, 6, size=(num_moves, batch_size, 1), device=device)

            # We gotta make sure we don't generate two turns on a consecutive face
            # So, we go to 5 and add one if we reach the previous face
            # This ensures no repeated faces without trial and error
            for move_idx in range(1, num_moves):
                prev_faces = random_turn_faces[move_idx - 1]
                new_faces = torch.randint(0, 5, size=(batch_size, 1), device=device)

                # If new_face >= prev_face, increment by 1 to skip the previous face
                new_faces = torch.where(new_faces >= prev_faces, new_faces + 1, new_faces)

                random_turn_faces[move_idx] = new_faces

        random_turn_dirs = torch.randint(0, 3, size=(num_moves, batch_size, 1), device=device)
        random_turns = torch.concat([random_turn_faces, random_turn_dirs], dim=-1)

        # Apply to cube set
        cube_set.do_turns(random_turns)

        inputs = cube_set.get_obs()
        target_outputs = (
            # Calculate the flattened 0-18 move index
            random_turn_faces[-1] * 3 + random_turn_dirs[-1]
        ).squeeze(-1)

        all_inputs.append(inputs)
        all_target_outputs.append(target_outputs)

    all_inputs = torch.cat(all_inputs, dim=0)
    all_target_outputs = torch.cat(all_target_outputs, dim=0)
    assert all_inputs.size(0) == all_target_outputs.size(0)
    assert all_inputs.size(0) == batch_size * num_batches

    # Shuffle so that the batches don't each have their own move count
    shuffle_idc = torch.randperm(all_inputs.size(0))
    all_inputs = all_inputs[shuffle_idc]
    all_target_outputs = all_target_outputs[shuffle_idc]

    all_batches = []
    for i in range(num_batches):
        start = i * batch_size
        end = (i+1) * batch_size
        batch_inputs = all_inputs[start:end]
        batch_target_outputs = all_target_outputs[start:end]

        all_batches.append(DataBatch(batch_inputs, batch_target_outputs))

    return all_batches

if __name__ == "__main__":
    batches = gen_batches(2, 5, 10, "cuda")
    for batch in batches:
        print(batch.inputs.shape, batch.target_outputs.shape)