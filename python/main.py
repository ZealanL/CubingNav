import torch
from torch.optim.lr_scheduler import StepLR
import wandb

import datagen
import export_model
from cube_set import CubeSet

from models import *
from soap import SOAP

DEVICE = "cuda"

MIN_SCRAMBLE_MOVES = 7
MAX_SCRAMBLE_MOVES = 18
SCRAMBLE_EXP = 1.8

BATCH_SIZE = 2048
MAX_ITRS = 1_000_000
LOG_INTERVAL = 50
START_LR = 2e-3
MIN_LR = 2e-4
LR_DECAY = 1e-4

VALUE_LOSS_WEIGHT = 0.5 # Scale how important value v.s. policy is

EXPORT_INTERVAL = 3000
EXPORT_PATH = "../checkpoint/model.onnx"

USE_WANDB = True

@torch.no_grad()
def calc_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    guesses = logits.argmax(dim=1)
    accuracy = (guesses == targets).to(torch.float32).mean()
    return accuracy

# Returns portion of cubes solved in the given num_attempt_moves
@torch.no_grad()
def test_policy_solve_ability(model: torch.nn.Module, num_cubes: int, num_scramble_moves: int, num_attempt_moves: int):
    cube_set = CubeSet(num_cubes, DEVICE)
    cube_set.scramble_all(
        torch.tensor([num_scramble_moves], device=DEVICE).repeat(num_cubes)
    )

    solved_cubes_mask = cube_set.get_solved_mask()
    for i in range(num_attempt_moves):
        obs = cube_set.get_obs()
        logits: torch.Tensor = model.forward(obs)[0]
        move_indices = logits.argmax(-1)

        # Invert the direction of the moves
        move_faces = move_indices // 3
        move_dirs = move_indices % 3

        invert_map = torch.tensor([1, 0, 2], device=DEVICE)
        inverted_move_dirs = invert_map[move_dirs]
        moves = torch.concat([
            move_faces.unsqueeze(-1),
            inverted_move_dirs.unsqueeze(-1),
        ], dim=-1)

        cube_set.do_turn(moves)

        solved_cubes_mask |= cube_set.get_solved_mask()

    return solved_cubes_mask.to(torch.float32).mean().cpu().item()

def test_policy_beam_searches(model: torch.nn.Module, num_cubes: int, num_scramble_moves: int, beam_width: int):
    cube_set = CubeSet(num_cubes * beam_width, DEVICE)
    scrambling_moves = cube_set.make_scrambling_moves(
        torch.tensor([num_scramble_moves], device=DEVICE).repeat(num_cubes * beam_width)
    )

    # Make them the same for each set beam search
    scrambling_moves = scrambling_moves[:num_cubes].repeat_interleave(beam_width, dim=0)
    pass


#################################################

def main():
    obs_size = datagen.get_obs_size()

    print("Creating model...")
    model = PVModel(
        seq_length=obs_size, num_token_types=CubeSet.NUM_TOKEN_TYPES, num_output_types=6*3,
        embedding_dim=16, shared_head_outputs=256
    )

    model.to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=START_LR)
    scheduler = StepLR(optim, step_size=1, gamma=(1 - LR_DECAY))

    wandb_run = None

    for itr in range(MAX_ITRS):
        batch = datagen.gen_batch(BATCH_SIZE, DEVICE, MIN_SCRAMBLE_MOVES, MAX_SCRAMBLE_MOVES, SCRAMBLE_EXP)
        (tb_inputs, tb_target_last_moves, tb_target_values) = (batch.inputs, batch.last_moves, batch.solve_dists)
        tb_policy_logits, tb_values = model(tb_inputs)

        tb_policy_loss = torch.nn.functional.cross_entropy(tb_policy_logits, tb_target_last_moves)
        tb_value_loss = (tb_values - tb_target_values).square().mean()
        tb_combined_loss = tb_policy_loss + tb_value_loss*VALUE_LOSS_WEIGHT
        tb_combined_loss.backward()
        optim.step()
        optim.zero_grad()

        last_lr = scheduler.get_last_lr()[0]
        if last_lr > MIN_LR:
            scheduler.step()

        if (itr % LOG_INTERVAL) == 0:

            #test_policy_beam_searches(model, 20, 16, 64)

            metrics = {
                "policy_loss": tb_policy_loss.detach().cpu().item(),
                "value_loss": tb_value_loss.detach().cpu().item(),
                "value_avg_error": (tb_values - tb_target_values).detach().abs().mean().cpu().item(),
                "policy_first_accuracy": calc_accuracy(tb_policy_logits, tb_target_last_moves).cpu().item(),
                "lr": last_lr,

                "policy_solve_4_in_4": test_policy_solve_ability(model, 512, 4, 4),
                "policy_solve_8_in_8": test_policy_solve_ability(model, 1024, 8, 8),
                "policy_solve_16_in_16": test_policy_solve_ability(model, 1024, 16, 16),
            }

            ###############

            print(f"[{itr}]:")
            for (k, v) in metrics.items():
                print(f"\t{k}: {v}")

            if USE_WANDB:
                if wandb_run is None:
                    wandb_run = wandb.init(project="CubeSolver", name="neorun")
                wandb_run.log(data=metrics, step=itr)

        if (itr > 0) and (itr % EXPORT_INTERVAL) == 0:
            # Export model
            print("Exporting model...")
            t_ref_inputs = torch.ones((1, obs_size), dtype=batch.inputs.dtype, device=DEVICE)
            export_model.save_to_onnx(model, t_ref_inputs, EXPORT_PATH)

if __name__ == '__main__':
    main()