import torch
from torch.optim.lr_scheduler import StepLR
import wandb

import datagen
import export_model
from cube_set import CubeSet, NUM_UNIQUE_MOVES

from models import *

DEVICE = "cuda"

MIN_SCRAMBLE_MOVES = 0
MAX_SCRAMBLE_MOVES = 18
SCRAMBLE_EXP_MIN = 1.0
SCRAMBLE_EXP_INC = 5e-5
SCRAMBLE_EXP_MAX = 1.7

BATCH_SIZE = 1024
MAX_ITRS = 1_000_000
LOG_INTERVAL = 50
START_LR = 1e-3
MIN_LR = 2e-4
LR_DECAY = 1e-4 # Disabled for now

UPDATE_TARGET_INTERVAL = 25

EXPORT_INTERVAL = 3000
EXPORT_PATH = "../checkpoint/model.onnx"

USE_WANDB = 1

@torch.no_grad()
def calc_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    guesses = logits.argmax(dim=1)
    accuracy = (guesses == targets).to(torch.float32).mean()
    return accuracy


@torch.no_grad()
def test_value_solve_ability(model: torch.nn.Module, num_cubes: int, num_scramble_moves: int, num_attempt_moves: int):
    model.eval()
    cube_set = CubeSet(num_cubes, DEVICE)
    cube_set.scramble_all(
        torch.tensor([num_scramble_moves], device=DEVICE).repeat(num_cubes)
    )

    solved_ever_mask = cube_set.get_solved_mask()

    for step in range(num_attempt_moves):
        next_obs_list = []
        for i in range(NUM_UNIQUE_MOVES):
            moves = torch.full((num_cubes,), i, device=DEVICE)
            cube_set.do_turn(moves)
            next_obs_list.append(cube_set.get_obs())
            cube_set.do_turn_inv(moves)

        all_next_obs = torch.stack(next_obs_list)
        next_vals = model(
            all_next_obs.view(-1, all_next_obs.size(-1))
        ).view(NUM_UNIQUE_MOVES, num_cubes)
        best_moves = next_vals.argmin(dim=0)  # (num_cubes,)
        cube_set.do_turn(best_moves)

        solved_ever_mask |= cube_set.get_solved_mask()

    model.train()
    return solved_ever_mask.to(torch.float32).mean().cpu().item()

#################################################

def main():
    obs_size = datagen.get_obs_size()

    print("Creating model...")
    embedding_dim = 24
    shared_head_outputs = 128
    model = CTGModel(
        seq_length=obs_size, num_token_types=CubeSet.NUM_TOKEN_TYPES,
        embedding_dim=embedding_dim, shared_head_outputs=shared_head_outputs
    )
    model.to(DEVICE)

    with torch.no_grad():
        target_model = CTGModel(
            seq_length=obs_size, num_token_types=CubeSet.NUM_TOKEN_TYPES,
            embedding_dim=embedding_dim, shared_head_outputs=shared_head_outputs
        )
        target_model.to(DEVICE)
        target_model.load_state_dict(model.state_dict())
        target_model.eval()

    optim = torch.optim.AdamW(model.parameters(), lr=START_LR)
    scheduler = StepLR(optim, step_size=1, gamma=(1 - LR_DECAY))

    scramble_exp = SCRAMBLE_EXP_MIN

    wandb_run = None

    for itr in range(MAX_ITRS):
        #print("Generating data...")
        batch = datagen.gen_batch(BATCH_SIZE, DEVICE, MIN_SCRAMBLE_MOVES, MAX_SCRAMBLE_MOVES, scramble_exp)
        tb_inputs = batch.cubes_obs # (b, obs_size)
        tb_next_inputs = batch.next_cubes_obs # (b, num_moves, obs_size)
        tb_solved_mask = batch.solved_mask

        # Cost estimate of the scramble
        #print("Estimating values (grad)...")
        tb_values = model(tb_inputs)

        # Determine minimum cost estimate of the neighbor cubes
        #print("Calculating targets (no grad)...")
        with torch.no_grad():
            tb_next_inputs_squeeze = tb_next_inputs.reshape(-1, tb_inputs.size(-1)) # (BATCH_SIZE * num_moves, obs_size)
            tb_next_values = target_model(tb_next_inputs_squeeze).reshape(BATCH_SIZE, -1)
            tb_min_next_values = tb_next_values.min(-1)[0]
            tb_value_targets = tb_min_next_values + 1.0

            # Make solved cubes have a target of zero
            tb_value_targets *= ~tb_solved_mask

        #print("Computing loss..")
        tb_loss = (tb_value_targets - tb_values).pow(2).mean()
        tb_loss.backward()
        optim.step()
        optim.zero_grad()

        scramble_exp = min(scramble_exp + SCRAMBLE_EXP_INC, SCRAMBLE_EXP_MAX)

        #print("Stepping...")

        last_lr = scheduler.get_last_lr()[0]
        if last_lr > MIN_LR:
            scheduler.step()

        if (itr % UPDATE_TARGET_INTERVAL) == 0:
            target_model.load_state_dict(model.state_dict())
            target_model.eval()

        if (itr % LOG_INTERVAL) == 0:
            with torch.no_grad():
                mean_val_err = (tb_value_targets - tb_values).abs().mean()
                val_accuracy = ((tb_value_targets - tb_values).abs() < 0.5).to(torch.float32).mean()
                val_mean = tb_values.mean()

                solved_portion = batch.solved_mask.to(torch.float32).mean()
                move_count_error = (tb_values - batch.move_counts.to(torch.float32)).abs().mean()

            metrics = {
                "loss": tb_loss.detach().cpu().item(),
                "mean_val_err": mean_val_err,
                "val_accuracy": val_accuracy,
                "val_mean": val_mean,
                "solved_portion": solved_portion,
                "move_count_error": move_count_error,

                "policy_solve_8_in_8": test_value_solve_ability(model, 512, 8, 8),
                "policy_solve_16_in_16": test_value_solve_ability(model, 512, 16, 16),
                "policy_solve_max_in_20": test_value_solve_ability(model, 512, 99, 20),

                "scramble_exp": scramble_exp,
                "lr": last_lr,
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
            t_ref_inputs = torch.ones((1, obs_size), dtype=batch.cubes_obs.dtype, device=DEVICE)
            export_model.save_to_onnx(model, t_ref_inputs, EXPORT_PATH)

if __name__ == '__main__':
    main()