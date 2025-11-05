import torch
from torch.optim.lr_scheduler import StepLR
import wandb

import datagen
import export_model
from cube_set import CubeSet

from models import *

DEVICE = "cuda"

MIN_SCRAMBLE_MOVES = 8
MAX_SCRAMBLE_MOVES = 18

BATCH_SIZE = 4096
MAX_ITRS = 1_000_000
LOG_INTERVAL = 50
START_LR = 2e-3
MIN_LR = 1e-4
LR_DECAY = 6e-4

EXPORT_INTERVAL = 2000
EXPORT_PATH = "../checkpoint/model.onnx"

USE_WANDB = True

def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits, targets)

@torch.no_grad()
def calc_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    guesses = logits.argmax(dim=1)
    accuracy = (guesses == targets).to(torch.float32).mean()
    return accuracy

# Returns portion of cubes solved in the given num_attempt_moves
@torch.no_grad()
def test_solving_ability(model: torch.nn.Module, num_cubes: int, num_moves: int, num_attempt_moves: int):
    cube_set = CubeSet(num_cubes, DEVICE)
    cube_set.scramble_all(num_moves)

    solved_cubes_mask = cube_set.get_solved_mask()
    for i in range(num_attempt_moves):
        obs = cube_set.get_obs()
        logits: torch.Tensor = model.forward(obs)
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

def main():
    obs_size = datagen.get_obs_size()

    print("Creating model...")
    if True:
        model = SimpleModel(
            seq_length=obs_size, num_token_types=CubeSet.NUM_TOKEN_TYPES, num_output_types=6*3,
            embedding_dim=48
        )
    else:
        model = TransformerModel(
            seq_length=obs_size, num_token_types=CubeSet.NUM_TOKEN_TYPES, num_output_types=6 * 3,
            embedding_dim=512, tf_layers=1, num_heads=4,tf_ffn_dim=1024, out_ffn_dim=128
        )
    model.to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=START_LR)
    scheduler = StepLR(optim, step_size=1, gamma=(1 - LR_DECAY))

    wandb_run = None

    queued_batches = []
    for itr in range(MAX_ITRS):
        if len(queued_batches) == 0:
            # Queue up more batches!
            queued_batches = datagen.gen_batches(BATCH_SIZE, MIN_SCRAMBLE_MOVES, MAX_SCRAMBLE_MOVES, DEVICE)

        batch = queued_batches.pop(0)
        (tb_inputs, tb_target_outputs) = (batch.inputs, batch.target_outputs)
        tb_outputs = model(tb_inputs)

        tb_loss = loss_fn(tb_outputs, tb_target_outputs)
        tb_loss.backward()
        optim.step()
        optim.zero_grad()

        last_lr = scheduler.get_last_lr()[0]
        if last_lr > MIN_LR:
            scheduler.step()

        if (itr % LOG_INTERVAL) == 0:
            metrics = {}

            metrics["train_loss"] = tb_loss.detach().cpu().item()
            metrics["train_accuracy"] = calc_accuracy(tb_outputs, tb_target_outputs).cpu().item()
            metrics["lr"] = last_lr

            metrics["solve_4_4"]  = test_solving_ability(model, 1024, 4, 4)
            metrics["solve_8_8"] = test_solving_ability(model, 4096, 8, 8)
            metrics["solve_16_16"] = test_solving_ability(model, 4096, 16, 16)

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