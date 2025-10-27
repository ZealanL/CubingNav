import torch
from torch.optim.lr_scheduler import StepLR

import export_model
import wandb

from models import *
import dataset

DEVICE = "cuda"
DATABASE_PATH = "../data/"

BATCH_SIZE = 2048
MAX_ITRS = 1_000_000
LOG_INTERVAL = 10
START_LR = 1e-3
MIN_LR = 2e-4
LR_DECAY = 2e-3

EXPORT_INTERVAL = 1000
EXPORT_PATH = "../checkpoint/model.onnx"

USE_WANDB = True

def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits, targets)

@torch.no_grad()
def calc_expect_err(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = torch.nn.functional.softmax(logits, dim=-1)
    expectation = (probs * torch.arange(logits.shape[-1], device=DEVICE).unsqueeze(0)).sum(-1)
    expectation_avg_err = (expectation - targets.flatten().to(torch.float32)).abs().mean()
    return expectation_avg_err

def main():
    print("Loading dataset...")
    db = dataset.Dataset(
        base_path=DATABASE_PATH,
        eval_num=BATCH_SIZE * 2,
        num_token_types = 8*8*3 + 12*12*2 + 1,
        max_move_count = 20
    )
    print(" > Size: ", db.num)

    print("Creating model...")
    model = SimpleModel(
        seq_length=db.seq_length, num_token_types=db.num_token_types, num_output_types=(db.max_move_count+1),
        embedding_dim=256, dropout=0.0
    )
    model.to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=START_LR)
    scheduler = StepLR(optim, step_size=1, gamma=(1 - LR_DECAY))

    wandb_run = None

    for itr in range(MAX_ITRS):
        (tb_inputs, tb_target_outputs) = db.sample_batch(BATCH_SIZE, DEVICE)
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
            metrics["train_expect_err"] = calc_expect_err(tb_outputs, tb_target_outputs).cpu().item()

            with torch.no_grad():
                (te_inputs, te_target_outputs) = db.get_eval_batch(DEVICE)
                te_outputs = model.forward(te_inputs)
                metrics["eval_expect_err"] = calc_expect_err(te_outputs, te_target_outputs).cpu().item()

                # Get a sense of eval accuracy
                print(te_target_outputs[:30].detach().cpu().tolist())
                print(te_outputs.argmax(-1)[:30].detach().cpu().tolist())
            metrics["lr"] = last_lr

            ###############

            print(f"[{itr}]:")
            for (k, v) in metrics.items():
                print(f"\t{k}: {v}")

            if USE_WANDB:
                if wandb_run is None:
                    wandb_run = wandb.init(project="CubeSolver", name="run")
                wandb_run.log(metrics, step=itr)

        if (itr > 0) and (itr % EXPORT_INTERVAL) == 0:
            # Export model
            print("Exporting model...")
            t_ref_inputs = torch.ones((1, db.seq_length), dtype=torch.long, device=DEVICE)
            export_model.save_to_onnx(model, t_ref_inputs, EXPORT_PATH)

if __name__ == '__main__':
    main()