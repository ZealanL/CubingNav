import torch
from torch.optim.lr_scheduler import StepLR
import wandb

from model import Model
import dataset

DEVICE = "cuda"
DATABASE_PATH = "../data/"
BATCH_SIZE = 2048
MAX_ITRS = 1_000_000
LOG_INTERVAL = 10
START_LR = 1e-3
MIN_LR = 2e-4
LR_DECAY = 5e-3

USE_WANDB = True

def loss_fn(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    errors = (outputs - targets).abs()
    return errors.pow(2).mean()

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
    model = Model(
        seq_length=db.seq_length,
        num_token_types=db.num_token_types,
        embedding_dim=128,
        num_heads=1,
        tf_ffn_dim=256,
        out_ffn_dim=256,
        tf_dropout=0.00,
        out_ffn_dropout=0.00
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

            with torch.no_grad():
                (te_inputs, te_target_outputs) = db.get_eval_batch(DEVICE)
                te_outputs = model.forward(te_inputs)
                te_loss = loss_fn(te_outputs, te_target_outputs)
                metrics["eval_loss"] = te_loss.detach().cpu().item()

            metrics["lr"] = last_lr

            ###############

            print(f"[{itr}]:")
            for (k, v) in metrics.items():
                print(f"\t{k}: {v}")

            if USE_WANDB:
                if wandb_run is None:
                    wandb_run = wandb.init(project="CubeSolver", name="run")
                wandb_run.log(metrics, step=itr)

if __name__ == '__main__':
    main()