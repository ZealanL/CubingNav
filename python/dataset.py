import torch
import os
import numpy as np

class Dataset:
    def __init__(self, base_path: str, eval_num: int, num_token_types, max_move_count):
        np_tokens = np.load(os.path.join(base_path, "db_tokens.npy"))
        np_move_counts = np.load(os.path.join(base_path, "db_move_counts.npy"))

        assert np_tokens.shape[0] == np_move_counts.shape[0]
        assert 0 < eval_num < np_move_counts.shape[0]
        self.num = np_move_counts.shape[0] - eval_num
        self.eval_num = eval_num

        self._t_tokens = torch.from_numpy(np_tokens).unsqueeze(-1).to(torch.int32)
        self._t_move_counts = torch.from_numpy(np_move_counts).unsqueeze(-1)

        self.seq_length = self._t_tokens.shape[1]

        self.num_token_types = num_token_types
        assert self._t_tokens.max().item() < self.num_token_types

        self.max_move_count = max_move_count

    def get_move_count_frac(self, move_counts: torch.Tensor):
        return (move_counts - 1).to(torch.float32) / (self.max_move_count - 1)

    def sample_batch(self, batch_size, device) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randint(0, self.num - self.eval_num, (batch_size,)).to(torch.long)

        t_tokens = self._t_tokens[indices].to(torch.long).to(device)
        t_move_counts = self._t_move_counts[indices].to(device)
        return (t_tokens, self.get_move_count_frac(t_move_counts))

    def get_eval_batch(self, device) -> tuple[torch.Tensor, torch.Tensor]:
        eval_start_idx = self.num - self.eval_num
        t_tokens = self._t_tokens[eval_start_idx:].to(torch.long).to(device)
        t_move_count = self._t_move_counts[eval_start_idx:].to(device)
        return (t_tokens, self.get_move_count_frac(t_move_count))