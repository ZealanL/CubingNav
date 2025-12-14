import torch

class PVModel(torch.nn.Module):
    def __init__(self,
                 seq_length, num_token_types, num_output_types, embedding_dim=16, shared_head_outputs=512):
        super().__init__()
        self.seq_length = seq_length
        self.num_output_types = num_output_types

        self.embeddings = torch.nn.Embedding(num_token_types, embedding_dim)
        self.shared_head = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),

            torch.nn.Linear(seq_length * embedding_dim, shared_head_outputs * 2),
            torch.nn.LayerNorm(shared_head_outputs * 2),
            torch.nn.ReLU(),

            torch.nn.Linear(shared_head_outputs * 2, shared_head_outputs),
            torch.nn.LayerNorm(shared_head_outputs),
            torch.nn.ReLU(),
        )

        # Last move prediction
        self.policy_tail = torch.nn.Sequential(
            torch.nn.Linear(shared_head_outputs, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ReLU(),

            torch.nn.Linear(256, num_output_types)
        )

        # Value prediction (less params)
        self.value_tail = torch.nn.Sequential(
            torch.nn.Linear(shared_head_outputs, 128),
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),

            torch.nn.Linear(128, 1)
        )

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.embeddings(tokens.reshape(-1, self.seq_length))
        shared_outputs = self.shared_head(embeddings)
        policy_outputs = self.policy_tail(shared_outputs)
        value_outputs = self.value_tail(shared_outputs).squeeze(-1)
        return policy_outputs,value_outputs