import torch

class PVModel(torch.nn.Module):
    def __init__(self,
                 seq_length, num_token_types, num_output_types, embedding_dim=16):
        super().__init__()
        self.seq_length = seq_length
        self.num_output_types = num_output_types

        self.embeddings = torch.nn.Embedding(num_token_types, embedding_dim)
        SHARED_OUTPUTS = 512
        self.shared_head = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),

            torch.nn.Linear(seq_length * embedding_dim, SHARED_OUTPUTS * 2),
            torch.nn.LayerNorm(SHARED_OUTPUTS * 2),
            torch.nn.ReLU(),

            torch.nn.Linear(SHARED_OUTPUTS * 2, SHARED_OUTPUTS),
            torch.nn.LayerNorm(SHARED_OUTPUTS),
            torch.nn.ReLU(),
        )

        # Last move prediction
        self.policy_tail = torch.nn.Sequential(
            torch.nn.Linear(SHARED_OUTPUTS, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ReLU(),

            torch.nn.Linear(256, num_output_types)
        )

        # Value prediction (less params)
        self.value_tail = torch.nn.Sequential(
            torch.nn.Linear(SHARED_OUTPUTS, 128),
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),

            torch.nn.Linear(128, 1)
        )

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.embeddings(tokens.reshape(-1, self.seq_length))
        shared_outputs = self.shared_head(embeddings)
        policy_outputs = self.policy_tail(shared_outputs)
        value_outputs = self.value_tail(shared_outputs)
        return policy_outputs,value_outputs