import torch

class TransformerModel(torch.nn.Module):
    def __init__(self,
                 seq_length, num_token_types, num_output_types,
                 embedding_dim=128, tf_layers=4, num_heads=4, tf_ffn_dim=512, out_ffn_dim=256
    ):
        super().__init__()
        self.seq_length = seq_length
        self.num_output_types = num_output_types

        self.embeddings = torch.nn.Embedding(num_token_types, embedding_dim)
        self.pos_embeddings = torch.nn.Embedding(seq_length, embedding_dim)
        transformer_layer = torch.nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=tf_ffn_dim,
            dropout=0.0,
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            transformer_layer,
            num_layers=tf_layers,
        )

        self.output_ffn = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),

            torch.nn.Linear(seq_length * embedding_dim, out_ffn_dim),
            torch.nn.LayerNorm(out_ffn_dim),
            torch.nn.GELU(),

            # Use two outputs
            torch.nn.Linear(out_ffn_dim, num_output_types),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embeddings = self.embeddings(tokens.reshape(-1, self.seq_length))
        pos_embeddings = self.pos_embeddings(torch.arange(self.seq_length, device=tokens.device))

        transformer_outputs = self.transformer_encoder(embeddings + pos_embeddings)
        logits = self.output_ffn(transformer_outputs)
        return logits

class SimpleModel(torch.nn.Module):
    def __init__(self,
                 seq_length, num_token_types, num_output_types, embedding_dim=16):
        super().__init__()
        self.seq_length = seq_length
        self.num_output_types = num_output_types

        self.embeddings = torch.nn.Embedding(num_token_types, embedding_dim)
        self.ffn = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),

            torch.nn.Linear(seq_length * embedding_dim, 512),
            torch.nn.LayerNorm(512),
            torch.nn.ReLU(),

            torch.nn.Linear(512, 384),
            torch.nn.LayerNorm(384),
            torch.nn.ReLU(),

            torch.nn.Linear(384, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ReLU(),

            torch.nn.Linear(256, num_output_types),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embeddings = self.embeddings(tokens.reshape(-1, self.seq_length))
        logits = self.ffn(embeddings)
        return logits