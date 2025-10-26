import torch

class Model(torch.nn.Module):
    def __init__(self,
                 seq_length, num_token_types,
                 embedding_dim=128, num_heads=4, tf_ffn_dim=512, out_ffn_dim=256, tf_dropout=0.05, out_ffn_dropout=0.05
    ):
        super().__init__()
        self.seq_length = seq_length

        self.embeddings = torch.nn.Embedding(num_token_types, embedding_dim)
        self.transformer_layer = torch.nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=tf_ffn_dim,
            dropout=tf_dropout,
        )

        self.output_ffn = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),

            torch.nn.Linear(seq_length * embedding_dim, out_ffn_dim),
            torch.nn.LayerNorm(out_ffn_dim),
            torch.nn.Dropout(out_ffn_dropout),
            torch.nn.GELU(),

            torch.nn.Linear(out_ffn_dim, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embeddings = self.embeddings(tokens.reshape(-1, self.seq_length))
        transformer_outputs = self.transformer_layer(embeddings)
        outputs = self.output_ffn(transformer_outputs)

        return outputs