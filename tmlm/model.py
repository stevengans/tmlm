import mlx.core as mx
import mlx.nn as nn


class TinyMetalLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        dims: int,
        num_heads: int,
        checkpoint: bool,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, dims)
        self.pe = nn.SinusoidalPositionalEncoding(dims)
        self.transformer = nn.TransformerEncoder(
            num_layers, dims, num_heads, norm_first=True, checkpoint=checkpoint
        )
        self.out_proj = nn.Linear(dims, vocab_size)

    def __call__(self, x, pred=False):
        L = x.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
        x = self.embedding(x)
        x = x + self.pe(mx.arange(L))
        x = self.transformer(x, mask)
        x = self.out_proj(x)
        if pred:
            x = mx.mean(x, axis=2)
            return nn.relu(x)
        return x

    def loss(self, x, y, reduce=True):
        logits = self(x)
        losses = nn.losses.cross_entropy(logits, y)
        return mx.mean(losses) if reduce else mx.mean(losses, axis=(-1, -2))


class Predictor(nn.Module):
    def __init__(
        self,
        num_layers: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int
    ):
        super().__init__()
        layer_sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        self.layers = [nn.Linear(i_dim, o_dim) for i_dim, o_dim in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.sigmoid = nn.Sigmoid()

    def __call__(self, x):
        for l in self.layers[:-1]:
            x = mx.maximum(l(x), 0.0)
        x = self.layers[-1](x)
        return self.sigmoid(x)

    def loss_fn(self, X, y):
        return mx.mean(nn.losses.cross_entropy(self(X), y))
