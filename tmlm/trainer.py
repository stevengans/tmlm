import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from tmlm.model import TinyMetalLM
from tqdm import tqdm


class Trainer:
    def __init__(self, train_dataset, train_dataset_len, vocab_size, block_size, batch_size):
        self.batch_size = batch_size
        self.block_size = block_size
        self.train_dataset = train_dataset
        self.train_dataset_len = train_dataset_len
        self.max_epochs = 10
        self.betas = (0.9, 0.95)
        self.grad_norm_clip = 1.0
        self.warmup_tokens = 512 * 20
        self.learning_rate = 6e-4
        self.lr_warmup = 200
        self.final_tokens = 200 * train_dataset_len * self.block_size
        self.ckpt_path = None
        self.num_workers = 4
        self.tokens = 0
        self.tlm = TinyMetalLM(vocab_size, num_layers=4, dims=768, num_heads=16, checkpoint=False)
        mx.eval(self.tlm.parameters())
        n_params = sum(x.size for k, x in tree_flatten(self.tlm.parameters()) if "embedding" not in k)
        print(f"Training a transformer with {n_params / 1024 ** 2:.3f} M parameters")
        self.optimizer = optim.AdamW(learning_rate=self.learning_rate, weight_decay=1e-1)
        self.loss_and_grad = nn.value_and_grad(self.tlm, self.tlm.loss)

    def save_checkpoints(self):
        if self.ckpt_path is not None:
            self.tlm.save_weights(self.ckpt_path)

    def train_step(self, idx, inputs):
        inputs, targets = map(mx.array, inputs)
        loss, grads = self.loss_and_grad(inputs, targets)
        self.optimizer.learning_rate = min(1, idx / self.lr_warmup) * self.learning_rate
        self.optimizer.update(self.tlm, grads)
        del grads
        mx.eval(loss, self.tlm.parameters())
        return loss

    def train(self):
        train_pb_max_len = 1000
        losses = []
        pbar = tqdm(enumerate(self.train_dataset), total=train_pb_max_len)
        for epoch in range(self.max_epochs):
            for idx, inputs in pbar:
                loss = self.train_step(idx, inputs)
                loss_item = loss.item()
                losses.append(loss_item)
                pbar.set_description(f"Iteration: {idx + 1}, Loss: {loss_item:.3f}")
            train_loss = np.mean(losses)
            print(f"Epoch {epoch+1}: Mean loss {train_loss:.3f}.")
            self.save_checkpoints()

