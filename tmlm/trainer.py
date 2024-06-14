import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from tmlm.model import TinyMetalLM, Predictor
from tqdm import tqdm


class Trainer:
    def __init__(self, train_dataset, block_size, batch_size, predictor):
        self.batch_size = batch_size
        self.block_size = block_size
        self.train_dataset = train_dataset
        self.train_dataset_len = len(train_dataset)
        self.vocab_size = train_dataset.vocab_size
        self.max_epochs = 20
        self.mini_epochs = 10
        self.learning_rate = 6e-4
        self.lr_warmup = 200
        self.ckpt_path = "model.safetensors"
        self.lrp_path = "lrp.safetensors"
        self.predictor = predictor
        self.tlm = TinyMetalLM(vocab_size=self.vocab_size, num_layers=4, dims=768, num_heads=16, checkpoint=False)
        mx.eval(self.tlm.parameters())
        n_params = sum(x.size for k, x in tree_flatten(self.tlm.parameters()) if "embedding" not in k)
        print(f"Training a Transformer with {n_params / 1024 ** 2:.3f} M parameters")
        if predictor:
            self.lrp = Predictor(num_layers=4, input_dim=256, hidden_dim=128, output_dim=block_size)
            mx.eval(self.lrp.parameters())
            n_params = sum(x.size for k, x in tree_flatten(self.lrp.parameters()) if "embedding" not in k)
            print(f"Training a Low Rank Predictor with {n_params / 1024 ** 2:.3f} M parameters")
            self.lrp_loss = nn.value_and_grad(self.lrp, self.lrp.loss_fn)
            self.lrp_opt = optim.SGD(learning_rate=1e-1)
        self.tlm_opt = optim.AdamW(learning_rate=self.learning_rate, weight_decay=1e-1)
        self.tlm_loss = nn.value_and_grad(self.tlm, self.tlm.loss)

    def save_checkpoints(self, pred=False):
        if self.ckpt_path is not None:
            self.tlm.save_weights(self.ckpt_path)
        if pred:
            self.lrp.save_weights(self.lrp_path)

    def load(self, pred=False):
        if self.ckpt_path is not None:
            self.tlm.load_weights(self.ckpt_path)
        if pred:
            self.lrp.load_weights(self.lrp_path)

    def pred_step(self, inputs):
        inputs, _ = map(mx.array, inputs)
        targets = self.tlm(inputs, pred=True)
        loss, grads = self.lrp_loss(inputs, targets)
        self.lrp_opt.update(self.lrp, grads)
        del grads
        mx.eval(loss, self.lrp.parameters())
        return loss

    def train_step(self, idx, inputs):
        inputs, targets = map(mx.array, inputs)
        loss, grads = self.tlm_loss(inputs, targets)
        self.tlm_opt.learning_rate = min(1, idx / self.lr_warmup) * self.learning_rate
        self.tlm_opt.update(self.tlm, grads)
        del grads
        mx.eval(loss, self.tlm.parameters())
        return loss

    def train(self):
        losses = []
        for epoch in range(self.max_epochs):
            pbar = tqdm(self.train_dataset, total=len(self.train_dataset))
            for idx, inputs in enumerate(pbar):
                loss = self.train_step(idx, inputs)
                loss_item = loss.item()
                losses.append(loss_item)
                pbar.set_description(f"Iteration: {idx + 1}, Loss: {loss_item:.3f}")
                if idx == len(self.train_dataset):
                    break
            train_loss = np.mean(losses)
            print(f"Epoch {epoch+1}: TMLM Mean loss {train_loss:.3f}.")
            self.save_checkpoints()
        if self.predictor:
            for mini in range(self.mini_epochs):
                pbar = tqdm(self.train_dataset, total=len(self.train_dataset))
                for idx, inputs in enumerate(pbar):
                    loss = self.pred_step(inputs)
                    loss_item = loss.item()
                    losses.append(loss_item)
                    pbar.set_description(f"Iteration: {idx + 1}, Loss: {loss_item:.3f}")
                    if idx == len(self.train_dataset):
                        break
                train_loss = np.mean(losses)
                print(f"Epoch {mini+1}: LRP Mean loss {train_loss:.3f}.")
                self.save_checkpoints(pred=True)

