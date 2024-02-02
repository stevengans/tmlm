import math
import numpy as np
from tqdm import tqdm


class CharDataset:

    def __init__(self, data, block_size, batch_size):
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print("Unique CHARS: ", chars)
        print("Text Size: ", data_size)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.data = np.array([self.stoi[char] for char in tqdm(data, desc='Converting characters')], dtype=np.uint32)
        self.batch_size = batch_size
        self.block_size = block_size
        self.vocab_size = vocab_size

    def __len__(self):
        return math.ceil(len(self.data) / (self.block_size + 1))

    def __iter__(self):
        inputs, targets = self.to_samples()
        s = 0
        while True:
            if s == 0:
                perm = np.random.permutation(inputs.shape[0])
            ids = perm[s: s + self.batch_size]
            yield inputs[ids], targets[ids]
            s += self.batch_size
            if s >= inputs.shape[0]:
                s = 0

    def to_samples(self):
        tokens = self.data.size
        window_size = self.block_size + 1
        samples = tokens - window_size + 1
        X = np.lib.stride_tricks.as_strided(
            self.data,
            shape=(samples, window_size),
            strides=(self.data.itemsize, self.data.itemsize),
        )
        return X[:, :-1], X[:, 1:]

    __call__ = __iter__
