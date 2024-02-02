from tmlm.trainer import Trainer
from tmlm.tokenizer import CharDataset
from datasets import load_dataset


block_size = 256
batch_size = 64
dataset = load_dataset("roneneldan/TinyStories", split="train")
train_set = '\n\n'.join(dataset["text"])
train_dataset_gen = CharDataset(train_set, block_size, batch_size)
trainer = Trainer(train_dataset_gen, len(train_dataset_gen), train_dataset_gen.vocab_size, block_size, batch_size)
trainer.train()
