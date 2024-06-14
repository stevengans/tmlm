from tmlm.trainer import Trainer
from tmlm.tokenizer import CharDataset
from datasets import load_dataset


block_size = 256
batch_size = 64

dataset = load_dataset("ccdv/arxiv-summarization", trust_remote_code=True)
train_set = '\n\n'.join(dataset["train"]['abstract'])
train_dataset_gen = CharDataset(train_set, block_size, batch_size)
trainer = Trainer(train_dataset_gen, block_size, batch_size, predictor=True)
trainer.train()

