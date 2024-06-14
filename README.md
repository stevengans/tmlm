## Tiny Metal Language Model (TMLM)

<img src="splash.jpeg" alt="splash">

An MLX (https://github.com/ml-explore/mlx) Implementation of the TinyStories Model (https://arxiv.org/abs/2305.07759) using the 33M Parameter configuration.

For on-device inference, an implementation of Low Rank Predictor (https://arxiv.org/pdf/2312.11514).

To get started:
`pip install requirements.text`

To start training 33M Model: `python main.py`

To start training 33M Model and Low Rank Predictor: `python low_rank_predictor.py`

