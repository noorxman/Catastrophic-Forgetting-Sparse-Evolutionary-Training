# pytorch-ewc
Adaptation of the unofficial PyTorch implementation of DeepMind's paper to combine SET and EWC [Overcoming Catastrophic Forgetting, PNAS 2017](https://arxiv.org/abs/1612.00796) for the Information Theory course.

![graphic-image](./arts/graphic-image.jpg)

## Results

Continual Learning **without EWC** (*left*) and **with EWC** (*right*).

<img width="300" src="arts/precision-plain.png" /> <img width="300" src="arts/precision-consolidated.png" />


## Installation
```
$ git clone https://github.com/kuc2477/pytorch-ewc && cd pytorch-ewc
$ pip install -r requirements.txt
```


## CLI
Implementation CLI is provided by `main.py`

## Reference
- [Overcoming Catastrophic Forgetting, PNAS 2017](https://arxiv.org/abs/1612.00796)

## Author
Ha Junsoo / [@kuc2477](https://github.com/kuc2477) / MIT License
