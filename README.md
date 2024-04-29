# RL Final Project

By Justin Sasek and Harshal Bharatia

## Description

This project explores how reinforcement learning can be applied to a partially observable environment. We create a 2D tennis environment from scratch, code several baseline agents, and implement a REINFORCE-style agent that uses memory to address the partial observability of the environment.

## How to Run

First, setup a conda environment with the necessary dependencies.

```bash
# make a fresh conda environment with python 3.10.13
conda create -n rl_final_project python==3.10.13
conda activate rl_final_project
# install the package and its dependencies
pip install -e .
```

Then, to train the REINFORCE-style algorithms (including the ones with memory), run the training_loop.py script.

```bash

python3 rltennis/src/rl/training_loop.py

```

The other baseline algorithms are located in a different script because they do not require memory. These can be run with the following command.

```bash