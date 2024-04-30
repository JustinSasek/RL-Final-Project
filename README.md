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
pip install -e rltennis/src/rl
```

Then, to train the REINFORCE-style algorithms (including the ones with memory), run the training_loop.py script.

```bash

python3 rltennis/src/rl/training_loop.py

```

The other baseline algorithms are located in a different script because they do not require memory. These can be run with the following instructions.

### 1. Python Environment Setup

A python virtual environment was created using the virtualenv tool which can be installed with the commands in a Linux terminal:

``` bash
sudo apt-get update
sudo apt-get install python3-dev python3-pip
sudo pip3 install -U virtualenv
```

A virtual environment must be created and activated such as follows:

``` bash
mkdir ~/pythonEnv; cd ~/pythonEnv
virtualenv -p /usr/bin/python3.10 ~/pythonEnv/rl
.  ~/pythonEnv/rl/bin/activate
pip install --upgrade pip
```

With this virtual environment activated, ensure that atleast the following packages are installed:

``` bash
sudo apt-get install build-essential cmake git unzip pkg-config
sudo apt-get install libopenblas-dev liblapack-dev
pip install tensorflow
pip install matplotlib scipy pyyaml
pip install h5py
pip install pandas
pip install -U scikit-learn\
-Y 80  \
-YM 150 \
#-T lines

pip install opencv-python
sudo apt-get install hdf5-tools
pip install gymnasium
```

### 2. Eclipse IDE Environment

Install Eclipse eclipse-2024-03-R as obtained from [eclipse.org/downloads/packages](eclipse.org/downloads/packages) for Eclipse IDE for Enterprise Java and Web Developers and install the PyDev version 12.0.0.202402010911 addon using Eclipse Marketplace.

Using a Linux terminal and active virtual environment described in earlier section, launch the installed eclipse binary from the terminal. Configure the Python Interpreter in Windows -> Preferences -> PyDev -> Interpreters -> Python Interpreter -> New -> Choose from list and select python3 from the virtual-env.

Launch the eclipse IDE and use File -> Import -> Existing Projects into Workspace and navigate to the project's base directory to find the existing project and import the project. Once the project is imported, it is now possible to run the project executable classes described below by doing the following steps:

1. Open the `executable.py` file in the editor.
2. Right click the file and in the pop-up menu, select Run as -> Python Run
3. Stop the program and configure its arguments by clicking the menu Run -> Run Configurations ... -> Python Run -> `executable .py` and specify the required arguments in the Arguments tab

### 3. Agent Game Driver

Program Path: `test/rl/test_tennis/agentGame.py`
Run this driver program to perform training and visual rendering of a trained agent's play using different RL algorithms.

Different command line options and in-program ASCII menu based options provides good flexibility to perform extensive training and testing for a wide array of usecases.

Logging can be activated in all the following cases by using the `-l` option with a logging configuration JSON file such as the `logConfig.json` and `logConfigDev2.json` files to show a detailed view of the workings of the program.

Some of the exemplary test run options include the following:

1. Train SARSA Agent with greedy policy and epsilon 5%, learning rate 20% and discount factor of 95%:
`--agentType SARSA -s 20 -e 1000 -E 20 --greed 0.05 --learnRate 0.2 --discount 0.95`

    In-program menu option: "100: Train Agent" to train the agent. Output artifacts under `data/rl/output` directory and intermediate models under `data/rl/output/models` directory

2. Run a pre-trained agent with a previously saved model under /tmp/tnmodel.npy:
`--agentType SARSA -s 20 -e 1000 -E 20 --greed 0.05 --learnRate 0.2 --discount 0.95`

    In-program menu option:

   - "103: Show View"
   - "101: Load Model From File", select model file e.g. `/tmp/tnmodel.npy`
   - "105: Play Agent Episode"

    Now a window with agent acting as player should appear playing against the system using model from the supplied /tmp/tnmodel.npy.

3. Train DQN Agent with greedy policy and epsilon 5%, learning rate 20% and discount factor of 95%, mini-batch size as 16
`--agentType DQN -s 20 -e 200 -E 5 --greed 0.05 --learnRate 0.2 --discount 0.95 --batchSize 16`

    In-program menu option: 100 to train the agent.

4. A rendering of the Q-Tables as training progresses can be obtained for a set of models saved from a previous run.

    In-program menu option:
    "111: Render QTable for episode models", select directory of pre-saved model files and an optional directory where the result svg files must be stored.

### 4. Interactive Game

Program Path: `test/rl/test_tennis/interactiveGame.py`
This test program can be used to demonstrate the interactive play tennis environment where a human player can play against the system.

Command Line Argument: `--seed=20`

The in-program menu allows the human player to select a move action (during system shot) or a fire action for player's shot. Additionally, it is possible to capture all actions taken during a game in the `data/rl/output/action.tn` file and use this file to replay the actions (Option 101: Replay Actions from File) for an automatic batch play.

### 5. Environment Animations

Program Path: `test/rl/test_tennis/testAnimation.py`

This test program tests some of the animations that are necessary to render the optional visual rendering aspect of the custom Tennis environment created for this project.

