Reinforcement Learning With a Custom Discrete Tennis Environment:
----------------------------------------------------------------

The instructions to run the programs in this project are for a Ubuntu 22.04
system running python 3.10.

[A] Python Environment Setup:
A python virtual environment was created using the virtualenv tool which
can be installed with the commands in a Linux terminal:

	$ sudo apt-get update
    $ sudo apt-get install python3-dev python3-pip
    $ sudo pip3 install -U virtualenv

A virtual environment must be created and activated such as follows:
	$ mkdir ~/pythonEnv; cd ~/pythonEnv
	$ virtualenv -p /usr/bin/python3.10 ~/pythonEnv/rl
	$ .  ~/pythonEnv/rl/bin/activate
	$ pip install --upgrade pip

With this virtual environment activated, ensure that atleast the following 
packages are installed:
    $ sudo apt-get install build-essential cmake git unzip pkg-config
	$ sudo apt-get install libopenblas-dev liblapack-dev
	$ pip install tensorflow
	$ pip install matplotlib scipy pyyaml
	$ pip install h5py
	$ pip install pandas
	$ pip install -U scikit-learn
	$ pip install opencv-python
	$ sudo apt-get install hdf5-tools
	$ pip install gymnasium

[B] Runtime Scripts
The source comprises of scripts that may be used to launch the corresponding
python executable module in a Linux Terminal with an active virtualenv 
prepared as per the instructions in above section. Ensure that the scripts
have execute permission for current user. (if not, do chmod +x <script-path>)

Use the -h command line option to obtain additional command-line options for 
each module, as applicable.

[C] Agent Game Driver:
Script Path: script/runAgentGame.sh
Program Path: test/rl/test_tennis/agentGame.py

Run this driver program to perform training and visual rendering of a trained 
agent's play using different RL algorithms.

Different command line options and in-program ASCII menu based options
provides good flexibility to perform extensive training and testing for
a wide array of usecases. 

Logging can be activated in all the following cases by using the -l option with
a logging configuration JSON file such as the logConfig.json and
logConfigDev2.json files to show a detailed view of the workings of the program.

Some of the exemplary test run options include the following:
1) Train SARSA Agent with greedy policy and epsilon 5%, learning rate 20% and
discount factor of 95%:
--agentType SARSA -s 20 -e 1000 -E 20 --greed 0.05 --learnRate 0.2 --discount 0.95

In-program menu option: "100: Train Agent" to train the agent.
Output artifacts under data/rl/output directory and intermediate models under
data/rl/output/models directory

2) Run a pre-trained agent with a previously saved model under /tmp/tnmodel.npy:
--agentType SARSA -s 20 -e 1000 -E 20 --greed 0.05 --learnRate 0.2 --discount 0.95

In-program menu option: 
  a) "103: Show View" 
  b) "101: Load Model From File", select model file e.g. /tmp/tnmodel.npy
  c) "105: Play Agent Episode"

Now a window with agent acting as player should appear playing against the
system using model from the supplied /tmp/tnmodel.npy.

3) Train DQN Agent with greedy policy and epsilon 5%, learning rate 20% and
discount factor of 95%, mini-batch size as 16
--agentType DQN -s 20 -e 200 -E 5 --greed 0.05 --learnRate 0.2 --discount 0.95 --batchSize 16

In-program menu option: 100 to train the agent.

4) A rendering of the Q-Tables as training progresses can be obtained for a
set of models saved from a previous run.

In-program menu option:
  "111: Render QTable for episode models", select directory of pre-saved model
  files and an optional directory where the result svg files must be stored.

[D] Interactive Game
Script Path: script/runInteractiveGame.sh
Program Path: test/rl/test_tennis/interactiveGame.py

This test program can be used to demonstrate the interactive play tennis
environment where a human player can play against the system.

Command Line Argument: --seed=20

The in-program menu allows the human player to select a move action (during 
system shot) or a fire action for player's shot. Additionally, it is possible to
capture all actions taken during a game in the data/rl/output/action.tn file
and use this file to replay the actions (Option 101: Replay Actions from File) 
for an automatic batch play.

[E] Environment Animations
Script Path: script/runTestAnimation.sh
Program Path: test/rl/test_tennis/testAnimation.py

This test program tests some of the animations that are necessary to render
the optional visual rendering aspect of the custom Tennis environment created
for this project.


