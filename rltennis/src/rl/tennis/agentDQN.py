from rl.tennis.discreteTennis import DiscreteTennis, TennisObservedState
import numpy as np
import logging
from keras import layers
from keras import models
from keras import optimizers
from keras import losses
from collections import deque
from rl.tennis.agentBase import BaseTennisAgent, Policy

LOGGER_BASE = "rl.tennis.DQNTennisAgent"

class DQNTennisAgent(BaseTennisAgent):
    _logger = logging.getLogger(LOGGER_BASE)
    NAME = "DQN"
    _LOG_TAG = {"lgmod": "RLAGT_DQN"}
    
    """
    DQN On-Policy Temporal Difference Learning  Agent implementation of the tennis agent.
    """
    def __init__(self, agent_config, cell_x, cell_y, policy, init_observed_state, random):
        """
        DQN tennis agent constructor.
        
        :param cell_x Tennis-court granularity along X-axis (0-1). Gives 1/cell_x + 1 positions on
            court along the X-axis.
        :param cell_y Tennis-court granularity along Y-axis (0-1). Gives 1/cell_y + 1 positions on
            court along the Y-axis.
        :param policy Implementation of action selection Policy.
        :param init_observed_state Environment's observed state at initialization of this agent.
        :param random    Random instance to be used for randomness.
        """
        super().__init__(len(DiscreteTennis.ACTION_SPEC)/2, policy)

        self.config = agent_config
        self._random = random

        # Court granularity along X-axis
        self._cell_x = cell_x
        
        # Court granularity along Y-axis
        self._cell_y = cell_y
        
        # Represents if agent is to fire the next shot. Agent always serves the first set.
        self._agent_fire = True
        # Previous state of the agent.
        self._prev_state = None
        # Last known action returned by next_action (based on _action and _agent_fire)
        self._action_with_fire = None
        
        # Learning rate for bellman eqn.
        self._alpha = self.config.get("learn_rate", 0.1)
        
        # Discount factor for bellman eqn
        self._gamma = self.config.get("discount_factor", 0.95)
        
        self.tau = self.config.get("dqn_tau", 0.125)
        self._batch_size = self.config.get("batch_size", 32)
        self._sync_target_steps = self.config.get("_sync_target_steps", 32)
        self._max_buffer = self.config.get("replay_buffer", 4 * 1024)
        
        # Agent state represents the cUrrent state of this agent
        self._state = None
        self._to_next_state(init_observed_state)
        # Action taken by this agent for self._state current state.
        # Set the initial action - normally, this would require a policy based action in the
        # initial state using _to_policy_action. However, tennis rules requires first action to 
        # be none as player has to serve without moving away from the baseline-center.
        self._action = DiscreteTennis.DIR_NONE  
        
        num_dir = DiscreteTennis.DIR_MAX + 1
        self._number_output = num_dir
        self._number_input = 1 + 1 + num_dir # X-Pos, Y-Pos, encoded drift dirs.
        self._train_input = np.zeros((self._batch_size, self._number_input))
        self._train_output = np.zeros((self._batch_size, self._number_output))
        
        self._train_query_current = np.zeros((self._batch_size, self._number_input))
        self._train_query_future = np.zeros((self._batch_size, self._number_input))
        
        # Queue upto max_buffer items, discarding older values once limit is reached.
        self._memory  = deque(maxlen=self._max_buffer)
        
        # Neural network that does predictions for each action.
        self._predictModel = self._build_model()
        # Target network that produces the target Q-value at each attempt.  
        self._targetModel = self._build_model()
        self._update_target = 0

    def get_exploit_policy(self):
        """
        Get a policy that can be used to exploit learned best action given current state using reinforce.
        The arguments to select_action for this policy depends on this agent's implementation requirements
        for this exploit policy.
        
        :return Implementation of Policy
        """
        class ExploitDQNPolicy(Policy):
            def __init__(self, agent):
                self._agent = agent
                
            def select_action(self, *args, **kwargs):
                """
                Select an action, given the specified values corresponding to each action.
                
                :return index of the selected action.
                """
                curr_state = args[0] if len(args) > 0 else kwargs.get("state")
                target = self._agent._predictModel.predict(curr_state, verbose=None)
                action = np.argmax(target[0])
                return action
            
        return(ExploitDQNPolicy(self))
    
    def _build_model(self):
        # With States: 9 (x-pos) X 9 (y-pos) X 9 (drift-dir) = 729, Actions = 9 : Total comb to learn = 6561
        # Number of hyper-params needs to learn approx =  6561 x 10 approx 64K.
        
        #numpos_x = int(1.0/self._cell_x) + 1
        #numpos_y = int(1.0/self._cell_y) + 1
        #numdrift_dir = DiscreteTennis.DIR_MAX + 1   # Include DIR_NONE
        # TODO Adjust hidden layer size based on numpos_x and numpos_y
        
        # X, Y positions are relative whereas drift-dir is categorical. SO we have 2+9 = 11 inputs.
        # Hidden layers 11 inputs : 32, 32, 64 : 9 output = 64K
        # model = models.Sequential()
        #
        # model.add(layers.Dense(32, input_dim=self._number_input, activation='relu'))
        # model.add(layers.Dense(64, activation='relu'))
        # model.add(layers.Dense(128, activation='relu'))
        # model.add(layers.Dense(self._number_output, activation='linear'))
        #
        # model.summary()
        #
        # model.compile(loss='mean_squared_error',
        #     optimizer=optimizers.Adam(learning_rate=1e-3),
        #     metrics=['acc'])
        
        model = models.Sequential()
        
        model.add(layers.Dense(32, input_dim=self._number_input))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('tanh'))
    
        model.add(layers.Dense(64))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('tanh'))
        
        model.add(layers.Dense(128))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('tanh'))
        
        model.add(layers.Dense(self._number_output, activation='linear'))
        
        model.summary()
        
        # model.compile(loss='mean_squared_error',
        #     optimizer=optimizers.Adam(learning_rate=1e-3),
        #     metrics=['acc'])
        
        model.compile(loss=losses.Huber(),
            optimizer=optimizers.Adam(learning_rate=1e-3),
            metrics=['acc'])
        return(model)

    def _remember(self, state, action, reward, new_state, done):
        """
        Save the result of current action so that it can be replayed in the future.
        
        :param state Current state
        :param action Current action
        :param reward Reward that was returned for taking the current action in current state.
        :param new_state Next state to which the environment transited due to current action.
        :param done True if the episode reached termination.
        """
        self._memory.append([state, action, reward, new_state, done])
    
    def _replay(self, batchSize=32):
        """
        Replay a batch of remembered actions to cause the prediction model to learn from these actions.
        """
        if len(self._memory) < batchSize: 
            return None # Not enough memory available yet - do nothing.

        # Query all current and next states for their action values for each sample in the minibatch
        training_samples = self._random.sample(self._memory, batchSize)
        query_index = 0
        for sample in training_samples:
            state, action, reward, new_state, done = sample
            self._train_query_current[query_index] = state[0]
            self._train_query_future[query_index] = new_state[0]
            query_index = query_index + 1
        target_batch_current = self._predictModel.predict(self._train_query_current, verbose=None)
        target_batch_future = self._targetModel.predict(self._train_query_future, verbose=None)
        
        # Using Bellman equation get the expected action values for each current state in the minibatch.
        query_index = 0
        batch_index = 0
        for sample in training_samples:
            state, action, reward, new_state, done = sample
            st_target = target_batch_current[query_index]
            nst_target= target_batch_future[query_index]
            query_index = query_index + 1
            
            # Compute the target Q-Value for current training sample.
            # state is a batch input array of size 1 with only one state entry.
            #   The state entry comprises x_pos, y_pos, encoded drift_dir
            # target is a batch output array of size 1 with only one result entry.
            #   The result entry comprises action-values one for each of the nine actions in input state. 
            if done:
                st_target[action] = reward
            else:
                q_future = max(nst_target)
                st_target[action] = reward + q_future * self._gamma
            
            self._train_input[batch_index] = state[0]
            self._train_output[batch_index] = st_target
            
            batch_index = batch_index + 1
        
        # Train for loss between current action values and the expected action values computed above:
        
        # loss function is difference between predicted action-values and reinforced action-value computed above.
        # optimizer does gradient descent.
        # TODO: See if fitting a bigger batch works better and explore tape-gradient otherwise.
        result = self._predictModel.fit(self._train_input, self._train_output, epochs=1, verbose=None)
        return result
    
    def _replay_single(self, batchSize=32):
        """
        Replay a batch of remembered actions to cause the prediction model to learn from these actions.
        """
        if len(self._memory) < batchSize: 
            return None # Not enough memory available yet - do nothing.

        training_samples = self._random.sample(self._memory, batchSize)
        batch_index = 0
        for sample in training_samples:
            state, action, reward, new_state, done = sample
            # Compute the target Q-Value for current training sample.
            # state is a batch input array of size 1 with only one state entry.
            #   The state entry comprises x_pos, y_pos, encoded drift_dir
            # target is a batch output array of size 1 with only one result entry.
            #   The result entry comprises action-values one for each of the nine actions in input state. 
            target = self._targetModel.predict(state, verbose=None)
            if done:
                target[0][action] = reward
            else:
                q_future = max(self._targetModel.predict(new_state, verbose=None)[0])
                target[0][action] = reward + q_future * self._gamma
            
            self._train_input[batch_index] = state[0]
            self._train_output[batch_index] = target[0]
            batch_index = batch_index + 1
            # loss function is difference between predicted action-values and reinforced action-value computed above.
            # optimizer does gradient descent.
            # TODO: See if fitting a bigger batch works better and explore tape-gradient otherwise.
        result = self._predictModel.fit(self._train_input, self._train_output, epochs=1, verbose=None)
        return result
    
    def _sync_target_model(self):
        """
        Synchronize the target model to bear the same weights as the prediction model, essentially
        equalizing the two models.
        """
        weights = self._predictModel.get_weights()
        target_weights = self._targetModel.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self._targetModel.set_weights(target_weights)
        
    def next_action(self):
        """
        Compute the next action to be taken by this agent based in current state of the agent.
        
        :return Next action.
        """
        # Agent always computes the next action during last reinforce/initialization.
        dir2act = DiscreteTennis.DIR_ACTION_MAP[self._action]
        self._action_with_fire = dir2act[1] if self._agent_fire else dir2act[0]
        return self._action_with_fire
    
    def reinforce(self, observed_state, reward, done, truncated, info, episode, episode_step):
        """
        Reinforce the agent based on specified reward and environment change in response to the last action.
        
        :param observed_state Current environment state as result of taking previous action
        :param reward      Reward obtained for previous action.
        :param done        True if the current episode is completed.
        :param truncated   True if the current episode is truncated.
        :param info        Additional environment specific information.
        :param episode     Current episode number
        :param episode_step Step count within the current episode.
        """
        # Prefix P stands for previous, N for next.
        self._to_next_state(observed_state)
        self._update_target = self._update_target + 1
        if self._prev_state is not None:
            self._remember(self._prev_state, self._action, reward, self._state, done)
            result = self._replay(self._batch_size)
            if episode_step % 10 == 0 and result is not None:
                if self._logger.isEnabledFor(logging.DEBUG):
                    self._logger.debug("Episode {}-{}, Loss={}, Accuracy={}".format(episode, episode_step, result.history["loss"], result.history["acc"]),
                        extra=self._LOG_TAG)
            if self._update_target >= self._sync_target_steps:
                self._sync_target_model()
                self._update_target = 0
        
        self._action = self._policy.select_action(self._state) # A'
        
    def prepare_action(self, observed_state):
        """
        Prepare to take next action.
        
        :param observed_state Current environment state as result of taking previous action
        """
        # Prefix P stands for previous, N for next.
        self._to_next_state(observed_state)
        target = self._agent._predictModel.predict(self._state, verbose=None)
        self._action = np.argmax(target[0])
        
    def _to_drift_dir(self, player_pos, ball_pos):
        """
        Get the direction in which the ball is drifting for the player.
        
        :param player_pos    Tuple representing player's current position.
        :param ball_pos      Tuple representing position of the ball received by the player.
        :return Direction of ball relative to the player.
        """
        if player_pos[0] == ball_pos[0]:
            if player_pos[1] < ball_pos[1]:
                return DiscreteTennis.DIR_FRONT
            elif player_pos[1] > ball_pos[1]:
                return DiscreteTennis.DIR_BACK
            else:
                return DiscreteTennis.DIR_NONE
        
        if player_pos[1] == ball_pos[1]:
            if player_pos[0] < ball_pos[0]:
                return DiscreteTennis.DIR_RIGHT
            else:
                return DiscreteTennis.DIR_LEFT
        
        if player_pos[0] < ball_pos[0]:
            if player_pos[1] < ball_pos[1]:
                return DiscreteTennis.DIR_FRONTRIGHT
            else: # player_pos[1] > ball_pos[1]:
                return DiscreteTennis.DIR_BACKRIGHT
        
        if player_pos[0] > ball_pos[0]:
            if player_pos[1] < ball_pos[1]:
                return DiscreteTennis.DIR_FRONTLEFT
            else: # player_pos[1] > ball_pos[1]:
                return DiscreteTennis.DIR_BACKLEFT
        
        return DiscreteTennis.DIR_NONE
        
    def _to_next_state(self, observed_state):
        """
        Transition the agent to next state based on the observed state of the environment.
        
        :param observed_state    Observed state of the environment
        """
        player_fires = False
        new_game = TennisObservedState.get_current_game_result(observed_state) != 0
        ball_pos = TennisObservedState.get_ball_position(observed_state)
        if new_game:
            player_fires = TennisObservedState.get_next_server(observed_state) == DiscreteTennis.PLAYER
        else:
            player_fires = ball_pos[2] == DiscreteTennis.PLAYER # Ball is given to player.
            
        player_pos = TennisObservedState.get_player_position(observed_state)
        player_xunit =  int(player_pos[0]/self._cell_x)
        player_yunit =  int(player_pos[1]/self._cell_y)
        
        # If ball is to be fired by player, ball_position computes player's most recent drift. 
        # If ball is fired by system, ball_source_position provides player's most recent drift.
        drift_pos = ball_pos if player_fires else TennisObservedState.get_ball_source_position(observed_state)
        drift_dir = self._to_drift_dir(player_pos, drift_pos)
        
        self._agent_fire = player_fires
        self._prev_state = self._state
        self._state = self._encode_state(player_yunit, player_xunit, drift_dir)
    
    def _encode_state(self, player_yunit, player_xunit, drift_dir):
        """
        Encode the agent state to inputs suitable for the neural network.
        The x and y positions are linear and continuous positions along court axis and remain verbatim.
        The drift-direction is categorical and therefore is one-hot encoded.
        
        :param player_yunit    Player's vertical position in its court from 0 to 1
        :param player_xunit    Player's horizontal position in its court from 0 to 1
        :param drift_dir       Ball's drift direction.
        """
        enc_state = np.zeros((1,int(self.num_action+2)))
        enc_state[0][0] = player_yunit
        enc_state[0][1] = player_xunit
        act_st = 2
        for act_index in range(self.num_action):
            enc_state[0][act_st + act_index] = act_index == drift_dir
        
        return enc_state
    
    def save_model(self, model_file, add_ext = True):
        """
        Save the learned model to a file.
        
        :param model_file     Model file where the agent's current model is to be saved.
        :param add_ext        Add model file extension if not already specified.
        """
        if not model_file.endswith(".keras"):
            model_file = model_file + ".keras"
        self._predictModel.save(model_file)
    
    def load_model(self, model_file):
        """
        Load a previously learned model from specified file.
        
        :param model_file    Model file from which the agent's current model is to be replaced with that from file.
        """
        self._predictModel = models.load_model(model_file)
        self._targetModel = models.clone_model(self._predictModel)
            
    def get_qtable(self):
        shape = (int(1/self._cell_y), int(1/self._cell_x), int(DiscreteTennis.DIR_MAX + 1), int(DiscreteTennis.DIR_MAX + 1))
        qtable = np.zeros(shape, dtype=np.float32)
        for y in range(shape[0]):
            for x in range(shape[1]):
                for drift_dir in range(shape[2]):
                    state = self._encode_state(y * self._cell_y, x * self._cell_x, drift_dir)
                    target = self._predictModel.predict(state, verbose=None)
                    for action in range(shape[3]):
                        qtable[y][x][drift_dir][action] = target[0][action]
        return qtable
    
    def tostr_model(self):
        """
        Get a human readable representation of the learned model as a string.
        
        :return A readable model
        """
        shape = (int(1/self._cell_y), int(1/self._cell_x), int(DiscreteTennis.DIR_MAX + 1), int(DiscreteTennis.DIR_MAX + 1))
        result = ""
        separator = ""
        dir_ref = ""
        for drift_dir in range(DiscreteTennis.DIR_MAX + 1):
            dir_ref = dir_ref + "{}{:>5s}".format(separator, DiscreteTennis.DIR_NAME_MAP.get(drift_dir)[1])
            separator = ", "
        for y in range(shape[0]):
            for x in range(shape[1]):
                result = result + "\n@({:>2d}, {:>2d}):  {}".format(x, y, dir_ref)
                for drift_dir in range(shape[2]):
                    result = result + "\n        {}: ".format(DiscreteTennis.DIR_NAME_MAP[drift_dir][1])
                    separator = ""
                    state = self._encode_state(y * self._cell_y, x * self._cell_x, drift_dir)
                    target = self._predictModel.predict(state, verbose=None)
                    for action in range(shape[3]):
                        result = result + "{}{:>5.2f}".format(separator, target[0][action])
                        separator = ", "
        return result
    
    def close(self):
        self._predictModel = None
        self._targetModel = None
        self._memory = None
        self._train_input = None
        self._train_output = None
        self._train_query = None