from rl.tennis.discreteTennis import DiscreteTennis, TennisObservedState
import numpy as np
from rl.tennis.agentBase import BaseTennisAgent, Policy

class SarsaTennisAgent(BaseTennisAgent):
    NAME = "SARSA"
    
    """
    SARSA On-Policy Temporal Difference Learning  Agent implementation of the tennis agent.
    """
    def __init__(self, agent_config, cell_x, cell_y, policy, init_observed_state):
        """
        SARSA tennis agent constructor.
        
        :param cell_x Tennis-court granularity along X-axis (0-1). Gives 1/cell_x + 1 positions on
            court along the X-axis.
        :param cell_y Tennis-court granularity along Y-axis (0-1). Gives 1/cell_y + 1 positions on
            court along the Y-axis.
        :param policy Implementation of action selection Policy.
        :param init_observed_state Environment's observed state at initialization of this agent.
        """
        super().__init__(len(DiscreteTennis.ACTION_SPEC)/2, policy)
        
        self.config = agent_config 
        numpos_x = int(1.0/cell_x) + 1
        numpos_y = int(1.0/cell_y) + 1
        numdrift_dir = DiscreteTennis.DIR_MAX + 1   # Include DIR_NONE
        # Court granularity along X-axis
        self._cell_x = cell_x
        
        # Court granularity along Y-axis
        self._cell_y = cell_y
        
        self.num_action = int(len(DiscreteTennis.ACTION_SPEC)/2)
        
        # 9 x 9 x 9 x 9 = 6561
        self.shape = (numpos_x, numpos_y, numdrift_dir, self.num_action)
        # Q-Table
        # Axis-0 (leftmost): Court-pos-row
        # Axis-1 (next to leftmost): Court-pos-col
        # Axis-2 Drift direction
        # Axis-3 Action move direction: This stores the action q-values.
        self._qtable = np.zeros(self.shape, dtype=np.float32)
        
        # Represents if agent is to fire the next shot. Agent always serves the first set.
        self._agent_fire = True
        # Previous state of the agent.
        self._prev_state = None
        # Last known action returned by next_action (based on _action and _agent_fire)
        self._action_with_fire = None
        
        # Learning rate for bellman eqn.
        #self._alpha = 0.2 
        self._alpha = self.config.get("learn_rate", 0.2)
        
        # Discount factor for bellman eqn
        #self._gamma = 0.95 
        self._gamma = self.config.get("discount_factor", 0.95)
        
        # Agent state represents the cUrrent state of this agent
        self._state = None
        self._to_next_state(init_observed_state)
        # Action taken by this agent for self._state current state.
        # Set the initial action - normally, this would require a policy based action in the
        # initial state using _to_policy_action. However, tennis rules requires first action to 
        # be none as player has to serve without moving away from the baseline-center.
        self._action = DiscreteTennis.DIR_NONE  

    def get_exploit_policy(self):
        """
        Get a policy that can be used to exploit learned best action given current state using reinforce.
        The arguments to select_action for this policy depends on this agent's implementation requirements
        for this exploit policy.
        
        :return Implementation of Policy
        """
        class EExploitMaxActionValuePolicy(Policy):
            """
            Policy to exploit action with maximum action-value.
            """
            def select_action(self, *args, **kwargs):
                """
                Select an action, given the specified values corresponding to each action.
                
                :return index of the selected action.
                """
                action_value_arr = args[0] if len(args) > 0 else kwargs.get("action_value_arr")
                return np.argmax(action_value_arr)
        return(EExploitMaxActionValuePolicy())
    
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
        pst = self._prev_state
        nst = self._state

        paction = self._action
        
        naction_qarr = self._qtable[nst[0]][nst[1]][nst[2]] # Next Q(S')
        # Arg must agree what is supported by get_exploit_policy impl.
        naction = self._policy.select_action(naction_qarr) # A'
        self._action = naction
        
        if pst is None: # No past state yet, nothing to reinforce.
            return
        paction_qarr = self._qtable[pst[0]][pst[1]][pst[2]] # Prev: Q(S)
        
        paction_value = paction_qarr[paction]       # Q(S, A)
        naction_value = naction_qarr[naction]       # Q(S', A')
        
        # Q(S, A) = Q(S, A) + alpha [R + gamma Q(S', A') - Q(S, A)
        paction_qarr[paction] = paction_value + \
            self._alpha * (reward + (self._gamma * naction_value) - paction_value)
            
    def prepare_action(self, observed_state):
        """
        Prepare to take next action.
        
        :param observed_state Current environment state as result of taking previous action
        """
        # Prefix P stands for previous, N for next.
        self._to_next_state(observed_state)
        nst = self._state
        naction_qarr = self._qtable[nst[0]][nst[1]][nst[2]] # Next Q(S')
        naction = np.argmax(naction_qarr)
        self._action = naction
        
    def _to_policy_action(self, state):
        """
        Get the policy based action for the specified state.
        
        :param state Agent state for which policy based action is sought.
        :return Policy based action that corresponds to the specified agent state. 
        """
        action_qarr = self._qtable[state[0]][state[1]][state[2]]
        return self._policy.select_action(action_qarr)
    
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
        # qarr axis 0 (left-most) is court-pos-row, axis 1 is court-pos-col, 
        # axis 2 is drift-dir axis 3 is action-move-dir
        self._state = (player_yunit, player_xunit, drift_dir)
    
    def save_model(self, model_file, add_ext = True):
        """
        Save the learned model to a file.
        
        :param model_file     Model file where the agent's current model is to be saved.
        :param add_ext        Add model file extension if not already specified.
        """
        if add_ext and not model_file.endswith(".npy"):
            model_file = model_file + ".npy"
        np.save(model_file, self._qtable)
    
    def load_model(self, model_file):
        """
        Load a previously learned model from specified file.
        
        :param model_file    Model file from which the agent's current model is to be replaced with that from file.
        """
        self._qtable = np.load(model_file)
            
    def tostr_model(self):
        """
        Get a human readable representation of the learned model as a string.
        
        :return A readable model
        """
        return self.tostr_qtable(self._qtable)
    
    @staticmethod
    def tostr_qtable(qtable):
        shape = qtable.shape
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
                    for action in range(shape[3]):
                        if qtable[y][x][drift_dir][action] != 0.0:
                            result = result + "{}{:>5.2f}".format(separator, qtable[y][x][drift_dir][action])
                        else:
                            result = result + "{}{:>5s}".format(separator, "")
                        separator = ", "
        return result
    
    def get_qtable(self):
        return(self._qtable)
    
    def close(self):
        self._qtable = None
        