class TennisAgent:
    """
    Tennis player reinforcement learning agent interface.
    """
    def configure(self, config_file):
        """
        Configure the agent
        :param config_file Configuration file for the agent, with content depending on type of agent and its implementation.
        :return Next action
        """
        return None
    
    """
    Tennis player reinforcement learning agent interface.
    """
    def next_action(self):
        """
        Take next action based on current state and agent's policy
        
        :return Next action
        """
        return None
    
    def prepare_action(self, observed_state):
        """
        Prepare to take next action. This is used to prepare agent to compute the next action to be
        taken based on the current observed state of the environment before next_action can be
        called while agent is not undergoing training.
        
        :param observed_state Current environment state as result of taking previous action
        """
        pass
    
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
        pass
    
    def tostr_model(self):
        """
        Get a human readable representation of the learned model as a string.
        
        :return A readable model
        """
        return None
    
    def get_qtable(self):
        """
        Get q-table or its equivalent, if any, for this agent's model
        
        :return Q-Table or its equivalent, if any, else None
        """
        return None
    
    def close(self):
        return None
    
class Policy:
    """
    Action selection policy interface
    """
    def select_action(self, *args, **kwargs):
        """
        Select an action, given the specified values corresponding to each action.
        
        :args Positional arguments expected by the policy implementation.
        :kwargs Keyword arguments expected by the policy implementatsion.
        :return index of the selected action.
        """
        return None

class EGreedyPolicy(Policy):
    """
    Epsilon-greedy policy for action selection.
    """
    def __init__(self, epsilon, random, exploit_delegate_policy = None):
        """
        Epsilon-greedy policy constructor.
        
        :param epsilon    Percentage of times, agent must explore (0.0-1.0).
        :param random     Random implementation to be used.
        :param exploit_delegate_policy    Exploit delegate policy
        """
        self._epsilon = epsilon
        self._random = random
        self._num_action = 1
        self._exploit_delegate_policy = exploit_delegate_policy
    
    def set_num_action(self, num_action):
        """
        Set number of possible actions.
        
        :param num_action    Number of actions
        """
        self._num_action = num_action
        
    def set_exploit_delegate(self, policy):
        """
        Set the delegate policy to be used while not exploring but exploiting the learned actions.
        
        :param policy Exploit delegate policy
        """
        self._exploit_delegate_policy = policy
        
    def select_action(self, *args, **kwargs):
        """
        Select an action, given the specified values corresponding to each action.
        
        :return index of the selected action.
        """
        current_selector = self._random.random()
        if current_selector > self._epsilon and self._exploit_delegate_policy is not None:
            # Exploit
            return self._exploit_delegate_policy.select_action(args, kwargs)
        else:
            # Explore
            return self._random.randrange(0, self._num_action)
        
class BaseTennisAgent(TennisAgent):
    """
    Base Tennis Agent that handles managing concerns common across many typical Agent implementations. 
    """
    def __init__(self, num_action, policy):
        """
        Base Tennis Agent constructor.
        
        :param num_action     Number of possible actions for this agent implementation.
        :param policy        Action selection policy to be used for this agent.
        """
        self.num_action = int(num_action)
        self.set_policy(policy)
        
    def set_policy(self, policy):
        """
        Set agent's action selection policy
        
        :param policy    New policy implementation to be used.
        """
        self._policy = policy
        
        # If policy 
        set_num_action = getattr(policy, "set_num_action", None)
        if callable(set_num_action):
            self._policy.set_num_action(self.num_action)
            
        set_exploit_delegate = getattr(policy, "set_exploit_delegate", None)
        if callable(set_exploit_delegate):
            self._policy.set_exploit_delegate(self.get_exploit_policy())
    
    def get_exploit_policy(self):
        """
        Get a policy that can be used to exploit learned best action given current state using reinforce.
        The arguments to select_action for this policy depends on this agent's implementation requirements
        for this exploit policy. Derived classes must implement this method to provide a policy suitable
        for that agent's implementation.
        
        :return Implementation of Exploit Policy
        """
        return None