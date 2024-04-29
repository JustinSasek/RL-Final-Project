from rl.tennis.agentDQN import DQNTennisAgent
from rl.tennis.agentSARSA import SarsaTennisAgent

class AgentFactory:
    """
    Factory for creating different types of agents.
    """
    default_factory = None
    
    @classmethod
    def get_default_factory(cls):
        if cls.default_factory is not None:
            return cls.default_factory
        
        fac = AgentFactory()
        
        fac.register_agent(SarsaTennisAgent.NAME, 
             lambda agent_config, behavior, policy, init_observed_state: \
             SarsaTennisAgent(agent_config, behavior.cell_x, behavior.cell_y, policy, init_observed_state))

        fac.register_agent(DQNTennisAgent.NAME, 
             lambda agent_config, behavior, policy, init_observed_state: \
             DQNTennisAgent(agent_config, behavior.cell_x, behavior.cell_y, policy, init_observed_state, behavior.random))
                
        cls.default_factory = fac
        return cls.default_factory
    
    def __init__(self):
        """
        Constructor for agent building and management.
        """
        self._builder = {}
        self._reglist = []
    
    def register_agent(self, name, builder):
        self._builder[name] = builder
        self._reglist.append(name)
        
    def create_agent(self, agent_type, agent_config, behavior, policy, init_observed_state):
        builder = self._builder.get(agent_type)
        if builder is None:
            raise Exception("No registered agent type with name={}".format(agent_type))
        
        return(builder(agent_config, behavior, policy, init_observed_state))