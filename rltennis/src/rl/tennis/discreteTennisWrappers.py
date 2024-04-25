from .behaviorLearnable import LearnableTennisBehavior, TennisBehaviorShotRewardOnly
from .discreteTennis import DiscreteTennis


class DiscreteTennisEasy(DiscreteTennis):
    def __init__(self, seed=None, max_game_length=None):
        super().__init__(LearnableTennisBehavior(seed=seed))
        self.MAX_GAME_LENGTH = max_game_length


class DiscreteTennisHard(DiscreteTennis):
    def __init__(self, seed=None, max_game_length=None):
        super().__init__(TennisBehaviorShotRewardOnly(seed=seed))
        self.MAX_GAME_LENGTH = max_game_length
        self.SYSTEM_ALWAYS_SERVE = True
