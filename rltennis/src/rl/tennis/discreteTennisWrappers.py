from .behaviorLearnable import LearnableTennisBehavior, TennisBehaviorShotRewardOnly
from .discreteTennis import DiscreteTennis


class DiscreteTennisEasy(DiscreteTennis):
    def __init__(self, seed=None):
        behavior = LearnableTennisBehavior(seed=seed)
        behavior.REWARD_MAP = {
            DiscreteTennis.ACTIVITY_SYSTEM_INVALID_SHOT: 0,
            DiscreteTennis.ACTIVITY_SYSTEM_MISS: 0,
            DiscreteTennis.ACTIVITY_SYSTEM_SHOT: 0,
            DiscreteTennis.ACTIVITY_PLAYER_INVALID_SHOT: 0,
            DiscreteTennis.ACTIVITY_PLAYER_MISS: 0,
            DiscreteTennis.ACTIVITY_PLAYER_SHOT: 1,
        }
        super().__init__(behavior)

        self.MAX_GAME_LENGTH = 8
        self.SYSTEM_ALWAYS_SERVE = True


class DiscreteTennisHard(DiscreteTennis):
    def __init__(self, seed=None, max_game_length=None):
        behavior = TennisBehaviorShotRewardOnly(seed=seed)
        behavior.REWARD_MAP = {
            DiscreteTennis.ACTIVITY_SYSTEM_INVALID_SHOT: 0,
            DiscreteTennis.ACTIVITY_SYSTEM_MISS: 0,
            DiscreteTennis.ACTIVITY_SYSTEM_SHOT: 0,
            DiscreteTennis.ACTIVITY_PLAYER_INVALID_SHOT: 0,
            DiscreteTennis.ACTIVITY_PLAYER_MISS: 0,
            DiscreteTennis.ACTIVITY_PLAYER_SHOT: 1,
        }
        super().__init__(behavior)
        
        self.MAX_GAME_LENGTH = 8
        self.SYSTEM_ALWAYS_SERVE = True
