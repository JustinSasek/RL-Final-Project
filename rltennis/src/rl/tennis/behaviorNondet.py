# Refer: https://blog.paperspace.com/creating-custom-environments-openai-gym/
import cv2 
import random
import logging
from rl.tennis.discreteTennis import DiscreteTennis, TennisBehavior, RenderEvent, SystemShot, SystemMove

from math import sqrt

font = cv2.FONT_HERSHEY_COMPLEX_SMALL 

LOGGER_BASE = "rl.tennis.discreteTennis"
class SimpleDiscreteTennisBehavior(TennisBehavior):
    _logger = logging.getLogger(LOGGER_BASE)
    _LOG_TAG = {"lgmod": "TNENV"}
    
    REWARD_MAP = {
        DiscreteTennis.ACTIVITY_SYSTEM_INVALID_SHOT: 2,
        DiscreteTennis.ACTIVITY_SYSTEM_MISS: 2,
        DiscreteTennis.ACTIVITY_SYSTEM_SHOT: 0,
        DiscreteTennis.ACTIVITY_PLAYER_INVALID_SHOT: -2,
        DiscreteTennis.ACTIVITY_PLAYER_MISS: -2,
        DiscreteTennis.ACTIVITY_PLAYER_SHOT: 0
    }
    
    """
    Tennis behavior implementation that enables the user to specify a behavior to be exhibited by the
    environment. The behavior assumes a discrete game where the ball and player locations are at only
    designated places in the court.
    """
    def __init__(self, seed = 20):
        # Player and system positions snap to a grid with specified cell size.
        self.cell_x = 0.125
        self.cell_y = 0.125
        
        # Player is capable to move maximum this much distance.
        self.player_x = 0.125
        self.player_y = 0.125
        
        # System is capable to move maximum this much distance.
        self.system_x = 0.125
        self.system_y = 0.125
        
        # Shots with ball end positions this much away from borders are vulnerable shots. (left, back, right, front) 
        self.border = [0.1, 0.05, 0.1, 0.2]
        # Percentage of vulnerable shots that will be valid. Set it to 1.0 to disable this feature.  (left, back, right, front)
        #self.shot_valid_probability = [0.9, 0.9, 0.9, 0.2]
        self.shot_valid_probability = [1.0, 1.0, 1.0, 1.0]
        
        # Even if shot is within reach, distance becomes inflated this many percentage of times.
        # Set them to 0.0 to disable this feature.
        self.player_reach_factor = 0.1  # 10%
        self.system_reach_factor = 0.1
        
        # A difficulty scale from 0=every easy to 1=very hard. 
        self.difficulty = 0.95
        
        # This decides percentage of difficult shots that trigger the very difficult 2x horizontal 
        # displacement strategy wherein the shot is placed 2x distance away from the actor until
        # current game completes. This is akin moving player between down-the-line and cross-court
        # shots in tennis. This is currently stands at 10% of overall shots.
        self.difficulty_2x_factor = 0.40
        
        # Stretch the shot distance from peer randomly upto this much percentage of peer's reach.
        #
        # The difficulty percent is then used to compute shot distance such that difficulty percentage 
        # shots reside in this stretched extra distance.
        # 25% difficulty will cause 25% of shots to have the target shot within this stretched distance from peer.
        self.shot_target_stretch_factor = 0.1
        
        # If active, shot target hovers 2x away for firing at player. 
        self.player_2x_displace = False
        
        # If active, shot target hovers 2x away for firing at system.
        self.system_2x_displace = False
        
        self.random = random.Random(seed)
        self.start()
        
        
    def compute_reward(self, activity):
        """
        Compute reward value for the specified activity that occured during the step.
        
        :param activity Activity as defined by DiscreteTennis.ACTIVITY_* enumerations that occured during the step.
        """
        return self.REWARD_MAP[activity]

    
    def on_end_game(self):
        self.player_2x_displace = False
        self.system_2x_displace = False
    
    def start(self):
        """
        Start this behavior. Must be called if any configuration changes are performed upon creation
        of an instance of this class.
        """
        self.player_reach = sqrt(self.player_x * self.player_x + self.player_y * self.player_y)
        self.system_reach = sqrt(self.system_x * self.system_x + self.system_y * self.system_y)
        
    def is_player_reachable(self, ball_position, player_position):
        """
        Check if the player can reach the shot ending with the ball at specified position while the
        player is at specified position.
        
        :param ball_position    Tuple (x, y, owner) describing ball's end position.
        :param player_position  Typle (x, y) describing player's current position.
        :return True if the player is reachable
        """
        delta_x = abs(ball_position[0] - player_position[0])
        delta_y = abs(ball_position[1] - player_position[1])
        distance = sqrt(delta_x * delta_x + delta_y * delta_y)

        # Distance to pickup a shot is unpredictable and some long reach ones seems bit 
        # further than they are.
        if self.player_reach_factor > 0.0:
            expand = 1.0 + self.random.uniform(0, self.player_reach_factor)
            distance = distance * expand
        return(distance < self.player_reach)
    
    def is_system_reachable(self, ball_position, system_position):
        """
        Check if the system can reach the shot ending with the ball at specified position while the
        system is at specified position.
        
        :param ball_position    Tuple (x, y, owner) describing ball's end position.
        :param system_position  Tuple (x, y) describing system's current position.
        :return True if the system is reachable
        """
        delta_x = abs(ball_position[0] - system_position[0])
        delta_y = abs(ball_position[1] - system_position[1])
        distance = sqrt(delta_x * delta_x + delta_y * delta_y)
        
        # Distance to pickup a shot is unpredictable and some long reach ones seems bit 
        # further than they are
        if self.system_reach_factor > 0.0:
            expand = 1.0 + self.random.uniform(0, self.system_reach_factor)
            distance = distance * expand
        return(distance < self.player_reach)
    
    def is_shot_valid(self, actor, shot_start, shot_end):
        """
        Check if the specified shot by the actor is a valid shot. Actor loses points for invalid
        shots as per rules of tennis.
        
        :param actor    Actor who hit the shot DiscreteTennis.PLAYER/DiscreteTennis.SYSTEM
        :param shot_start  Tuple (x, y) describing position of the start of shot.
        :param shot_end  Tuple (x, y) describing position of the end of shot (in peer's court).
        :return True if the shot is a valid shot.
        """
        
        # (0-left, 1-back, 2-right, 3-front) 
        
        shot_x = shot_end[0]
        shot_y = shot_end[1]
        
        if shot_x < 0.0 or shot_x > 1.0 or \
            shot_y < 0.0 or shot_y > 1.0:
            # Shot is outside the court
            return False
        
        in_left = shot_x >= 0 and shot_x <= self.border[0]
        in_back = shot_y >= 0  and shot_y <= self.border[1]
        in_right = shot_y >= (1.0 - self.border[2]) and shot_y <= 1.0
        in_front = shot_y >= 1.0 - self.border[3] and shot_y <= 1.0
        if not in_left and not in_back and not in_right and not in_front:
            return True # Internal shot  

        # Border shots are valid with only for configured valid probability.
        rand = self.random.random()
        if (in_left and rand > self.shot_valid_probability[0]) or \
            (in_back and rand > self.shot_valid_probability[1]) or \
            (in_right and rand > self.shot_valid_probability[2]) or \
            (in_front and rand > self.shot_valid_probability[3]):
            return False    # Border shots are invalid if sample exceeds the allowed valid threshold.
        
        return True
    
    def next_system_action(self, fire, serve, env):
        """
        Take the next action on behalf of the system. This method updates the environment as a result of taking
        the step. Additionally, push any renderable events to enable visualization if needed. The method assumes
        that for the system's fire-shot, the shot is a valid shot within the reach of the system.
        
        :param fire If true, this step is a fire step for the system where the system is expected to hit
            a shot, thereby putting the ball in the player's court.
        :param serve If true, this step is to fire a serve by the system.
        :param env Environment to be updated upon taking the system action.
        """
        if serve:
            env._set_system_position(DiscreteTennis.MIDDLE, DiscreteTennis.BASE_LINE)
            env._set_player_position(DiscreteTennis.MIDDLE, DiscreteTennis.BASE_LINE)
            env._set_ball_position(DiscreteTennis.MIDDLE, DiscreteTennis.BASE_LINE, DiscreteTennis.SYSTEM)
        
        ball = env._get_ball_position()
        system_pos = env._get_system_position()
        if fire or serve:
            # Since the ball is within reach, first move the system to allowed court spot closest to ball's position.
            if not serve:
                if ball[2] != DiscreteTennis.SYSTEM:
                    raise Exception("Invalid ball status for firing system: " + str(ball[2]))
                
                system_pos = self._snap_to_grid(ball)
                env._set_system_position(*system_pos)
                if (self._logger.isEnabledFor(logging.DEBUG)):
                    self._logger.debug(env._log_state() + 
                    "System Moves for firing to {}".format(system_pos), extra=env._LOG_TAG)
            
            # Player may move during the shot and therefore the system always takes its action BEFORE the
            # player moves to the final destination. This makes it unpredictable (like in real game) on 
            # where the player will be and what would be considered to be within reach of the player.

            # Decide the ball shot target based on player and difficultly level
            ball_end = self.shot_target(DiscreteTennis.SYSTEM, system_pos, env._get_player_position())
            src_ball = env._get_ball_source_position()
            
            # TODO: System serve rendering.
            if env._render_view:
                shot_start = env._position_to_point(env._to_render_position(DiscreteTennis.SYSTEM, ball))
                shot_end = env._position_to_point(env._to_render_position(DiscreteTennis.PLAYER, ball_end))
                transit_start = None
                if not serve:
                    prev_shot_start = env._position_to_point(env._to_render_position(DiscreteTennis.PLAYER, src_ball))
                    transit_start = RenderEvent.intersect_to(prev_shot_start, shot_start, DiscreteTennis.RENDER_COURT_MIDLINE)
                env._append_render_event(SystemShot(env._system, env._ball, shot_start, transit_start, shot_start, shot_end))
            
            env._set_ball_source_position(*ball)
            env._set_ball_position(*ball_end)
            
            if (self._logger.isEnabledFor(logging.INFO)):
                self._logger.info(env._log_state() + 
                "System Shot from {} to {}".format(ball, ball_end), extra=self._LOG_TAG)
        
        else:
            # Move the system for player's shot.
            
            # System attempts to stay in middle towards base-line as that location has best shot capabilities.
            delta_x = system_pos[0] - DiscreteTennis.MIDDLE
            delta_y = system_pos[1] - DiscreteTennis.BASE_LINE 
            
            new_position_x = system_pos[0]
            new_position_y = system_pos[1]
            if abs(delta_x) > self.system_x:
                # Movement in x-direction is needed.
                # Move to left.
                new_position_x = (system_pos[0] - self.system_x) if delta_x < 0 else (system_pos[0] + self.system_x)
                new_position_x = max(0.0, min(new_position_x, 1.0))     # Clamp between 0.0 and 1.0
            
            if abs(delta_y) > self.system_y:
                # Movement in y-direction is needed.
                new_position_y = (system_pos[1] - self.system_y) if delta_y > 0 else (system_pos[1] + self.system_x)
                new_position_y = max(0.0, min(new_position_y, 1.0))     # Clamp between 0.0 and 1.0

            if new_position_x != system_pos[0] or new_position_y != system_pos[1]:
                new_position = (new_position_x, new_position_y)
                # Move the system to the new-position
                env._set_system_position(*new_position)
                
                if env._render_view:
                    env._append_render_event(SystemMove(env._system, None, 
                        env._position_to_point(env._to_render_position(DiscreteTennis.SYSTEM, (new_position_x, new_position_y)))))
                    
                if (self._logger.isEnabledFor(logging.INFO)):
                    self._logger.info(env._log_state() + 
                    "System Moves from {} to {}".format(system_pos, (new_position_x, new_position_y)), extra=self._LOG_TAG)
            
    def _snap_to_grid(self, pos):
        """
        Snap the position to closest grid cell corner.
        
        :param pos Position to be snapped (x, y).
        :return Closest grid corner (x, y)
        """
        grid_x = int(round(pos[0]/self.cell_x)) * self.cell_x
        grid_y = int(round(pos[1]/self.cell_y)) * self.cell_y
        return((grid_x, grid_y))
     
     
    def _shot_target_by2x_displace(self, actor, actor_at, peer_position, ball_owner):
        """
        Determine the ball target position where shot will end based on 2x horizontal distance from peer.
        
        :param actor Actor who is hitting the ball.
        :param actor_at Tuple (x, y) of the actor's current position
        :param peer_position Tuple (x, y) of the peer's current position at the time actor fires the shot.
        :return Target ball position where the shot will end (x, y, owner) 
        """
        peer_2x = 2 * self.player_x if actor == DiscreteTennis.SYSTEM else 2 * self.system_x
        peer_at_x = peer_position[0] 
        
        to_left = (peer_at_x - peer_2x);
        to_left = to_left if to_left >= 0 else None
        to_right = (peer_at_x + peer_2x)
        to_right = to_right if to_right <= 1.0 else None
        
        if to_left is not None or to_right is not None:
            if to_left is not None and to_right is not None:
                use_left = bool(self.random.getrandbits(1))     # Shoot 2x randomly to left/right of peer's current position.
                return((to_left, peer_position[1], ball_owner) if use_left else (to_right, peer_position[1], ball_owner))   
            elif to_left is not None:
                return(to_left, peer_position[1], ball_owner)  # Shoot 2x to left of peer's current position.
            elif to_right is not None: 
                return(to_right, peer_position[1], ball_owner) # Shoot 2x to right of peer's current position.
        return self._shot_target_by_stretch(actor, actor_at, peer_position, None, ball_owner)
    
    def _shot_target_by_stretch(self, actor, actor_at, peer_position, is_difficult, ball_owner):
        
        if is_difficult is None:
            actor_difficulty = self.difficulty if actor == DiscreteTennis.PLAYER else (1.0 - self.difficulty)
            difficult_eval = self.random.random()
            is_difficult = difficult_eval < actor_difficulty
        
        peer_reach_x = self.player_x
        peer_reach = self.player_reach
        if actor == DiscreteTennis.SYSTEM:
            peer_reach_x = self.system_x
            peer_reach = self.system_reach
        
        target_x = 0
        target_y = 0
        
        if is_difficult:
            # Shot must be within the stretched distance over the reach capacity of the peer.
            expand = 1.0 + self.random.uniform(0, self.shot_target_stretch_factor)
            target_reach = peer_reach * expand
            target_x = self.random.uniform(peer_reach_x, peer_reach_x * expand)
            target_y = sqrt(target_reach * target_reach - target_x * target_x) 
        else:
            within = self.random.random()
            target_reach = within * peer_reach
            target_x = self.random.uniform(0, peer_reach_x * within)
            target_y = sqrt(target_reach * target_reach - target_x * target_x)
        
        # Randomize the ball target direction w.r.t.the peer 
        to_left = bool(self.random.getrandbits(1))
        to_front = bool(self.random.getrandbits(1))
        
        ball_x = peer_position[0]
        ball_y = peer_position[1]
        
        ball_x = (ball_x - target_x) if to_left else (ball_x + target_x)
        ball_x = max(0.0, min(ball_x, 1.0))
        
        ball_y = (ball_y + target_y) if to_front else (ball_y - target_y)
        ball_y = max(0.0, min(ball_y, 1.0))
        
        return(ball_x, ball_y, ball_owner)    

    def shot_target(self, actor, hitter_at, peer_position):
        """
        Determine the ball target position where shot will end based on the expected difficulty of the game and
        shot_target_stretch_factor configured for this behavior.
        
        :param actor Actor who is hitting the ball.
        :param hitter_at Tuple (x, y) of the actor's current position
        :param peer_position Tuple (x, y) of the peer's current position at the time actor fires the shot.
        :return Target ball position where the shot will end (x, y, owner) 
        """
        # Randomize distance of the ball from peer based on difficulty level.
        ball_owner = DiscreteTennis.PLAYER if actor == DiscreteTennis.SYSTEM else DiscreteTennis.SYSTEM
        
        if actor == DiscreteTennis.PLAYER and self.system_2x_displace or \
            actor == DiscreteTennis.SYSTEM and self.player_2x_displace:
            # 2x displacement on - compute shot target by 2x displacement.
            return self._shot_target_by2x_displace(actor, hitter_at, peer_position, ball_owner)
        
        # Is shot difficult - Put shot in the stretched reach region difficulty% times.
        actor_difficulty = self.difficulty if actor == DiscreteTennis.PLAYER else (1.0 - self.difficulty)
        
        difficult_eval = self.random.random()
        is_difficult = difficult_eval < actor_difficulty
        is_very_difficult = difficult_eval < (actor_difficulty * self.difficulty_2x_factor)
        
        if is_very_difficult:
            # Turn 2x displacement on until current game ends to greatly improve chance of actor winning.
            # Note that peer can still win if peer moves correctly by 1x and hits a shot that actor misses.
            if actor == DiscreteTennis.PLAYER:
                self.system_2x_displace = True 
                return self._shot_target_by2x_displace(actor, hitter_at, peer_position, ball_owner)
            elif actor == DiscreteTennis.SYSTEM:
                self.player_2x_displace = True  # Turn 2x displacement on until current game ends.
                return self._shot_target_by2x_displace(actor, hitter_at, peer_position, ball_owner)
            
        return self._shot_target_by_stretch(actor, hitter_at, peer_position, is_difficult, ball_owner)