# Refer: https://blog.paperspace.com/creating-custom-environments-openai-gym/
import logging
import random
from functools import partial
from math import sqrt
from typing import Optional as Op

import numpy as np
from rl.tennis.discreteTennis import (
    DiscreteTennis,
    RenderEvent,
    SystemShot,
    TennisBehavior,
)

LOGGER_BASE = "rl.tennis.discreteTennis"


class LearnableTennisBehavior(TennisBehavior):
    """
    Tennis behavior implementation that enables the user to specify a behavior to be exhibited by the
    environment. The behavior assumes a discrete game where the ball and player locations are at only
    designated places in the court.
    """

    _logger = logging.getLogger(LOGGER_BASE)
    _LOG_TAG = {"lgmod": "TNENV"}

    REWARD_MAP = {
        DiscreteTennis.ACTIVITY_SYSTEM_INVALID_SHOT: 5,
        DiscreteTennis.ACTIVITY_SYSTEM_MISS: 10,
        DiscreteTennis.ACTIVITY_SYSTEM_SHOT: -1,
        DiscreteTennis.ACTIVITY_PLAYER_INVALID_SHOT: -5,
        DiscreteTennis.ACTIVITY_PLAYER_MISS: -10,
        DiscreteTennis.ACTIVITY_PLAYER_SHOT: 1,
    }

    def __init__(self, seed=20):
        # Player and system positions snap to a grid with specified cell size.

        # Cell-size along X-axis to form grid for the court. 0.125 gives an 8x8 grid.
        self.cell_x = 0.125
        # Cell-size along Y-axis to form grid for the court.  0.125 gives an 8x8 grid.
        self.cell_y = 0.125

        # Maximum distance along X-axis to the ball that the player can reach for hitting a shot.
        self.player_x = 0.125
        # Maximum distance along Y-axis to the ball that the player can reach for hitting a shot.
        self.player_y = 0.125

        # Maximum distance along X-axis to the ball that the system can reach for hitting a shot.
        self.system_x = 0.125
        # Maximum distance along Y-axis to the ball that the system can reach for hitting a shot.
        self.system_y = 0.125

        # Distance from border when shots are vulnerable, specified along the court's (left, back, right, front) boundary.
        self.border = [0.125, 0.0, 0.125, 0.25]
        # Percentage of vulnerable shots that is valid along the (left, back, right, front) boundaries. Set it to 1.0 to disable this feature.

        # 100% System Success
        # self.shot_valid_probability = [0.9, 0.9, 0.9, 0.2]
        self.shot_valid_probability = [1.0, 1.0, 1.0, 1.0]

        # A difficulty scale to determine how easy it is for a player to score against system.
        # It varies from 0.0 for very easy to 1.0 for very hard.

        # Difficulty 0.9 - Moved from system to player winning in 12 episodes.
        # self.difficulty = 0.95
        self.difficulty = 0.95

        # 100% System Success
        # self.difficulty = 1.0

        # Even if shot is within reach, distance becomes inflated this many percentage of times.
        # Set them to 0.0 to disable this feature.
        self.player_reach_factor = 0.0
        self.system_reach_factor = 0.0

        self.random = random.Random(seed)

        self._shot_seq_factory = ShotSequenceFactory.get_default_factory()
        self._actor_mgmt: list[Op[ActorShotManagement]] = [None, None]
        # Player does not miss any within reach shot for reducing non-determinism.
        self._actor_mgmt[DiscreteTennis.PLAYER] = ActorShotManagement(0.0)
        # System misses some within reach shots depending on game difficulty. This is to make it easier for
        # player to win when difficulty level is low.

        # self._actor_mgmt[DiscreteTennis.SYSTEM] = ActorShotManagement((1 - self.difficulty)/4)
        self._actor_mgmt[DiscreteTennis.SYSTEM] = ActorShotManagement(0.0)
        self.start()

    def compute_reward(self, activity):
        """
        Compute reward value for the specified activity that occured during the step.

        :param activity Activity as defined by DiscreteTennis.ACTIVITY_* enumerations that occured during the step.
        """
        return self.REWARD_MAP[activity]

    def on_end_game(self):
        self._actor_mgmt[DiscreteTennis.PLAYER].clear_game()
        self._actor_mgmt[DiscreteTennis.SYSTEM].clear_game()
        self.time_step = -1

    def start(self):
        """
        Start this behavior. Must be called if any configuration changes are performed upon creation
        of an instance of this class.
        """
        self.player_reach = sqrt(
            self.player_x * self.player_x + self.player_y * self.player_y
        )
        self.system_reach = sqrt(
            self.system_x * self.system_x + self.system_y * self.system_y
        )
        self.time_step = -1

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

        if distance > self.player_reach:
            return False

        miss_prob = self._actor_mgmt[DiscreteTennis.PLAYER]._miss_probability
        if miss_prob > 0.0:
            miss = self.random.random()
            return miss > miss_prob

        return True

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

        if distance > self.player_reach:
            return False

        miss_prob = self._actor_mgmt[DiscreteTennis.SYSTEM]._miss_probability
        if miss_prob > 0.0:
            miss = self.random.random()
            return miss > miss_prob

        return True

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

        if shot_x < 0.0 or shot_x > 1.0 or shot_y < 0.0 or shot_y > 1.0:
            # Shot is outside the court
            return False

        in_left = shot_x >= 0 and shot_x <= self.border[0]
        in_back = shot_y >= 0 and shot_y <= self.border[1]
        in_right = shot_y >= (1.0 - self.border[2]) and shot_y <= 1.0
        in_front = shot_y >= 1.0 - self.border[3] and shot_y <= 1.0
        if not in_left and not in_back and not in_right and not in_front:
            return True  # Internal shot

        # Border shots are valid with only for configured valid probability.
        rand = self.random.random()
        if (
            (in_left and rand > self.shot_valid_probability[0])
            or (in_back and rand > self.shot_valid_probability[1])
            or (in_right and rand > self.shot_valid_probability[2])
            or (in_front and rand > self.shot_valid_probability[3])
        ):
            return False  # Border shots are invalid if sample exceeds the allowed valid threshold.

        return True

    def next_system_action(self, fire: bool, serve: bool, env: DiscreteTennis):
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
            env._set_ball_position(
                DiscreteTennis.MIDDLE, DiscreteTennis.BASE_LINE, DiscreteTennis.SYSTEM
            )

        ball = env._get_ball_position()
        system_pos = env._get_system_position()
        if fire or serve:
            # Since the ball is within reach, first move the system to allowed court spot closest to ball's position.
            if not serve:
                if ball[2] != DiscreteTennis.SYSTEM:
                    raise Exception(
                        "Invalid ball status for firing system: " + str(ball[2])
                    )

                # Linear progressions work well when entity moves before the shot. However, if system always
                # moves before each shot, it makes it very difficult for player to win. Therefore, depending
                # on current difficulty level, we purposefully do not always move the system giving the
                # player who has not yet learned to move before each shot a chance to still win.

                # Tune this for balancing system v/s player win performance:
                # move_threshold = (1.0 - (self.difficulty * 0.75))    # Player wins more often
                # move_threshold = 0;    # System always wins
                move_threshold = 1.0 - (
                    self.difficulty
                )  # System wins slightly more against untrained player with 50% difficulty.
                move_test = self.random.random()
                if move_test >= move_threshold:
                    system_pos = self._snap_to_grid(ball)
                    env._set_system_position(*system_pos)
                    if self._logger.isEnabledFor(logging.DEBUG):
                        self._logger.debug(
                            env._log_state()
                            + "System Moves for firing to {}".format(system_pos),
                            extra=env._LOG_TAG,
                        )

            # Player may move during the shot and therefore the system always takes its action BEFORE the
            # player moves to the final destination. This makes it unpredictable (like in real game) on
            # where the player will be and what would be considered to be within reach of the player.

            ball_end = self.shot_target(
                DiscreteTennis.SYSTEM, system_pos, env._get_player_position()
            )
            src_ball = env._get_ball_source_position()

            if env._render_view:
                system_pt = env._position_to_point(
                    env._to_render_position(DiscreteTennis.SYSTEM, system_pos)
                )
                shot_start = env._position_to_point(
                    env._to_render_position(DiscreteTennis.SYSTEM, ball)
                )
                shot_end = env._position_to_point(
                    env._to_render_position(DiscreteTennis.PLAYER, ball_end)
                )
                transit_start = None
                if not serve:
                    prev_shot_start = env._position_to_point(
                        env._to_render_position(DiscreteTennis.PLAYER, src_ball)
                    )
                    transit_start = RenderEvent.intersect_to(
                        prev_shot_start, shot_start, DiscreteTennis.RENDER_COURT_MIDLINE
                    )
                env._append_render_event(
                    SystemShot(
                        env._system,
                        env._ball,
                        system_pt,
                        transit_start,
                        shot_start,
                        shot_end,
                    )
                )

            env._set_ball_source_position(*ball)
            env._set_ball_position(*ball_end)

            if self._logger.isEnabledFor(logging.INFO):
                self._logger.info(
                    env._log_state()
                    + "System Shot from {} to {}".format(ball, ball_end),
                    extra=self._LOG_TAG,
                )
        # System never moves during the player shot.

    def _snap_to_grid(self, pos):
        """
        Snap the position to closest grid cell corner.

        :param pos Position to be snapped (x, y).
        :return Closest grid corner (x, y)
        """
        grid_x = int(round(pos[0] / self.cell_x)) * self.cell_x
        grid_y = int(round(pos[1] / self.cell_y)) * self.cell_y
        return (grid_x, grid_y)

    def shot_target(self, hitter, hitter_at, receiver_at):
        """
        Determine the ball target position where shot will end.

        :param hitter Actor who is hitting the ball.
        :param hitter_at Tuple (x, y) of the actor's current position
        :param receiver_at Tuple (x, y) of the peer's current position at the time actor fires the shot.
        :return Target ball position where the shot will end (x, y, owner)
        """
        self.time_step += 1
        receiver = (
            DiscreteTennis.PLAYER
            if hitter == DiscreteTennis.SYSTEM
            else DiscreteTennis.SYSTEM
        )
        shot_at = None
        for _ in range(10):
            seq = self._actor_mgmt[receiver]._shot_seq
            if seq is None:
                seq = self._shot_seq_factory.rand_seq(self, receiver, receiver_at)
                self._actor_mgmt[receiver]._shot_seq = seq
                if self._logger.isEnabledFor(logging.INFO):
                    self._logger.info(
                        "{}New sequence {}".format(DiscreteTennis._LOG_LINE, seq),
                        extra=self._LOG_TAG,
                    )

            shot_at = seq.shot_target(hitter_at, receiver_at, self.time_step)
            if shot_at is not None:
                return shot_at
            else:
                if self._logger.isEnabledFor(logging.INFO):
                    self._logger.info(
                        "{}Seq History: {}".format(
                            DiscreteTennis._LOG_LINE, seq._history
                        ),
                        extra=self._LOG_TAG,
                    )
                self._actor_mgmt[receiver].clear_game()
        return (receiver_at[0], receiver_at[1], receiver)


class ActorShotManagement:
    """
    Manage shot targeting strategy for an entity.
    """

    def __init__(self, difficulty):
        # Entity's difficulty level
        self._miss_probability = difficulty
        self.clear_game()

    def clear_game(self):
        """
        Clear shot-targeting sequence, if any, that is currently active for this entity.
        """
        self._shot_seq = None


class ShotSequence:
    """
    Interface representing a sequence of shot targets for a specific entity based on a specific learnable strategy.
    """

    def shot_target(self, hitter_at, receiver_at, time_step):
        """
        Determine the ball target position where shot hitter hits the shot at the receiver represented
        in this sequence with the shot hitter and receiver entities at the specified positions.

        :param hitter_at Tuple (x, y) of the shot hitter's current position
        :param receiver_at Tuple (x, y) of the shot receiver's current position at the time hitter fires the shot.
        :return Target ball position where the shot will end (x, y, receiver) if possible for this shot sequence
            else None
        """
        raise Exception("Not implemented - shot_target")


class ShotSequenceFactory:
    """
    Factory for creating a linear progression sequence to be used by the learnable behavior.
    """

    default_factory = None

    @classmethod
    def get_default_factory(cls):
        if cls.default_factory is not None:
            return cls.default_factory

        fac = ShotSequenceFactory()

        # fac.register_seq(Linear1xProgression.NAME,
        #     lambda behavior, receiver, start_at, nav_dir, step_delta, count_step, upto: \
        #     Linear1xProgression(receiver, start_at, nav_dir, step_delta, count_step, upto))

        fac.register_seq(
            Linear1xProgression.NAME,
            lambda behavior, receiver, start_at, nav_dir, step_delta, count_step, upto: LinearRandom1xProgression(
                behavior, receiver, start_at, nav_dir, step_delta, count_step, upto
            ),
        )

        # fac.register_seq(LinearRandom1xEnd2xProgression.NAME,
        #      lambda behavior, receiver, start_at, nav_dir, step_delta, count_step, upto: \
        #      LinearRandom1xEnd2xProgression(behavior, receiver, start_at, nav_dir, step_delta, count_step, upto))

        cls.default_factory = fac
        return cls.default_factory

    def __init__(self):
        """
        Constructor for linear progression sequence.
        """
        self._builder = {}
        self._reglist = []

    def register_seq(self, name, builder):
        self._builder[name] = builder
        self._reglist.append(name)

    def build_seq(self, seq_typename, behavior, receiver, start_at, nav_dir, upto=None):
        """
        Build a linear progression sequence with traits represented by seq_index for the specified receiver starting
        at the specified start_at position to move in the specified nav_dir direction.

        :param seq_typename Progression sequence typename registered for this factory.
        :param behavior Behavior to use this sequence.
        :param receiver Entity to receive the shot. Player/System
        :param start_at Tuple describing current position of the receiver.
        :param nav_dir Direction of sequence navigation as defined by DIR_ enumerations of this class.
        :param upto Optional position that limits the extent of the sequence navigation. If absent, the navigation
          continues until court boundary is reached.
        :return Linear progression sequence built here.
        """
        builder = self._builder.get(seq_typename)
        if builder is None:
            raise Exception(
                "No registered sequence type with name={}".format(seq_typename)
            )

        receiver_x = (
            behavior.player_x
            if receiver == DiscreteTennis.PLAYER
            else behavior.system_x
        )
        receiver_y = (
            behavior.player_y
            if receiver == DiscreteTennis.PLAYER
            else behavior.system_y
        )

        # Propagation can proceed in 8 directions - 4 F,L,B,R, and 4 FL, BL, BR, FR.
        # Randomly pick a direction and see if progression possible in that direction.
        near_net_y = 1 - receiver_y  # Cannot hit the net so stay 1x away from net.
        step_delta = None
        min_step = 3  # The progresion must span atleast this many steps forwards.

        step_x = 0
        step_y = 0
        total_steps = None
        if nav_dir == DiscreteTennis.DIR_FRONT:
            upto = (start_at[0], near_net_y) if upto is None else upto
            step_y = int((upto[1] - start_at[1]) / receiver_y)
            total_steps = step_y
            if step_y < min_step:
                return None  # Cannot navigate in this direction.
            step_delta = (0, receiver_y)

        elif nav_dir == DiscreteTennis.DIR_LEFT:
            upto = (0.0, start_at[1]) if upto is None else upto
            step_x = int((start_at[0] - upto[0]) / receiver_x)
            total_steps = step_x
            if step_x < min_step:
                return None  # Cannot navigate in this direction.
            step_delta = (-receiver_x, 0)

        elif nav_dir == DiscreteTennis.DIR_BACK:
            upto = (start_at[0], 0.0) if upto is None else upto
            step_y = int((start_at[1] - upto[1]) / receiver_y)
            total_steps = step_y
            if step_y < min_step:
                return None  # Cannot navigate in this direction.
            step_delta = (0, -receiver_y)

        elif nav_dir == DiscreteTennis.DIR_RIGHT:
            upto = (1.0, start_at[1]) if upto is None else upto
            step_x = int((upto[0] - start_at[0]) / receiver_x)
            total_steps = step_x
            if step_x < min_step:
                return None  # Cannot navigate in this direction.
            step_delta = (receiver_x, 0)

        elif nav_dir == DiscreteTennis.DIR_FRONTLEFT:
            upto = (0.0, near_net_y) if upto is None else upto
            step_x = int((start_at[0] - upto[0]) / receiver_x)
            step_y = int((upto[1] - start_at[1]) / receiver_y)
            if step_x < min_step or step_y < min_step:
                return None  # Cannot navigate in this direction.
            total_steps = min(step_x, step_y)
            upto = (
                start_at[0] - (receiver_x * total_steps),
                start_at[1] + (receiver_y * total_steps),
            )
            step_delta = (-receiver_x, receiver_y)

        elif nav_dir == DiscreteTennis.DIR_BACKLEFT:
            upto = (0.0, 0.0) if upto is None else upto
            step_x = int((start_at[0] - upto[0]) / receiver_x)
            step_y = int((start_at[1] - upto[1]) / receiver_y)
            if step_x < min_step or step_y < min_step:
                return None  # Cannot navigate in this direction.
            total_steps = min(step_x, step_y)
            upto = (
                start_at[0] - (receiver_x * total_steps),
                start_at[1] - (receiver_y * total_steps),
            )
            step_delta = (-receiver_x, -receiver_y)

        elif nav_dir == DiscreteTennis.DIR_BACKRIGHT:
            upto = (1.0, 0.0) if upto is None else upto
            step_x = int((upto[0] - start_at[0]) / receiver_x)
            step_y = int((start_at[1] - upto[1]) / receiver_y)
            if step_x < min_step or step_y < min_step:
                return None  # Cannot navigate in this direction.

            total_steps = min(step_x, step_y)
            upto = (
                start_at[0] + (receiver_x * total_steps),
                start_at[1] - (receiver_y * total_steps),
            )
            step_delta = (receiver_x, -receiver_y)

        elif nav_dir == DiscreteTennis.DIR_FRONTRIGHT:
            upto = (1.0, near_net_y) if upto is None else upto
            step_x = int((upto[0] - start_at[0]) / receiver_x)
            step_y = int((upto[1] - start_at[1]) / receiver_y)
            if step_x < min_step or step_y < min_step:
                return None  # Cannot navigate in this direction.

            total_steps = min(step_x, step_y)
            upto = (
                start_at[0] + (receiver_x * total_steps),
                start_at[1] + (receiver_y * total_steps),
            )
            step_delta = (receiver_x, receiver_y)

        else:
            return None

        return builder(
            behavior, receiver, start_at, nav_dir, step_delta, total_steps, upto
        )

    def rand_seq(self, behavior, receiver, start_at):
        """
        Build a linear progression sequence of a random type for the specified receiver starting
        at the specified start_at position to move in a random direction.

        :param behavior Behavior to use this sequence.
        :param receiver Entity to receive the shot. Player/System
        :param start_at Tuple describing current position of the receiver.
        :return Linear progression sequence built here.
        """
        reg_index = behavior.random.randrange(0, len(self._reglist))
        seq_typename = self._reglist[reg_index]
        examine_dir = set()

        # Examine each of the 8 possible directions but do so in random order.
        nav_dir = None
        while len(examine_dir) < DiscreteTennis.DIR_MAX:
            nav_dir = behavior.random.randrange(
                DiscreteTennis.DIR_MIN, DiscreteTennis.DIR_MAX
            )
            if nav_dir in examine_dir:
                continue  # Already examined this direction earlier - skip it.

            examine_dir.add(nav_dir)
            seq = self.build_seq(seq_typename, behavior, receiver, start_at, nav_dir)
            if seq is not None:
                return seq


class LinearProgression(ShotSequence):
    """
    A shot-sequence implementation that computes shots to be targeted at an entity in strict monotonically
    increasing/decreasing 1x distance from the last target. The progression could be horizontal, vertical
    or diagonal.
    """

    def __init__(self, receiver, start_at, nav_dir, step_delta, estimated_steps, upto):
        """
        :param receiver    Actor to receive the shots - player/system.
        :param start_at Tuple defining receiver's start (and end) position for this sequence.
        :param step_delta Tuple defining x,y progression distance for each step. (player_x,0) for horizontal
        increasing progression, (-player_x,0) for horizontal decreasing progression, (0,player_y) for vertical
        increasing progression, (0,-player_y) for vertical decreasing progression, (player_x,player_y) for
        diagonal increasing progression, (-player_x,-player_y) for diagonal decreasing progression. The start_at
        has to compatible with the progression direction such that there is atleast one or more steps that
        are possible from the start_at to the court boundary in the forward direction of progression.
        :param estimated_steps    Estimated number of forward steps. It need not be exact but mere hint of
        progression estimations.
        :param upto    If present, defines the maximum x,y bound upto which the progression must
        proceed before returning back to start. If absent, progression proceeds until the court
        boundary.
        """
        # Actor who will be receiving the shots as they linearly progress as computed by this shot-sequence.
        self._receiver = receiver
        # Progression start and end-point. The first shot will be self._step_delta away from here and the
        # last step will be at this position.
        self._start_at = start_at
        # Current position in progression from which the next shot is to be computed.
        self._shot_at = list(start_at)
        # Amount of (x, y) to move at each progression step.
        self._step_delta = list(step_delta)
        # Progression direction . True for first part of sequence until reaching court boundary.
        # False for second part of sequence from the court-boundary back to _startr_at.
        self._forward = True
        # Navigation direction
        self._nav_dir = nav_dir
        # Limit of progression.
        self._upto = upto

        self._history = None
        self._log_history = False
        self._estimated_steps = estimated_steps
        self._step_taken = 0

    def set_log_history(self, log_history):
        self._log_history = log_history
        if log_history:
            self._history = []
        else:
            self._history = None

    def shot_target(self, hitter_at, receiver_at, time_step):
        """
        Determine the ball target position where peer actor hits the shot at this actor with
        actor and peer entities at specified positions.

        :param hitter_at Tuple (x, y) of the shot hitter's current position
        :param receiver_at Tuple (x, y) of the shot receiver's current position at the time hitter fires the shot.
        :return Target ball position where the shot will end (x, y, receiver) if possible for this shot sequence
            else None
        """
        if not self._next_pos(time_step):
            return None
        ret = (self._shot_at[0], self._shot_at[1], self._receiver)
        self._step_taken = self._step_taken + 1
        if self._log_history:
            self._history.append(ret)
        return ret

    def _next_pos(self, time_step):
        """
        Upto the _shot_at to the position of next shot for the receiver represented in this shot sequence
        if such shot is possible.

        :return True if _shot_at was positioned to the next shot's position, False otherwise.
        """
        return False

    def _is_out_of_bound(self, pt_x, pt_y):
        """
        Check if the specified point is beyond the limits specified for this linear progression.

        :return True if the point is out-of-bound for the linear progression's limit.
        """
        return (
            (pt_x < 0.0 or pt_x > 1.0 or pt_y < 0.0 or pt_y > 1.0)
            or (self._step_delta[0] > 0 and pt_x > self._upto[0])
            or (self._step_delta[0] < 0 and pt_x < self._upto[0])
            or (self._step_delta[1] > 0 and pt_y > self._upto[1])
            or (self._step_delta[1] < 0 and pt_y < self._upto[1])
        )

    def _get_name(self):
        return None

    def __repr__(self):
        entity = "Player" if self._receiver == DiscreteTennis.PLAYER else "System"
        prog_name = self._get_name()
        dir_name = DiscreteTennis.DIR_NAME_MAP.get(self._nav_dir)[0]
        return "Receiver {}({}, {}): {}, {}".format(
            entity, self._start_at[0], self._start_at[1], prog_name, dir_name
        )


class Linear1xProgression(LinearProgression):
    """
    A shot-sequence implementation that computes shots to be targeted at an entity in strict monotonically
    increasing/decreasing 1x distance from the last target. The progression could be horizontal, vertical
    or diagonal.
    """

    NAME = "linear1x"

    def _get_name(self):
        return self.NAME

    def _next_pos(self, time_step):
        """
        Get the position of next shot for the receiver represented in this shot sequence if such shot is possible.

        :return True if next shot is possible. The _shot_at is positioned to the next shot's position.
        """
        if not self._forward and self._shot_at == self._start_at:
            # Reached back to starting position.
            return False

        next_x = self._shot_at[0] + self._step_delta[0]
        next_y = self._shot_at[1] + self._step_delta[1]

        if self._is_out_of_bound(next_x, next_y):
            if self._forward:
                # Reached the court boundary, reverse direction.
                self._forward = False
                self._step_delta = [self._step_delta[0] * -1, self._step_delta[1] * -1]
                self._upto = self._start_at
                return self._next_pos(time_step)
            else:
                return False  # End of progression.

        self._shot_at[0] = next_x
        self._shot_at[1] = next_y
        return True


class LinearRandom1xProgression(LinearProgression):
    """
    A shot-sequence implementation that computes shots to be targeted at an entity in a random
    increasing/decreasing 1x distance from the last target. The progression could be horizontal,
    vertical or diagonal. Probability of shot causing sequence to move forward is twice in forward
    direction when the progression is moving forward and vice-versa.
    """

    NAME = "linearRand1x"

    def __init__(
        self, behavior, receiver, start_at, nav_dir, step_delta, estimated_steps, upto
    ):
        super().__init__(receiver, start_at, nav_dir, step_delta, estimated_steps, upto)
        # Random generator to be used.
        self._random = behavior.random

    def _get_name(self):
        return self.NAME

    def _next_pos(self, time_step):
        """
        Get the position of next shot for the receiver represented in this shot sequence if such shot is possible.

        :return True if next shot is possible. The _shot_at is positioned to the next shot's position.
        """
        if not self._forward and self._shot_at == self._start_at:
            # Reached back to starting position.
            return False

        # Randomize until 80% steps traversed With 2:1 probability forward, subsequently, all steps strictly deterministic.
        reverse_multiplier = (
            -1
            if self._step_taken < int(self._estimated_steps * 0.8)
            and self._random.randint(0, 2) == 2
            else 1
        )  # Randoms: 0, 1, 2.

        next_x = self._shot_at[0] + reverse_multiplier * self._step_delta[0]
        next_y = self._shot_at[1] + reverse_multiplier * self._step_delta[1]

        # If while jumping around randomly around the start, if we go out of bound in other direction,
        # reverse direction and continue.
        if reverse_multiplier == -1 and (
            next_x < 0.0 or next_y < 0.0 or next_x > 1.0 or next_y > 1.0
        ):
            reverse_multiplier = 1
            next_x = self._shot_at[0] + reverse_multiplier * self._step_delta[0]
            next_y = self._shot_at[1] + reverse_multiplier * self._step_delta[1]

        if self._is_out_of_bound(next_x, next_y):
            if self._forward:
                # Reached the court boundary, reverse direction.
                self._forward = False
                self._step_delta = [self._step_delta[0] * -1, self._step_delta[1] * -1]
                self._upto = self._start_at
                return self._next_pos(time_step)
            else:
                return False  # End of progression.

        self._shot_at[0] = next_x
        self._shot_at[1] = next_y
        return True

    def __repr__(self):
        return super().__repr__()


class LinearRandom1xEnd2xProgression(LinearProgression):
    """
    A shot-sequence implementation that computes shots to be targeted at an entity in a random
    increasing/decreasing 1x distance from the last target. The progression could be horizontal,
    vertical or diagonal. However, if the last target is at 2x position from the edge of the court,
    shot target will be at increasing 2x and if shot target is 1x position from edge of the court,
    shot target will be decreasing 2x. Probability of shot causing sequence to move forward is
    twice in forward direction when the progression is moving forward and vice-versa.
    """

    NAME = "linearrand1xend2x"

    def __init__(
        self, behavior, receiver, start_at, nav_dir, step_delta, estimated_steps, upto
    ):
        super().__init__(receiver, start_at, nav_dir, step_delta, estimated_steps, upto)
        # Random generator to be used.
        self._random = behavior.random
        self._forward_jump = (
            self._upto[0] - 2 * self._step_delta[0],
            self._upto[1] - 2 * self._step_delta[1],
        )
        self._backward_jump = (
            self._upto[0] - self._step_delta[0],
            self._upto[1] - self._step_delta[1],
        )

    def _get_name(self):
        return self.NAME

    def _next_pos(self, time_step):
        """
        Get the position of next shot for the receiver represented in this shot sequence if such shot is possible.

        :return True if next shot is possible. The _shot_at is positioned to the next shot's position.
        """
        if not self._forward and self._shot_at == self._start_at:
            # Reached back to starting position.
            return False

        if (
            self._forward_jump is not None
            and self._forward
            and self._shot_at[0] == self._forward_jump[0]
            and self._shot_at[1] == self._forward_jump[1]
        ):
            next_x = self._upto[0]
            next_y = self._upto[1]
            self._forward_jump = None
        elif (
            self._backward_jump is not None
            and not self._forward
            and self._shot_at[0] == self._backward_jump[0]
            and self._shot_at[1] == self._backward_jump[1]
        ):
            next_x = self._shot_at[0] + 2 * self._step_delta[0]
            next_y = self._shot_at[1] + 2 * self._step_delta[1]
            self._backward_jump = None
        else:
            # Randomize until 80% steps traversed With 2:1 probability forward, subsequently, all steps strictly deterministic.
            reverse_multiplier = (
                -1
                if self._step_taken < int(self._estimated_steps * 0.8)
                and self._random.randint(0, 2) == 2
                else 1
            )  # Randoms: 0, 1, 2.

            next_x = self._shot_at[0] + reverse_multiplier * self._step_delta[0]
            next_y = self._shot_at[1] + reverse_multiplier * self._step_delta[1]

            # If while jumping around randomly around the start, if we go out of bound in other direction,
            # reverse direction and continue.
            if reverse_multiplier == -1 and (
                next_x < 0.0 or next_y < 0.0 or next_x > 1.0 or next_y > 1.0
            ):
                reverse_multiplier = 1
                next_x = self._shot_at[0] + reverse_multiplier * self._step_delta[0]
                next_y = self._shot_at[1] + reverse_multiplier * self._step_delta[1]

        if self._is_out_of_bound(next_x, next_y):
            if self._forward:
                # Reached the court boundary, reverse direction.
                self._forward = False
                self._step_delta = [self._step_delta[0] * -1, self._step_delta[1] * -1]
                self._upto = self._start_at
                return self._next_pos(time_step)
            else:
                return False  # End of progression.

        self._shot_at[0] = next_x
        self._shot_at[1] = next_y
        return True

    def __repr__(self):
        return super().__repr__()


class ExtremeProgression(Linear1xProgression):
    """
    Always shoots as far right or left as possible
    """

    NAME = "extreme"

    def __init__(self, *args, direction: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.direction: int = direction

    def _next_pos(self, time_step) -> bool:
        """
        Get the position of next shot for the receiver represented in this shot sequence if such shot is possible.

        :return True if next shot is possible. The _shot_at is positioned to the next shot's position.
        """
        # Adjust x
        mult = 1 if time_step == 0 else 2  # only move 1x on first step, then get harder
        next_x = self._shot_at[0] + self.direction * 0.125 * mult
        if self._is_out_of_bound(next_x, self._shot_at[1]):
            # TODO add randomness
            pass
        else:
            self._shot_at[0] = next_x

        # Adjust y
        next_y = self._shot_at[1] + self._step_delta[1]
        if not self._is_out_of_bound(self._shot_at[0], next_y):
            self._shot_at[1] = next_y

        return True


class TennisBehaviorShotRewardOnly(LearnableTennisBehavior):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.difficulty = 1.0
        self.difficulty = 0.95
        self.player_shot_seq_factory = ShotSequenceFactory.get_default_factory()
        self.reset()

    def reset(self):
        super().reset()
        self.direction = self.random.choice([-1, 1])
        self.system_shot_seq_factory = ShotSequenceFactory()
        self.system_shot_seq_factory.register_seq(
            ExtremeProgression.NAME,
            lambda behavior, receiver, start_at, nav_dir, step_delta, count_step, upto: ExtremeProgression(
                receiver,
                start_at,
                nav_dir,
                step_delta,
                count_step,
                upto,
                direction=self.direction,
            ),
        )

    def shot_target(self, hitter, hitter_at, receiver_at):
        if hitter == DiscreteTennis.PLAYER:
            self._shot_seq_factory = self.player_shot_seq_factory
        else:
            self._shot_seq_factory = self.system_shot_seq_factory
        return super().shot_target(hitter, hitter_at, receiver_at)

    def is_player_reachable(self, ball_position, player_position):
        delta_x = abs(ball_position[0] - player_position[0])
        delta_y = abs(ball_position[1] - player_position[1])
        distance = sqrt(delta_x * delta_x + delta_y * delta_y)

        if distance > 0.15:
            return False

        miss_prob = self._actor_mgmt[DiscreteTennis.PLAYER]._miss_probability
        if miss_prob > 0.0:
            miss = self.random.random()
            return miss > miss_prob

        return True