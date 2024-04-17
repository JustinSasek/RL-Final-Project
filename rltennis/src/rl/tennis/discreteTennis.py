# Refer: https://blog.paperspace.com/creating-custom-environments-openai-gym/
import logging
import os
import sys

import cv2
import numpy as np
from gymnasium import Env, spaces

font = cv2.FONT_HERSHEY_COMPLEX_SMALL

LOGGER_BASE = "rl.tennis.discreteTennis"
LOGGER_ACTION = "rl.tennis.discreteTennis.action"


class ActionSpec:
    """
    Action specification that defines the value, meaning and additional useful runtime artifacts associated
    with allowed agent actions for the tennis game.
    """

    def __init__(self, value, name, firing, left, back, right, front):
        """
        Action specification

        :param value     Value of action as known in interface with the user for this environment
        :param name      Name of this action
        :param firing    True if this action involves firing the shot
        :param left      Units of left movement
        :param back      Units of back movement
        :param right     Units of right movement
        :param front     Units of front movement
        """
        self.value = value
        self.name = name
        self.is_firing = firing
        self.left = left
        self.right = right
        self.front = front
        self.back = back
        self.is_moving = left != 0 or back != 0 or right != 0 or front != 0


# Tennis environment with stateful observation space based on player and ball positioning.
#
# Coordinate systems
#
# There are two coordinate systems that are used for managing the position of player, system and ball.
# The game's state coordinate system is the system that is used for stateful management of the player/system and
# ball whereas the render coordinate system is used to show a view of the game.
#
# State Coordinate System: The court is divided symmetrically into two parts: Player Game Part and
# System Game Part. Each part's coordinate system 0 is near its left baseline w.r.t. the entity facing
# the net. For player this is top-right of canvas and for system, this is bottom-left of canvas as the
# player always plays near the top of canvas and system always plays near the bottom of canvas. So with
# respect to the entity, the x coordinate decreases to its left to 0 and increases to its right to 1 and
# y-coordinate decreases to 0 towards baseline and increases to 1 towards net. This makes the positioning and
# movement relative to entity symmetrical and consistent for both player and system and represents it as a
# percentage of the entity's side of the court.
#
# Render Coordinate System: In this system, the 0 lies in the top-left corner and x increases towards
# right upto maximum canvas width in pixels and y increases towards the bottom upto the maximum canvas
# height in pixels.
#
# Directions:
# Left: Player perceives the ball to his left.
# Right: Player perceives the ball to his right.
# Front: Player perceives the ball to be in front (shallow). a.k.a. Up
# Back: Player perceives the ball to be in back (deep). a.k.a Down
#
# Position represents a tuple with (x, y) whereas Point represents an instance of Point class with Point.x, Point.y.


class DiscreteTennis(Env):
    _logger = logging.getLogger(LOGGER_BASE)
    _action_logger = logging.getLogger(LOGGER_ACTION)
    _LOG_TAG = {"lgmod": "TNENV"}
    _LOG_LINE = "\n\t"

    # Threshold for the main horizontal lines in a tennis court: (base-line, serve-line, net-shot)
    Y_THRESHOLD = (0.0, 0.6, 0.96)
    # Threshold positions for main vertical lines in a tennis court: (left, middle, right)
    # For singles, the render mapping must map to the internal lines and doubles it must map to external lines.
    X_THRESHOLD = (0.0, 0.5, 1.0)

    # Constants for commonly used aspects of tennis game.
    # Baseline position y-value.
    BASE_LINE = Y_THRESHOLD[0]
    # Serve-line position y-value.
    SERVE_LINE = Y_THRESHOLD[1]
    # Net-shot position y-value.
    NET_LINE = Y_THRESHOLD[2]

    # Left line of court x-value.
    LEFT = X_THRESHOLD[0]
    # Middle line of court x-value.
    MIDDLE = X_THRESHOLD[1]
    # Right line of court x-value.
    RIGHT = X_THRESHOLD[2]

    # Index in state array for stateful artifacts of the game.

    # Player's y-position
    STATE_PLAYER_Y = 0

    # Player's x-position
    STATE_PLAYER_X = 1

    # System's y-position
    STATE_SYSTEM_Y = 2

    # System's x-position
    STATE_SYSTEM_X = 3

    # Ball's owner
    STATE_BALL_OWNER = 4

    # Ball's y position
    STATE_BALL_Y = 5

    # Ball's x position
    STATE_BALL_X = 6

    # Ball source's owner - ball is hit by source-owner to the current owner.
    STATE_SRCBALL_OWNER = 7

    # Ball source's y position
    STATE_SRCBALL_Y = 8

    # Ball source's x position
    STATE_SRCBALL_X = 9

    # Who is currently serving PLAYER/SYSTEM
    STATE_SERVE = 10

    # Player's score within current game.
    STATE_GAMESCORE_PLAYER = 11

    # System's score within current game
    STATE_GAMESCORE_SYSTEM = 12

    # Player's score of games within current set.
    STATE_SETSCORE_PLAYER = 13

    # System's score of games within current set.
    STATE_SETSCORE_SYSTEM = 14

    # Player's score of number of sets in match.
    STATE_MATCHSCORE_PLAYER = 15

    # System's score of number of sets in match.
    STATE_MATCHSCORE_SYSTEM = 16

    # Current game's completion - 0 means continues, +1 means player won, -1 means system won.
    STATE_CURRENTGAME_RESULT = 17

    STATE_SIZE = 18

    # Constant representing which side of the court is being represented.
    # Player/player-side of court.
    PLAYER = 0
    # System/system-side of court.
    SYSTEM = 1

    ACTIVITY_SYSTEM_INVALID_SHOT = (1,)
    ACTIVITY_SYSTEM_MISS = (2,)
    ACTIVITY_SYSTEM_SHOT = (3,)
    ACTIVITY_PLAYER_INVALID_SHOT = (4,)
    ACTIVITY_PLAYER_MISS = (5,)
    ACTIVITY_PLAYER_SHOT = (6,)

    # Rendering view design.

    # View dimensions of the tennis court.
    COURT = (240, 520)  # (width, height)

    # View dimensions of the overall view canvas where visual artifacts reside.
    RENDER_SHAPE = (800, 400, 4)  # (height, width, #channels)

    CANVAS_LIMIT = (RENDER_SHAPE[1], RENDER_SHAPE[0])

    # Margin around the court.
    MARGIN = (80, 140, 80, 140)  # (left, top, right, bottom)

    # Render whole court (singles and doubles) position of the court in the view.
    RENDER_COURT = [int(MARGIN[0] + (COURT[0] / 2)), int(MARGIN[1] + (COURT[1] / 2))]
    RENDER_COURT_MIDLINE = RENDER_SHAPE[0] / 2

    # Render offset from top of view of important player-side horizontal lines (Baseline, serveline, netline)
    # 0, (140, 260, 394), 400
    RENDER_PLAYER_Y = [
        MARGIN[1],
        int(MARGIN[1] + (COURT[1] * 0.27)),
        int(MARGIN[1] + (COURT[1] * 0.49)),
    ]

    # Render offset from top of view of important system-side horizontal lines (Baseline, serveline, netline)
    # 400, (406, 540, 660), 800
    RENDER_SYSTEM_Y = [
        MARGIN[1] + COURT[1],
        int(MARGIN[1] + (COURT[1] * 0.77)),
        int(MARGIN[1] + (COURT[1] * 0.51)),
    ]

    # Render offset from edges depending on type of game (singles, doubles)
    # (30, 0)
    RENDER_GAMETYPE_XOFFSET = [int(0.125 * COURT[0]), 0]

    # Render offset from left edge of canvas of important vertical lines (left, middle, right)
    # 0, [80, (110, 200, 290), 320], 400
    RENDER_X = [MARGIN[0], int(MARGIN[0] + (COURT[0] / 2)), MARGIN[0] + COURT[0]]

    # Render score position in the view (x, y)
    RENDER_SCORE_POSITION = (MARGIN[0], int(MARGIN[1] + COURT[1] + 50))

    # Game score map
    RENDER_GAMESCORE = {0: "0", 1: "15", 2: "30", 3: "40", 4: "Ad", 5: "Win"}

    # Set score map
    RENDER_SETSCORE = {
        0: "0",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "8",
        9: "9",
    }

    # MAtch score map
    RENDER_MATCHSCORE = {0: "0", 1: "1", 2: "2"}

    # Advantage score
    GAMESCORE_AD = 4

    # Scoe of 40 - ad only of other is also 40.
    GAMESCORE_40 = 3

    # Increment a single game score.
    GAMESCORE_DELTA = 1

    # Increment a single set score.
    SETSCORE_DELTA = 1

    # Increment a sets in match score.
    MATCHSCORE_DELTA = 1

    # Minimum game-score
    GAMESCORE_MINIMUM = 0.0

    # Maximum game-score
    GAMESCORE_MAXIMUM = 0.5

    # Minimum set-score
    SETSCORE_MINIMUM = 0.0

    # Maximum set-score
    SETSCORE_MAXIMUM = 1.0

    # Minimum set-score
    MATCHSCORE_MINIMUM = 0.0

    # Maximum set-score
    MATCHSCORE_MAXIMUM = 0.2

    GAMESTATUS_CONTINUE = 0

    GAMESTATUS_NEWGAME = 1

    GAMESTATUS_NEWSET = 2

    GAMESTATUS_COMPLETE = 3

    # Possible action map: ActionSpec(Value, Name, Firing, Left, Back, Right, Front)
    ACTION_SPEC = {
        0: ActionSpec(0, "No operation", False, 0, 0, 0, 0),
        1: ActionSpec(1, "Fire", True, 0, 0, 0, 0),
        2: ActionSpec(2, "Move front", False, 0, 0, 0, 1),
        3: ActionSpec(3, "Move right", False, 0, 0, 1, 0),
        4: ActionSpec(4, "Move left", False, 1, 0, 0, 0),
        5: ActionSpec(5, "Move back", False, 0, 1, 0, 0),
        6: ActionSpec(6, "Move front-right", False, 0, 0, 1, 1),
        7: ActionSpec(7, "Move front-left", False, 1, 0, 0, 1),
        8: ActionSpec(8, "Move back-right", False, 0, 1, 1, 0),
        9: ActionSpec(9, "Move back-left", False, 1, 1, 0, 0),
        10: ActionSpec(10, "Fire front", True, 0, 0, 0, 1),
        11: ActionSpec(11, "Fire right", True, 0, 0, 1, 0),
        12: ActionSpec(12, "Fire left", True, 1, 0, 0, 0),
        13: ActionSpec(13, "Fire back", True, 0, 1, 0, 0),
        14: ActionSpec(14, "Fire front-right", True, 0, 0, 1, 1),
        15: ActionSpec(15, "Fire front-left", True, 1, 0, 0, 1),
        16: ActionSpec(16, "Fire back-right", True, 0, 1, 1, 0),
        17: ActionSpec(17, "Fire back-left", True, 1, 1, 0, 0),
    }
    ACTION_NOFIRE_GEN = [0, 2, 3, 4, 5, 6, 7, 8, 9]
    ACTION_FIRE_GEN = [1, 10, 11, 12, 13, 14, 15, 16, 17]

    ACTOR_NAME_MAP = {PLAYER: "Player", SYSTEM: "SYstem"}

    DIR_FRONT = 0  # Progress in front of entity.
    DIR_LEFT = 1  # Progress to left of entity.
    DIR_BACK = 2  # Progress to behind the entity.
    DIR_RIGHT = 3  # Progress to right of the entity.
    DIR_FRONTLEFT = 4  # Progress diagonally to front-left of the entity.
    DIR_BACKLEFT = 5  # Progress diagonally to back-left of the entity.
    DIR_BACKRIGHT = 6  # Progress diagonally to back-right of the entity.
    DIR_FRONTRIGHT = 7  # Progress diagonally to front-right of the entity.
    DIR_NONE = 8  # No movement.
    DIR_MIN = 0  # Inclusive minimum value of valid direction.
    DIR_MAX = 8  # Exclusive maximum value of valid direction (without None).
    # Dictionary mapping the direction values to a human readable name for this direction
    DIR_NAME_MAP = {
        DIR_FRONT: ["Front", " F"],
        DIR_LEFT: ["Left", " L"],
        DIR_BACK: ["Back", " B"],
        DIR_RIGHT: ["Right", " R"],
        DIR_FRONTLEFT: ["Front-Left", "FL"],
        DIR_BACKLEFT: ["Back-Left", "BL"],
        DIR_BACKRIGHT: ["Back-Right", "BR"],
        DIR_FRONTRIGHT: ["Front-Right", "FR"],
        DIR_NONE: ["Nones", "NO"],
    }

    DIR_ACTION_MAP = {
        DIR_NONE: [0, 1],
        DIR_FRONT: [2, 10],
        DIR_RIGHT: [3, 11],
        DIR_LEFT: [4, 12],
        DIR_BACK: [5, 13],
        DIR_FRONTRIGHT: [6, 14],
        DIR_FRONTLEFT: [7, 15],
        DIR_BACKRIGHT: [8, 16],
        DIR_BACKLEFT: [9, 17],
    }

    def __init__(self, behavior):
        """
        Environment constructor.

        :param behavior    Tennis behavior to be used to control the behavior of this environment
        """
        super().__init__()

        self._behavior = behavior

        # Define a 2-D observation space
        self.observation_shape = (self.STATE_SIZE,)
        self.observation_space = spaces.Box(
            0.0, 1.0, self.observation_shape, np.float32
        )

        # Define an action space ranging from 0 to 4
        self.action_space = spaces.Discrete(
            18,
        )

        # Create a canvas to render the environment images upon (height, width, #channel)
        self.render_shape = self.RENDER_SHAPE
        self.canvas = np.ones(self.render_shape) * 1
        self.pristine_canvas = np.ones(self.render_shape) * 1

        self._court = Court("Court")
        self._player = Player("Player")
        self._system = System("System")
        self._ball = Ball("Ball")
        self.elements = [self._court, self._player, self._system, self._ball]
        self._court.set_position(self.RENDER_COURT[0], self.RENDER_COURT[1])
        self._court.show_at(self.pristine_canvas)

        self._state = np.zeros(self.observation_shape, dtype=np.float32)
        self._game_type = 0  # singles
        gametype_offset = self.RENDER_GAMETYPE_XOFFSET[self._game_type]
        self._render_x = (
            self.RENDER_X[0] + gametype_offset,
            self.RENDER_X[1],
            self.RENDER_X[2] - gametype_offset,
        )
        # Both player (rows 0 - 2) and system (rows 3 - 5) are in middle on base-lines.
        self._reset_game()
        self._render_view = False
        self._render_arr = []
        self._action_meaning_map = None
        self._animation_tester = None
        self.episode_done = False
        self._curr_episode = 0
        self._score_history = []
        self._stats_logger = None

        self.captureAction = None
        if self._action_logger.isEnabledFor(logging.INFO):
            self._action_logger.info("Starting a new game", extra=self._LOG_TAG)

    def set_stats_logger(self, stats_logger):
        """
        Set a new statistics logger

        :param stats_logger New statistics logger
        """
        if stats_logger is None:
            return
        old_stats_logger = self._stats_logger
        self._stats_logger = stats_logger
        return old_stats_logger

    def get_stats_logger(self):
        """
        Get current statistics logger

        :return Current statistics logger
        """
        return self._stats_logger

    def set_capture_action(self, capture_file):
        self.captureAction = open(capture_file, "w")

    def save_action(self, action):
        if self.captureAction is not None:
            buf = "{}\n".format(action)
            self.captureAction.write(buf)
            self.captureAction.flush()

    def set_animation_test(self, tester):
        self._animation_tester = tester

    def _reset_game(self):
        self._set_player_position(self.MIDDLE, self.BASE_LINE)
        self._set_system_position(self.MIDDLE, self.BASE_LINE)
        self._set_ball_position(self.MIDDLE, self.BASE_LINE, self.PLAYER)
        self._set_server(self.SYSTEM)
        self._set_score(0, 0, 0, 0, 0, 0)
        self._set_current_game_result(0)

        # Counter of number of steps that have occured in the current set.
        self._game_step_count = 0

    def get_state(self):
        """
        Get the current state of the environment.

        :return Current state of the environment.
        """
        return self._state

    def _get_logger(self):
        return self._logger

    def _get_player_position(self):
        return TennisObservedState.get_player_position(self._state)

    def _get_system_position(self):
        return TennisObservedState.get_system_position(self._state)

    def _get_ball_position(self):
        return TennisObservedState.get_ball_position(self._state)

    def _get_ball_source_position(self):
        return TennisObservedState.get_ball_source_position(self._state)

    def _get_server(self):
        return TennisObservedState.get_server(self._state)

    def _get_score(self):
        return TennisObservedState.get_score(self._state)

    def _get_score_num(self):
        return TennisObservedState.get_score_num(self._state)

    def _get_current_game_result(self):
        return TennisObservedState.get_current_game_result(self._state)

    def _set_player_position(self, x, y):
        return TennisState.set_player_position(self._state, x, y)

    def _set_system_position(self, x, y):
        return TennisState.set_system_position(self._state, x, y)

    def _set_ball_position(self, x, y, owner):
        return TennisState.set_ball_position(self._state, x, y, owner)

    def _set_ball_source_position(self, x, y, owner):
        return TennisState.set_ball_source_position(self._state, x, y, owner)

    def _set_server(self, server):
        return TennisState.set_server(self._state, server)

    def _set_score(
        self,
        player_game_score,
        system_game_score,
        player_set_score,
        system_set_score,
        player_match_score,
        system_match_score,
    ):
        return TennisState.set_score(
            self._state,
            player_game_score,
            system_game_score,
            player_set_score,
            system_set_score,
            player_match_score,
            system_match_score,
        )

    def _set_score_num(
        self,
        player_game_score,
        system_game_score,
        player_set_score,
        system_set_score,
        player_match_score,
        system_match_score,
    ):
        return TennisState.set_score_num(
            self._state,
            player_game_score,
            system_game_score,
            player_set_score,
            system_set_score,
            player_match_score,
            system_match_score,
        )

    def _set_current_game_result(self, value):
        return TennisState.set_current_game_result(self._state, value)

    def _to_render_y(self, side, y):
        mapper = self.RENDER_PLAYER_Y if side == self.PLAYER else self.RENDER_SYSTEM_Y
        return self._state_to_render_pos(y, self.Y_THRESHOLD, mapper)

    def _to_render_x(self, side, x):
        rendered_x = self._state_to_render_pos(x, self.X_THRESHOLD, self._render_x)
        return rendered_x if side != self.PLAYER else self.CANVAS_LIMIT[0] - rendered_x

    def _to_render_position(self, side, position):
        """
        Get the position in the render coordinate system for the specified (x, y) position.

        :param side    Side of court PLAYER/SYSTEM
        :param position (x,y) position for which render coordinate is needed.
        :return Render coordinates for the specified state coordinate system (x, y) position.
        """
        return (
            self._to_render_x(side, position[0]),
            self._to_render_y(side, position[1]),
        )

    def _to_render_point(self, side, point):
        """
        Get the point in the render coordinate system for the specified state coordinate system point.

        :param side    Side of court PLAYER/SYSTEM
        :param point   Point in state coordinate system point.
        :return Point in render coordinates for the specified state coordinate system point.
        """
        return Point(self._to_render_x(side, point.x), self._to_render_y(side, point.y))

    def _state_to_render_pos(self, state, threshold, mapper):
        # y value lies between these two mapper indices.
        low_index = -1
        high_index = -1
        for index in range(len(threshold)):
            if (
                abs(state - threshold[index]) < 0.001
            ):  # Exact y-value as defined in mapper threshold.
                return mapper[index]

            if state > threshold[index]:
                low_index = index
            if state < threshold[index]:
                high_index = index
                break
        if high_index == -1:
            high_index = len(threshold) - 1
        if low_index == -1:
            low_index = 0
        percent = 0
        if low_index < high_index:
            percent = (state - threshold[low_index]) / (
                threshold[high_index] - threshold[low_index]
            )
        elif low_index == len(threshold) - 1:
            # If exceeds max threshold, extrapolate at rate of distribution of entire threshold.
            return int(
                mapper[low_index]
                + (
                    (mapper[low_index] / threshold[low_index])
                    * (state - threshold[low_index])
                )
            )
        else:
            raise Exception("Invalid state position " + str(state))
        ret_value = mapper[low_index] + (
            percent * (mapper[high_index] - mapper[low_index])
        )
        return int(ret_value)

    def _append_render_event(self, event):
        """
        Append a render event for the current step in this environment.

        :param event RenderEvent instance being appended.
        """
        if event is None:
            return

        # Set the peer actor's move within the actor's shot event so that both can be rendered simultaneously.
        if len(self._render_arr) > 0:
            last_event = self._render_arr[-1]
            if last_event.name == "playerShot" and event.name == "systemMove":
                # Merge the player-shot with system-move.
                last_event._move_peer_actor_event = event
                return
            elif last_event.name == "systemShot" and event.name == "playerMove":
                last_event._move_peer_actor_event = event
                return

        self._render_arr.append(event)

    def _clean_render_events(self):
        self._render_arr.clear()

    def _draw_elements_on_canvas(self):
        # Init the canvas
        self.canvas = np.ones(self.render_shape) * 1

        x, y = self._get_player_position()
        player = self.elements[1]
        player.x = self._to_render_x(self.PLAYER, x)
        player.y = self._to_render_y(self.PLAYER, y)

        x, y = self._get_system_position()
        system = self.elements[2]
        system.x = self._to_render_x(self.SYSTEM, x)
        system.y = self._to_render_y(self.SYSTEM, y)

        # Adjust the elements based on current state.
        # x, y, owner = self._get_ball_position()
        # ball = self.elements[3]
        # ball.x =  self._to_render_x(owner, x)
        # ball.y =  self._to_render_y(owner, y)

        # Draw the elements on canvas
        for elem in self.elements:
            elem.show_at(self.canvas)

        RenderEvent.render_score(self.canvas, self._get_score())

    def reset(self):
        self._reset_game()

        if self._render_view:
            # Draw elements on the canvas
            self._draw_elements_on_canvas()

        # return the observation
        return self._state

    def render(self, mode="human"):
        if self._render_view:
            assert mode in [
                "human",
                "rgb_array",
            ], 'Invalid mode, must be either "human" or "rgb_array"'
            if mode == "human":
                if len(self._render_arr) > 0:
                    for event in self._render_arr:
                        event.render_event(self.canvas, self.pristine_canvas)
                else:
                    RenderEvent.render_frame(self.canvas)
                # canvas same format as returned by imread i.e. of shape (h, w, #channel)
                # cv2.imshow("Game", self.canvas)
                cv2.waitKey(10)

            elif mode == "rgb_array":
                return self._state

    def close(self):
        if self._stats_logger is not None:
            self._stats_logger.close()
            self._stats_logger = None
        if self.captureAction is not None:
            self.captureAction.close()
            self.captureAction = None
        cv2.destroyAllWindows()

    def get_action_meanings(self):
        if self._action_meaning_map is not None:
            return self._action_meaning_map
        self._action_meaning_map = {}
        for x in self.ACTION_SPEC:
            self._action_meaning_map[x] = self.ACTION_SPEC[x].name
        return self._action_meaning_map

    def get_action_name(self, action_id):
        return self.get_action_meanings().get(action_id, "Unknown-" + str(action_id))

    def step(self, action):
        """
        Step to perform an action by the agent causing an update of state and resulting
        a reward.

        Step is entered once while ball is in each actor's court. Game alternates between fire-step and
        position-step.

        Fire-step:
        When ball is in player's court, player typically have to choose a fire-action that allows the player
        to move the ball to the system's court.

        Position-step:
        When ball is in the system's court, player typically have to choose a move-action that allows the
        player to move to a location on his court that allows better chances to return the system's shot.

        Fire-step:
        - If the player cannot reach the ball after any simultaneous moving action, then player
        loses the point, step ends with a negative reward. Events: Player-miss

        - If player can reach the ball:
        a) Player hits it as required by the action
        b) Ball moves to the system end of the court based on player's action.
        c) System can move during the fire-step to better position returning player's shot

        Fire-step events: 1) Player-miss 2) Player-shot, System-move

        Position-step:
        System computes its action based on player's capability/hardness.
        - If system cannot reach the ball, player wins the point and step ends with positive reward.
        - If system can reach the ball:
          a) System hits it as required by the action
          b) Ball moves to the player end of the court based on system's action.
          c) Player can move during the position-step to better position returning system's shot

        Position-step events: 1) System-miss 2) System-shot, Player-move
        #"""

        if self._animation_tester:
            self._clean_render_events()
            self._animation_tester(self)
            return self._state, 1, False, False, []
        # return self._test_animate_event()

        # Assert that it is a valid action
        assert self.action_space.contains(action), "Invalid Action"

        if self.episode_done:
            self._game_step_count = 0  # Start a new episode
            self._set_score(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            self._score_history = []
            self.episode_done = False
            self._curr_episode = self._curr_episode + 1

        serving_actor = None
        if self._game_step_count == 0:  # step count is reset at end of each game.
            self._set_current_game_result(
                0
            )  # New game continues until result is known subsequently.
            serving_actor = TennisObservedState.get_next_server(self._state)
            self._set_server(serving_actor)

        fire_actor = serving_actor  # Actor to fire the shot in this step - player and sytem take turns to fire
        if fire_actor is None:
            curr_even_turn = (self._game_step_count % 2) == 0
            curr_server = self._get_server()
            if curr_server == self.PLAYER:
                fire_actor = self.PLAYER if curr_even_turn else self.SYSTEM
            else:
                fire_actor = self.SYSTEM if curr_even_turn else self.PLAYER

        self._game_step_count = self._game_step_count + 1

        # User must have already rendered the events from previous step by now so all past events must be cleared
        # before this step proceeds.
        self._clean_render_events()

        self.save_action(action)

        if fire_actor == self.PLAYER:
            return self._fire_player(action, serving_actor == self.PLAYER)
        else:
            return self._fire_system(action, serving_actor == self.SYSTEM)

    def _fire_player(self, player_action, serve):
        """
        Fire shot for player as required by player's action.

        :param player_action     Player's requested action
        :param serve             True if player is starting new set by serving first.
        """
        done = False
        truncated = False
        info = []

        if serve:
            self._set_ball_position(self.MIDDLE, self.BASE_LINE, self.PLAYER)
            self._set_player_position(self.MIDDLE, self.BASE_LINE)
            self._set_system_position(self.MIDDLE, self.BASE_LINE)
            player_action = 1  # Serve is always a fire - override user's selection.

        if self._action_logger.isEnabledFor(logging.INFO):
            action_name = self.get_action_name(player_action)
            self._action_logger.info(
                self._log_state()
                + "Action Selected: {}, {}, Firing: {}".format(
                    player_action, action_name, self.ACTOR_NAME_MAP[self.PLAYER]
                ),
                extra=self._LOG_TAG,
            )

        ball_pos = self._get_ball_position()
        player_pos = list(self._get_player_position())
        system_pos = self._get_system_position()
        action = self.ACTION_SPEC[player_action]

        if not serve and not self._behavior.is_shot_valid(
            self.SYSTEM, system_pos, ball_pos
        ):
            # Player's turn to fire but the system's shot is invalid giving the player a point.
            game_status = self._update_score_on_miss(self.SYSTEM)
            score = self._get_score()

            if self._render_view:
                # System hitting a bad shot looks same as player missing the shot.
                prev_shot_start = self._position_to_point(
                    self._to_render_position(
                        self.SYSTEM, self._get_ball_source_position()
                    )
                )
                shot_start = self._position_to_point(
                    self._to_render_position(self.PLAYER, ball_pos)
                )
                transit_start = RenderEvent.intersect_to(
                    prev_shot_start, shot_start, DiscreteTennis.RENDER_COURT_MIDLINE
                )

                self._append_render_event(
                    PlayerMiss(
                        self._player,
                        self._ball,
                        None,
                        transit_start,
                        shot_start,
                        None,
                        score,
                    )
                )

            done = self._check_complete_match(game_status)

            if self._logger.isEnabledFor(logging.INFO):
                self._logger.info(
                    self._log_state() + "Player Wins, shot invalid: {}".format(score),
                    extra=self._LOG_TAG,
                )
            return (
                self._state,
                self._behavior.compute_reward(self.ACTIVITY_SYSTEM_INVALID_SHOT),
                done,
                truncated,
                info,
            )

        # Move the player if this firing action also involves motion
        if action is not None and action.is_moving:
            if action.left:
                player_pos[0] = player_pos[0] - self._behavior.player_x
            if action.right:
                player_pos[0] = player_pos[0] + self._behavior.player_x
            if action.front:
                player_pos[1] = player_pos[1] + self._behavior.player_y
            if action.back:
                player_pos[1] = player_pos[1] - self._behavior.player_y

            # Player is not allowed to exceed the court bounds event if requested by the
            # player. Remove the action in direction of the offending move.
            player_pos[0] = max(
                0.0, min(player_pos[0], 1.0)
            )  # Clamp player position between 0.0 and 1.0
            player_pos[1] = max(
                0.0, min(player_pos[1], 1.0)
            )  # Clamp player position between 0.0 and 1.0

            self._set_player_position(*player_pos)
            if self._logger.isEnabledFor(logging.DEBUG):
                self._logger.debug(
                    self._log_state()
                    + "Player moved for firing to: {}".format(player_pos),
                    extra=self._LOG_TAG,
                )

        if (
            action is None
            or not action.is_firing
            or (
                not serve
                and not self._behavior.is_player_reachable(ball_pos, player_pos)
            )
        ):
            # Player cannot reach the ball as player either chose a non-firing action or ball is too far.
            game_status = self._update_score_on_miss(self.PLAYER)
            score = self._get_score()

            if self._render_view:
                prev_shot_start = self._position_to_point(
                    self._to_render_position(
                        self.SYSTEM, self._get_ball_source_position()
                    )
                )
                shot_start = self._position_to_point(
                    self._to_render_position(self.PLAYER, ball_pos)
                )
                transit_start = RenderEvent.intersect_to(
                    prev_shot_start, shot_start, DiscreteTennis.RENDER_COURT_MIDLINE
                )

                # This is a non-firing action and player was required to fire - so player loses this point.
                self._append_render_event(
                    PlayerMiss(
                        self._player,
                        self._ball,
                        None,
                        transit_start,
                        shot_start,
                        None,
                        score,
                    )
                )

            done = self._check_complete_match(game_status)
            if self._logger.isEnabledFor(logging.INFO):
                self._logger.info(
                    self._log_state() + "Player Miss: {}".format(score),
                    extra=self._LOG_TAG,
                )

            return (
                self._state,
                self._behavior.compute_reward(self.ACTIVITY_PLAYER_MISS),
                done,
                truncated,
                info,
            )

        # Player is hitting the shot.
        ball_end_pos = self._behavior.shot_target(self.PLAYER, player_pos, system_pos)
        if self._render_view:
            shot_start = self._position_to_point(
                self._to_render_position(self.PLAYER, ball_pos)
            )
            shot_end = self._position_to_point(
                self._to_render_position(self.SYSTEM, ball_end_pos)
            )
            player_at = self._position_to_point(
                self._to_render_position(self.PLAYER, player_pos)
            )
            transit_start = None
            if not serve:
                prev_shot_start = self._position_to_point(
                    self._to_render_position(
                        self.SYSTEM, self._get_ball_source_position()
                    )
                )
                transit_start = RenderEvent.intersect_to(
                    prev_shot_start, shot_start, DiscreteTennis.RENDER_COURT_MIDLINE
                )

            self._append_render_event(
                PlayerShot(
                    self._player,
                    self._ball,
                    player_at,
                    transit_start,
                    shot_start,
                    shot_end,
                )
            )

        self._set_ball_source_position(*ball_pos)
        self._set_ball_position(*ball_end_pos)
        if self._logger.isEnabledFor(logging.INFO):
            self._logger.info(
                self._log_state()
                + "Player Shot from {} to {}".format(ball_pos, ball_end_pos),
                extra=self._LOG_TAG,
            )

        # If system indicates that it is also moving, then allow that movement.
        self._behavior.next_system_action(False, False, self)
        return (
            self._state,
            self._behavior.compute_reward(self.ACTIVITY_PLAYER_SHOT),
            False,
            False,
            info,
        )

    def _fire_system(self, player_action, serve):
        """
        Fire shot for system and also perform any action required by player.

        :param player_action     Player's requested action
        :param serve             True if system is starting new set by serving first.
        """
        done = False
        truncated = False
        info = []

        if serve:
            self._set_ball_position(self.MIDDLE, self.BASE_LINE, self.SYSTEM)
            self._set_system_position(self.MIDDLE, self.BASE_LINE)
            self._set_player_position(self.MIDDLE, self.BASE_LINE)

        if self._action_logger.isEnabledFor(logging.INFO):
            action_name = self.get_action_name(player_action)
            self._action_logger.info(
                self._log_state()
                + "Action Selected: {}, {}, Firing: {}".format(
                    player_action, action_name, self.ACTOR_NAME_MAP[self.SYSTEM]
                ),
                extra=self._LOG_TAG,
            )

        ball_pos = self._get_ball_position()
        player_pos = list(self._get_player_position())
        system_pos = list(self._get_system_position())
        action = self.ACTION_SPEC[player_action]

        if not self._behavior.is_shot_valid(self.PLAYER, player_pos, ball_pos):
            # System's turn to fire but the player's shot is invalid giving the system a point.
            game_status = self._update_score_on_miss(self.PLAYER)
            score = self._get_score()

            if self._render_view:
                # Player hitting a bad shot looks same as system missing the shot.
                prev_shot_start = self._position_to_point(
                    self._to_render_position(
                        self.PLAYER, self._get_ball_source_position()
                    )
                )
                shot_start = self._position_to_point(
                    self._to_render_position(self.SYSTEM, ball_pos)
                )
                transit_start = RenderEvent.intersect_to(
                    prev_shot_start, shot_start, DiscreteTennis.RENDER_COURT_MIDLINE
                )

                self._append_render_event(
                    SystemMiss(
                        self._system,
                        self._ball,
                        None,
                        transit_start,
                        shot_start,
                        None,
                        score,
                    )
                )

            done = self._check_complete_match(game_status)
            if self._logger.isEnabledFor(logging.INFO):
                self._logger.info(
                    self._log_state() + "Player Shot Invalid", extra=self._LOG_TAG
                )
            return (
                self._state,
                self._behavior.compute_reward(self.ACTIVITY_PLAYER_INVALID_SHOT),
                done,
                truncated,
                info,
            )

        if not serve and not self._behavior.is_system_reachable(ball_pos, system_pos):
            # System cannot reach the ball as ball is too far from system
            game_status = self._update_score_on_miss(self.SYSTEM)
            score = self._get_score()

            if self._render_view:
                prev_shot_start = self._position_to_point(
                    self._to_render_position(
                        self.PLAYER, self._get_ball_source_position()
                    )
                )
                shot_start = self._position_to_point(
                    self._to_render_position(self.SYSTEM, ball_pos)
                )
                transit_start = RenderEvent.intersect_to(
                    prev_shot_start, shot_start, DiscreteTennis.RENDER_COURT_MIDLINE
                )

                self._append_render_event(
                    SystemMiss(
                        self._system,
                        self._ball,
                        None,
                        transit_start,
                        shot_start,
                        None,
                        score,
                    )
                )

            done = self._check_complete_match(game_status)
            if self._logger.isEnabledFor(logging.INFO):
                self._logger.info(
                    self._log_state() + "System Miss: {}".format(score),
                    extra=self._LOG_TAG,
                )

            return (
                self._state,
                self._behavior.compute_reward(self.ACTIVITY_SYSTEM_MISS),
                done,
                truncated,
                info,
            )

        self._behavior.next_system_action(True, serve, self)

        # Move the player if user has chosen a moving action. Note that a firing action is not allowed for the user
        # but if the firing action is accompanied with motion, that motion will be considered valid and user won't
        # be penalized for selecting that action. Thus Fire-Right will be considered Move-Right as player cannot fire
        # while system is firing.
        if action is not None and action.is_moving:
            if action.left:
                player_pos[0] = player_pos[0] - self._behavior.player_x
            if action.right:
                player_pos[0] = player_pos[0] + self._behavior.player_x
            if action.front:
                player_pos[1] = player_pos[1] + self._behavior.player_y
            if action.back:
                player_pos[1] = player_pos[1] - self._behavior.player_y
            if self._render_view:
                self._append_render_event(
                    PlayerMove(
                        self._player,
                        None,
                        self._position_to_point(
                            self._to_render_position(self.PLAYER, player_pos)
                        ),
                    )
                )
            self._set_player_position(*player_pos)
            if self._logger.isEnabledFor(logging.DEBUG):
                self._logger.debug(
                    self._log_state() + "Player Moved to {}".format(player_pos),
                    extra=self._LOG_TAG,
                )
        return (
            self._state,
            self._behavior.compute_reward(self.ACTIVITY_SYSTEM_SHOT),
            False,
            False,
            info,
        )

    def _check_complete_match(self, game_status):
        """
        Check if the match is complete.

        :return True if all sets are completed and match is now over.
        """
        # Reset the positions of each entity in view on start of a new game.
        # The positions of those entities will be reset at subsequent fire but if the
        # view does not show them at serve-line, it is confusing in the view
        # and therefore, this explicit resetting of positions is required.
        if self._render_view:
            serve_pos = (self.MIDDLE, self.BASE_LINE)
            self._append_render_event(
                PlayerMove(
                    self._player,
                    None,
                    self._position_to_point(
                        self._to_render_position(self.PLAYER, serve_pos)
                    ),
                )
            )
            self._append_render_event(
                SystemMove(
                    self._system,
                    None,
                    self._position_to_point(
                        self._to_render_position(self.SYSTEM, serve_pos)
                    ),
                )
            )

        done = game_status == self.GAMESTATUS_COMPLETE
        if done:
            self.episode_done = True
        return done

    @staticmethod
    def _position_to_point(position):
        return Point(position[0], position[1])

    def _log_state(self):
        ret = "{}{}{}".format(self._LOG_LINE, self._print_state(), self._LOG_LINE)
        return ret

    def _print_state(self):
        ret = "[{}] Player: ({:.3f}, {:.3f}), System: ({:.3f}, {:.3f}), Ball: ({:.3f}, {:.3f}, {}), Serve: {}, Score: ({:.1f}, {:.1f}, {:.1f}. {:.1f})".format(
            self._game_step_count,
            self._state[self.STATE_PLAYER_X],
            self._state[self.STATE_PLAYER_Y],
            self._state[self.STATE_SYSTEM_X],
            self._state[self.STATE_SYSTEM_Y],
            self._state[self.STATE_BALL_X],
            self._state[self.STATE_BALL_Y],
            "P" if self._state[self.STATE_BALL_OWNER] == self.PLAYER else "S",
            "P" if self._state[self.STATE_SERVE] == self.PLAYER else "S",
            self._state[self.STATE_GAMESCORE_PLAYER],
            self._state[self.STATE_GAMESCORE_SYSTEM],
            self._state[self.STATE_SETSCORE_PLAYER],
            self._state[self.STATE_SETSCORE_SYSTEM],
        )
        return ret

    def tostr_score(self, include_history=True):
        score = self._get_score()
        # #Sets, Curr-Set, Curr-game
        player_set_score = "Player {:>3s} {:>3s} {:>3s} ||| ".format(
            DiscreteTennis.RENDER_MATCHSCORE[
                TennisObservedState.to_score_num(score[4])
            ],
            DiscreteTennis.RENDER_SETSCORE[TennisObservedState.to_score_num(score[2])],
            DiscreteTennis.RENDER_GAMESCORE[TennisObservedState.to_score_num(score[0])],
        )
        system_set_score = "System {:>3s} {:>3s} {:>3s} ||| ".format(
            DiscreteTennis.RENDER_MATCHSCORE[
                TennisObservedState.to_score_num(score[5])
            ],
            DiscreteTennis.RENDER_SETSCORE[TennisObservedState.to_score_num(score[3])],
            DiscreteTennis.RENDER_GAMESCORE[TennisObservedState.to_score_num(score[1])],
        )
        if include_history:
            for item in self._score_history:
                if item[2] == 1:
                    player_set_score = player_set_score + " |{:^3s}||".format(
                        DiscreteTennis.RENDER_SETSCORE[item[0]]
                    )
                    system_set_score = system_set_score + " |{:^3s}||".format(
                        DiscreteTennis.RENDER_SETSCORE[item[1]]
                    )
                else:
                    player_set_score = player_set_score + "{:>4s}".format(
                        DiscreteTennis.RENDER_GAMESCORE[item[0]]
                    )
                    system_set_score = system_set_score + "{:>4s}".format(
                        DiscreteTennis.RENDER_GAMESCORE[item[1]]
                    )

        result = "\n\t{}\n\t{}".format(player_set_score, system_set_score)
        return result

    def _update_score_on_miss(self, loser):
        """
        Update scores upon an entity missing shot

        :param loser Actor that lost the point PLAYER/SYSTEM
        :return GAMESTATUS_CONTINUE if current set continues, GAMESTATUS_NEWSET if new set must start, GAMESTATUS_COMPLETE if all sets completed
        """
        player_game, system_game, player_set, system_set, player_match, system_match = (
            self._get_score_num()
        )

        # Assume player missed the shot
        winner = self.SYSTEM
        loser_game = player_game
        winner_game = system_game
        loser_set = player_set
        winner_set = system_set
        loser_match = player_match
        winner_match = system_match

        if loser == self.SYSTEM:  # System missed the shot
            winner = self.PLAYER
            loser_game = system_game
            winner_game = player_game
            loser_set = system_set
            winner_set = player_set
            loser_match = system_match
            winner_match = player_match

        ret = self.GAMESTATUS_CONTINUE

        # Current game has completed with someone gaining a point.
        self._set_current_game_result(1 if loser == DiscreteTennis.SYSTEM else -1)
        curr_game_step_count = self._game_step_count
        self._game_step_count = 0
        self._behavior.on_end_game()

        event_type = 0  # None
        event_winner_game = winner_game
        event_loser_game = loser_game
        event_winner_set = winner_set
        event_loser_set = loser_set

        # Loser has advantage, remove the advantage.
        if loser_game == self.GAMESCORE_AD:
            loser_game -= self.GAMESCORE_DELTA
        else:
            if (
                winner_game == self.GAMESCORE_40 and loser_game < self.GAMESCORE_40
            ) or winner_game == self.GAMESCORE_AD:
                event_type = 1  # Game complete
                # Winner won this game
                self._score_history.append(
                    (winner_game, loser_game, 0)
                    if winner == self.PLAYER
                    else (loser_game, winner_game, 0)
                )
                winner_set = (
                    winner_set + 1
                )  # One more game in current set is won by this game's winner.
                event_winner_set = winner_set

                # Reset game scores to start new game.
                winner_game = 0
                loser_game = 0

                # Advantage set:
                # Currently we only implement advantage set  wherein there is no tie-breaker game at 6-6.
                # Game continues until someone wins by two games.
                # Deviation: First player to reach winning 9 games in a set is declared winner for the set.
                if (
                    winner_set >= 6 and (winner_set - loser_set) >= 2
                ) or winner_set >= 9:
                    self._score_history.append(
                        (winner_set, loser_set, 1)
                        if winner == self.PLAYER
                        else (loser_set, winner_set, 1)
                    )
                    event_type = 2  # Set complete
                    winner_match = (
                        winner_match + 1
                    )  # One more set is won by this game's winner.

                    # Reset games in set to start new set.
                    winner_set = 0
                    loser_set = 0

                    # Best of three sets: First player to win two sets wins the game -
                    if winner_match >= 2:
                        event_type = 3  # Match complete
                        ret = self.GAMESTATUS_COMPLETE
                    else:
                        ret = self.GAMESTATUS_NEWSET
                else:
                    ret = self.GAMESTATUS_NEWGAME
            else:
                winner_game += self.GAMESCORE_DELTA

        if winner == self.SYSTEM:
            self._set_score_num(
                loser_game,
                winner_game,
                loser_set,
                winner_set,
                loser_match,
                winner_match,
            )
            if event_type > 0 and self._stats_logger is not None:
                self._stats_logger.game_result(
                    self._curr_episode + 1,
                    event_type,
                    winner,
                    event_loser_game,
                    event_winner_game,
                    event_loser_set,
                    event_winner_set,
                    loser_match,
                    winner_match,
                    curr_game_step_count,
                )
        else:
            self._set_score_num(
                winner_game,
                loser_game,
                winner_set,
                loser_set,
                winner_match,
                loser_match,
            )
            if event_type > 0 and self._stats_logger is not None:
                self._stats_logger.game_result(
                    self._curr_episode + 1,
                    event_type,
                    winner,
                    event_winner_game,
                    event_loser_game,
                    event_winner_set,
                    event_loser_set,
                    winner_match,
                    loser_match,
                    curr_game_step_count,
                )

        return ret


class Point(object):
    def __init__(self, x=0, y=0, name=None):
        """
        Point constructor.

        :param x X-coordinate
        :param y Y-coordinate
        :param name Optional name representing the purpose of this point.
        """
        self.x = x
        self.y = y
        self.name = name
        self.cache_icon = None

    def set_position(self, x, y):
        self.x = x
        self.y = y

    def get_position(self):
        return (self.x, self.y)

    def move(self, del_x, del_y):
        self.x += del_x
        self.y += del_y

    def is_equal(self, other):
        return self.x == other.x and self.y == other.y

    def prep_to_merge(self, icon, w, h):
        """
        Set the icon image data and prepare it so that it can be merged with background using the
        show_at during game.

        :param icon Element's image data as read with imread and scaled by 255.
        :param w    Expected width in pixels of this element on canvas.
        :param h    Expected height in pixels of this element on canvas.
        """
        self.icon = icon  # Icon was Read with imread of shape (h, w, #channel)
        self.icon = cv2.resize(
            src=self.icon, dsize=(w, h)
        )  # resize needs size as (w, h)
        self.icon_w = w
        self.icon_h = h

        self.alpha = self.icon[:, :, 3]
        self.alpha = self.alpha[:, :, np.newaxis]
        self.one_minus_alpha = 1.0 - self.alpha
        self.icon_fg = self.icon * self.alpha

    def show_at(self, canvas):
        """
        Merge this element with the background by overlaying it with alpha bending.

        :param canvas    Canvas over which this element must be overlaid.
        """
        # x, y attributes are center of this element - get top-left corner and render whole width, height on canvas
        # x = self.x - int(self.icon_w/2);
        # y = self.y - int(self.icon_h/2);
        # crop_bg = canvas[y:y + self.icon_h, x : x + self.icon_w]
        # crop_bg = crop_bg * self.one_minus_alpha + self.icon_fg
        # canvas[y:y + self.icon_h, x : x + self.icon_w] = crop_bg

        x = self.x - int(self.icon_w / 2)
        y = self.y - int(self.icon_h / 2)

        if (
            x < 0
            or (x + self.icon_w) >= canvas.shape[1]
            or y < 0
            or (y + self.icon_h) >= canvas.shape[0]
        ):
            return  # Can't show out of bounds.

        self.prev_bg = canvas[y : y + self.icon_h, x : x + self.icon_w]
        crop_bg = self.prev_bg * self.one_minus_alpha + self.icon_fg
        canvas[y : y + self.icon_h, x : x + self.icon_w] = crop_bg

    def move_to(
        self, canvas, target_x, target_y, prev_bg=None, ret_prev=True, pristine=None
    ):
        """
        Merge this element with the background by overlaying it with alpha bending.

        :param canvas    Canvas over which this element must be overlaid.
        :param x         X offset in canvas (in pixels) where the center of this element must reside.
        :param y         Y offset in canvas (in pixels) where the center of this element must reside.
        :param prev_bg   Previously saved background for current location of this item that can be
           used to replace the background left open by this move.
        :param ret_prev  If true, return a copy of the background that existed at the target location
           before this item is moved there.
        :param pristine  Pristine canvas without this element.
        """
        # x = self.x - int(self.icon_w/2);
        # y = self.y - int(self.icon_h/2);
        # pristine_bg = pristine[y:y + self.icon_h, x : x + self.icon_w]
        # canvas[y:y + self.icon_h, x : x + self.icon_w] = pristine_bg    # Wipe element from canvas.
        #
        # # Overlay element onto canvas
        # x = target_x - int(self.icon_w/2);
        # y = target_y - int(self.icon_h/2);
        # prev_bg = canvas[y:y + self.icon_h, x : x + self.icon_w]
        # crop_bg = prev_bg * self.one_minus_alpha + self.icon_fg
        # canvas[y:y + self.icon_h, x : x + self.icon_w] = crop_bg
        # self.x = target_x
        # self.y = target_y
        if prev_bg is not None:
            # canvas[0:0 + self.icon_h, 100 : 100 + self.icon_w] = prev_bg
            x = self.x - int(self.icon_w / 2)
            y = self.y - int(self.icon_h / 2)
            canvas[y : y + self.icon_h, x : x + self.icon_w] = (
                prev_bg  # Wipe element from canvas.
            )
        elif pristine is not None:
            self.erase(
                canvas, pristine
            )  # Wipe element from canvas using pristine copy.

        # Allocate memory enough to save a copy of this item's size.
        if ret_prev and prev_bg is None:
            if self.cache_icon is None:
                self.cache_icon = np.zeros(
                    (self.icon_h, self.icon_w, self.icon.shape[2])
                )
            prev_bg = self.cache_icon

        # Overlay element onto canvas
        x = target_x - int(self.icon_w / 2)
        y = target_y - int(self.icon_h / 2)

        if (
            x < 0
            or (x + self.icon_w) >= canvas.shape[1]
            or y < 0
            or (y + self.icon_h) >= canvas.shape[0]
        ):
            return  # Move pushes out of bounds.

        # Sliced arrays are always views of original array.
        crop_bg = canvas[y : y + self.icon_h, x : x + self.icon_w]
        if ret_prev:
            # Copy the background as it will be overwritten here.
            prev_bg[0 : 0 + self.icon_h, 0 : 0 + self.icon_w] = crop_bg
        crop_bg = crop_bg * self.one_minus_alpha + self.icon_fg
        canvas[y : y + self.icon_h, x : x + self.icon_w] = crop_bg

        self.x = target_x
        self.y = target_y

        # canvas[0:0 + self.icon_h, 200 + (self.icon_w*2) : 200 + (self.icon_w*2) + self.icon_w] = save_bg
        return prev_bg

    def erase(self, canvas, pristine):
        x = self.x - int(self.icon_w / 2)
        y = self.y - int(self.icon_h / 2)
        pristine_bg = pristine[y : y + self.icon_h, x : x + self.icon_w]
        canvas[y : y + self.icon_h, x : x + self.icon_w] = (
            pristine_bg  # Wipe element from canvas.
        )

    def redraw(self, canvas, pristine):
        self.erase(canvas, pristine)
        self.show_at(canvas)

    def __repr__(self):
        return "({}, {})".format(self.x, self.y)


class Ball(Point):
    def __init__(self, name):
        super().__init__(name=name)
        icon = (
            cv2.imread(
                os.path.join(os.path.dirname(__file__), "ball.png"),
                cv2.IMREAD_UNCHANGED,
            )
            / 255.0
        )
        self.prep_to_merge(icon, 24, 24)


class Player(Point):
    def __init__(self, name):
        super().__init__(name=name)
        icon = (
            cv2.imread(
                os.path.join(os.path.dirname(__file__), "player.png"),
                cv2.IMREAD_UNCHANGED,
            )
            / 255.0
        )
        self.prep_to_merge(icon, 36, 36)
        self.actor_type = DiscreteTennis.PLAYER


class System(Point):
    def __init__(self, name):
        super().__init__(name=name)
        icon = (
            cv2.imread(
                os.path.join(os.path.dirname(__file__), "system.png"),
                cv2.IMREAD_UNCHANGED,
            )
            / 255.0
        )
        self.prep_to_merge(icon, 36, 36)
        self.actor_type = DiscreteTennis.SYSTEM


class Court(Point):
    def __init__(self, name):
        super().__init__(name=name)
        # Reads image of shape (h, w, #channel)
        icon = (
            cv2.imread(
                os.path.join(os.path.dirname(__file__), "court.png"),
                cv2.IMREAD_UNCHANGED,
            )
            / 255.0
        )
        self.prep_to_merge(icon, DiscreteTennis.COURT[0], DiscreteTennis.COURT[1])


# Animation events
# 1) Player-shot: Player hitting a shot/serve:
#    a) Ball approaches from net to the player's contact position. (Prev play's system's target position/
#       player serving)truncated
#    b) Player at contact position: i) Player already present ii) Player travels from prev to contact position.
#    c) Player hits ball in a specific direction destined for specific position in system court - ball
#       approaches from player to the net in the target direction.
# 2) System-shot: System hitting a shot/serve:
#    a) Ball approached from net to the system's contact position. (Prev play's player's target position/
#       system serving)
#    b) System at contact position i) System already present ii) System travels from prev to contact position.
#    c) System hits ball in a specific direction destined for specific position in player's court - ball
#       approaches from system to the net in the target direction.
# 3) Player-miss: Player missing a shot:
#    a) Ball approaches from net to the target position in player's court.
#    b) Player moves as directed by action but cannot reach target position.
#    c) Ball overshoots the court and hits margin.
#    d) System score increments. It may result in end of game, current set or all sets.
# 4) System-miss: System missing a shot:
#    a) Ball approaches from net to the target position in system's court.
#    b) System moves as directed by action but cannot reach target position.
#    c) Ball overshoots the court and hits margin.
#    d) Player score increments. It may result in end of game, current set or all sets.


# All points used in render-event specification are canvas relative absolute coordinates.
class RenderEvent:
    SHOT_FRAME = 4  # Number of frames to show for shots.
    FRAME_INTERVAL = 100  # milliseconds for a single frame.
    IMAGE_WINDOW = "Game"

    def __init__(
        self,
        name,
        actor,
        ball=None,
        actor_at=None,
        transit_start=None,
        shot_start=None,
        shot_end=None,
        actor_end=None,
    ):
        """
        Render Event

        :param name    Name of this event
        :param actor   Renderable  object of type Player/System that is a renderable element of the environment.
        :param actor_at Point representing current location of the actor. If none, it will be same as current position of actor.
        :param ball    Renderable object of type Ball that is a renderable element of the environment
        :param transit_start Point where the shot begins at the court mid-line coming towards the firing actor.
        :param shot_start Point where the firing actor hits the shot.
        :param shot_end Point where the firing actor targets the shot in peer's court.
        :param actor_end Point where the moving actor ends the move.
        """
        self._actor = actor
        self._shot_at = actor_at if actor_at is not None else actor
        self._actor_end = actor_end
        self.ball = ball
        self.name = name
        self._transit_start = transit_start
        self._shot_start = shot_start
        if shot_start is not None and shot_end is not None:
            self._transit_end = RenderEvent.intersect_to(
                shot_start,
                shot_end,
                (
                    DiscreteTennis.RENDER_PLAYER_Y[2]
                    if actor == DiscreteTennis.PLAYER
                    else DiscreteTennis.RENDER_SYSTEM_Y[2]
                ),
            )

        self._shot_end = shot_end
        self._move_peer_actor_event = None

    def set_move_peer_actor_event(self, event):
        """
        Set a render event for moving a peer actor while this event is being rendered.

        :param event A PlayerMove or SystemMove event that must be rendered simultaneously.
        """
        self._move_peer_actor_event = event

    def to_path(self, start, end, count, add_end=False, add_start=False):
        """
        Get count intermediate points on path from start point to the end point.

        :return array of count points if possible else None
        """
        if start is None or end is None:
            return None

        delta_x = end.x - start.x
        parts = count + 1
        ret_arr = []
        if add_start:
            ret_arr.append(start)

        if abs(delta_x) <= 0.001:
            # Simply divide along the y-axis for infinite slope
            delta_y = end.y - start.y
            delta_y = delta_y / parts
            prev_y = start.y
            for _ in range(count):
                part_y = int(prev_y + delta_y)
                prev_y = part_y
                ret_arr.append(Point(start.x, part_y))
        else:
            slope = (end.y - start.y) / delta_x
            delta_x = delta_x / parts
            prev_x = start.x
            prev_y = start.y
            for _ in range(count):
                part_x = int(prev_x + delta_x)
                part_y = int(prev_y + (slope * delta_x))
                ret_arr.append(Point(part_x, part_y))
                prev_x = part_x
                prev_y = part_y

        if add_end:
            ret_arr.append(end)
        return ret_arr

    @staticmethod
    def render_frame(canvas):
        cv2.imshow(RenderEvent.IMAGE_WINDOW, canvas)
        cv2.waitKey(RenderEvent.FRAME_INTERVAL)

    def _begin_actor(self, actor, start_at, canvas, pristine):
        if actor.x != start_at.x or actor.y != start_at.y:
            actor.erase(canvas, pristine)
            actor.set_position(start_at.x, start_at.y)
            actor.show_at(canvas)
        else:
            actor.redraw(canvas, pristine)

    def render_shot(self, canvas, pristine):
        """
        Renders an animated ball-shot moving from the transit_start to the shot_start and then back to transit_end.
        If current actor is not already at shot_start, that actor will be moved to the shot_start during the
        first leg. If a move peer actor event is specified, that peer actor is also moved during secong leg.
        """
        # Actor must move from actor_at to _shot_start
        # while ball moves from _transit_start to _shot_start

        actor_points = None
        self.ball.erase(canvas, pristine)
        self._begin_actor(self._actor, self._shot_at, canvas, pristine)
        # First actor moves to the start point (if not already there)

        prev_ball_bg = None  # Moved here to retain background between two trajectories.
        if (
            self._transit_start is not None
        ):  # During a serve, transit_start will be None
            if not self._actor.is_equal(self._shot_start):
                actor_points = self.to_path(
                    self._actor, self._shot_start, self.SHOT_FRAME - 1, True
                )
            ball_points = self.to_path(
                self._transit_start, self._shot_start, self.SHOT_FRAME - 1, True
            )

            # First show the leg from net to the actor.
            prev_actor_bg = None
            for frame in range(self.SHOT_FRAME):
                curr_point = ball_points[frame]
                prev_ball_bg = self.ball.move_to(
                    canvas, curr_point.x, curr_point.y, prev_ball_bg, True, pristine
                )

                if actor_points is not None:
                    curr_point = actor_points[frame]
                    prev_actor_bg = self._actor.move_to(
                        canvas,
                        curr_point.x,
                        curr_point.y,
                        prev_actor_bg,
                        True,
                        pristine,
                    )
                self.render_frame(canvas)

        # If another simultaneous peer actor move is also specified, peer actor must be moved while this
        # actor's shot moves towards the net.
        peer_actor = None
        peer_actor_points = None
        if self._move_peer_actor_event is not None:
            peer_actor = self._move_peer_actor_event._actor
            peer_actor_points = self.to_path(
                self._move_peer_actor_event._shot_at,
                self._move_peer_actor_event._actor_end,
                self.SHOT_FRAME - 1,
                True,
            )

        # If actor is not at the shot-start, after hitting the shot actor moves back to actor_at.
        actor_points = None
        if not self._shot_at.is_equal(self._shot_start):
            actor_points = self.to_path(
                self._shot_start, self._shot_at, self.SHOT_FRAME - 1, True
            )

        # Now show the leg from actor to net.
        ball_points = self.to_path(
            self._shot_start, self._transit_end, self.SHOT_FRAME - 1, True
        )
        prev_actor_bg = None
        prev_peer_actor_bg = None
        for frame in range(self.SHOT_FRAME):
            curr_point = ball_points[frame]
            prev_ball_bg = self.ball.move_to(
                canvas, curr_point.x, curr_point.y, prev_ball_bg, True, pristine
            )

            if peer_actor_points is not None:  # Move peer actor, if one is specified.
                curr_point = peer_actor_points[frame]
                prev_peer_actor_bg = peer_actor.move_to(
                    canvas,
                    curr_point.x,
                    curr_point.y,
                    prev_peer_actor_bg,
                    True,
                    pristine,
                )

            if (
                actor_points is not None
            ):  # Move actor back to its position before the shot.
                curr_point = actor_points[frame]
                prev_actor_bg = self._actor.move_to(
                    canvas, curr_point.x, curr_point.y, prev_actor_bg, True, pristine
                )

            self.render_frame(canvas)

        # Ball leaving actor wipes out part of player - rerender the player.
        # self._begin_actor(self._actor, self._shot_at, canvas, pristine)
        # self.render_frame(canvas)

    def render_miss(self, canvas, pristine):
        """
        Renders an animated ball missed-shot moving from the transit_start to the shot_start and then out-of-court.
        If current actor is not already at shot_start, that actor will be moved to actor_end, if specified during
        first leg. If a move peer actor event is specified, that peer actor is also moved during second leg.
        """

        # Actor must move from actor_at to _shot_start
        # while ball moves from _transit_start to _shot_start
        actor_points = None
        self.ball.erase(canvas, pristine)
        self._begin_actor(self._actor, self._shot_at, canvas, pristine)

        # Debug
        self.render_frame(canvas)

        if self._actor_end is not None and not self._actor.is_equal(self._actor_end):
            actor_points = self.to_path(
                self._actor, self._actor_end, self.SHOT_FRAME - 1, True
            )
        ball_points = self.to_path(
            self._transit_start, self._shot_start, self.SHOT_FRAME - 1, True
        )

        # First show the leg from net to the actor.
        prev_actor_bg = None
        prev_ball_bg = None
        for frame in range(self.SHOT_FRAME):
            if actor_points is not None:
                curr_point = actor_points[frame]
                prev_actor_bg = self._actor.move_to(
                    canvas, curr_point.x, curr_point.y, prev_actor_bg, True, pristine
                )
            curr_point = ball_points[frame]
            prev_ball_bg = self.ball.move_to(
                canvas, curr_point.x, curr_point.y, prev_ball_bg, True, pristine
            )
            self.render_frame(canvas)

        # If another simultaneous peer actor move is also specified, peer actor must be moved while this
        # actor's shot moves towards the net.
        actor = None
        actor_points = None
        if self._move_peer_actor_event is not None:
            actor = self._move_peer_actor_event._actor
            actor_points = self.to_path(
                self._move_peer_actor_event._shot_at,
                self._move_peer_actor_event._actor_end,
                self.SHOT_FRAME - 1,
                True,
            )

        # Find the point where ball hits the edge of view.
        end_point = self._trace_to_edge(
            self._actor.actor_type == DiscreteTennis.PLAYER,
            self._transit_start,
            self._shot_start,
            DiscreteTennis.CANVAS_LIMIT[0],
            DiscreteTennis.CANVAS_LIMIT[1],
        )
        end_point = Point(*end_point)

        # Now show the leg from missed shot to the out-of-court.
        ball_points = self.to_path(
            self._shot_start, end_point, self.SHOT_FRAME - 1, True
        )
        for frame in range(self.SHOT_FRAME):
            if actor_points is not None:  # Move peer actor, if one is specified.
                curr_point = actor_points[frame]
                prev_actor_bg = actor.move_to(
                    canvas, curr_point.x, curr_point.y, prev_actor_bg, True, pristine
                )

            curr_point = ball_points[frame]
            prev_ball_bg = self.ball.move_to(
                canvas, curr_point.x, curr_point.y, prev_ball_bg, True, pristine
            )
            self.render_frame(canvas)

    def render_actor_move(self, canvas, pristine):
        """
        Renders an animated actor movement from the actor's current position to the actor-end.
        """
        # Actor must move from actor_at to _actor_end
        if self._actor_end is None:
            return  # Cannot move if we don't know the destination.

        self._begin_actor(self._actor, self._shot_at, canvas, pristine)
        actor_points = self.to_path(
            self._actor, self._actor_end, self.SHOT_FRAME - 1, True
        )
        # First show the leg from net to the actor.
        prev_actor_bg = None
        for frame in range(self.SHOT_FRAME):
            curr_point = actor_points[frame]
            prev_actor_bg = self._actor.move_to(
                canvas, curr_point.x, curr_point.y, prev_actor_bg, True, pristine
            )
            self.render_frame(canvas)

    @staticmethod
    def render_score(canvas, score):
        """
        Renders current score.
        """
        # Put the score on canvas
        pos = DiscreteTennis.RENDER_SCORE_POSITION
        pos_x = pos[0]
        pos_y = pos[1]
        canvas[pos_y - 20 : pos_y + 30, pos_x : pos_x + 250] = 1

        # self.render_frame(canvas)

        cv2.putText(canvas, "Player ", pos, font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        pos_x = pos_x + 100

        score_val = DiscreteTennis.RENDER_MATCHSCORE[
            TennisObservedState.to_score_num(score[4])
        ]
        cv2.putText(
            canvas, score_val, (pos_x, pos_y), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA
        )
        score_val = DiscreteTennis.RENDER_SETSCORE[
            TennisObservedState.to_score_num(score[2])
        ]
        cv2.putText(
            canvas,
            score_val,
            (pos_x + 40, pos_y),
            font,
            0.8,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )
        score_val = DiscreteTennis.RENDER_GAMESCORE[
            TennisObservedState.to_score_num(score[0])
        ]
        cv2.putText(
            canvas,
            score_val,
            (pos_x + 80, pos_y),
            font,
            0.8,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        pos_x = pos[0]
        pos_y = pos[1] + 18
        cv2.putText(
            canvas, "System ", (pos_x, pos_y), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA
        )
        pos_x = pos_x + 100
        score_val = DiscreteTennis.RENDER_SETSCORE[
            TennisObservedState.to_score_num(score[5])
        ]
        cv2.putText(
            canvas, score_val, (pos_x, pos_y), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA
        )
        score_val = DiscreteTennis.RENDER_SETSCORE[
            TennisObservedState.to_score_num(score[3])
        ]
        cv2.putText(
            canvas,
            score_val,
            (pos_x + 40, pos_y),
            font,
            0.8,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )
        score_val = DiscreteTennis.RENDER_GAMESCORE[
            TennisObservedState.to_score_num(score[1])
        ]
        cv2.putText(
            canvas,
            score_val,
            (pos_x + 80, pos_y),
            font,
            0.8,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def intersect_to(start, end, target_y):
        """
        Find the point of intersection of line from start to end meeting a horizontal line at target_y.

        :param start Start point
        :param end   End point
        :param target_y Y value at which intersection point is needed.
        :return Point of intersection if present and within bounds, else None
        """
        delta_x = end.x - start.x
        if abs(delta_x) < 0.001:
            return Point(start.x, target_y)

        slope = (end.y - start.y) / delta_x
        if abs(slope) < 0.001:
            return None  # Does not intersect

        delta_y = target_y - start.y
        delta_x = delta_y / slope

        pt_x = start.x + delta_x
        pt_y = target_y
        if pt_x < 0 or pt_x > DiscreteTennis.CANVAS_LIMIT[0]:
            return None
        return Point(int(pt_x), int(pt_y))

    def _trace_to_edge(self, top_edge, start, end, limit_x, limit_y):
        """
        Trace the trajectory of the ball from specified start point to end point to hit the edge of
        the rectangle bounder by 0 and specified limits.

        :param top_edge True if the path is to traced upwards towards 0 y.
        :param start Start point of the ball trajectory
        :param end End point of the ball trajectory
        :param limit_x Maximum x possible for the ball to travel.
        :param limit_y Maximum y possible for the ball to travel.
        """
        delta_x = end.x - start.x
        if abs(delta_x) < 0.001:
            return (start.x, 0 if top_edge else limit_y)

        slope = (end.y - start.y) / delta_x
        if abs(slope) < 0.001:
            return None

        # If tracing is needed to proceed towards top edge of court, then
        # maximum delta y between the end-point and start point is the value of start point's y value
        # otherwise it is difference between canvas's maximum limit and start point.
        max_delta_y = start.y if top_edge else limit_y - start.y
        max_delta_x = max_delta_y / slope
        ret_x = (start.x - max_delta_x) if top_edge else (start.x + max_delta_x)
        ret_y = 0 if top_edge else limit_y

        if ret_x < 0:
            excess_x = 0 - ret_x
            excess_y = slope * excess_x
            ret_y += excess_y
        elif ret_x > limit_x:
            excess_x = ret_x - limit_x
            excess_y = slope * excess_x
            ret_y -= excess_y
        return (int(ret_x), int(ret_y))


class PlayerShot(RenderEvent):
    def __init__(self, player, ball, player_at, transit_start, shot_start, shot_end):
        """
        :param player_at Starting position of player to hit this shot.
        :param transit_start Position of ball at net from previous shot traveling towards the player.
            Same as shot_start for a serve shot.
        :param shot_start Position of ball where shot must begin.
        :param transit_end Position of ball at net for current shot traveling towards the system.
        :param shot_end Position in system court where the ball must end after completion of this shot.
        """
        super().__init__(
            "playerShot", player, ball, player_at, transit_start, shot_start, shot_end
        )

    def render_event(self, canvas, pristine):
        self.render_shot(canvas, pristine)


class SystemShot(RenderEvent):
    def __init__(self, system, ball, system_at, transit_start, shot_start, shot_end):
        """
        :param system_at Starting position of system to hit this shot.
        :param transit_start Position of ball at net from previous shot traveling towards the system.
            Same as shot_start for a serve shot.
        :param shot_start Position of ball where shot must begin.
        :param transit_end Position of ball at net for current shot traveling towards the player.
        :param shot_end Position in player court where the ball must end after completion of this shot.
        """
        super().__init__(
            "systemShot", system, ball, system_at, transit_start, shot_start, shot_end
        )

    def render_event(self, canvas, pristine):
        self.render_shot(canvas, pristine)


class PlayerMiss(RenderEvent):
    def __init__(
        self, player, ball, player_at, transit_start, shot_start, player_end, score
    ):
        """
        :param player_at Starting position of player to hit this shot.
        :param transit_start Position of ball at net from previous shot traveling towards the player.
        :param shot_start Position of ball where shot would begin if not missed.
        :param score Score updated as a result of this miss
        :param player_end Player's end position if player moved during this but couldn't reach shot_start.
        """
        super().__init__(
            "playerMiss",
            player,
            ball,
            player_at,
            transit_start,
            shot_start,
            actor_end=player_end,
        )
        self._score = score

    def render_event(self, canvas, pristine):
        self.render_miss(canvas, pristine)
        self.render_score(canvas, self._score)
        self.render_frame(canvas)


class SystemMiss(RenderEvent):
    def __init__(
        self, system, ball, system_at, transit_start, shot_start, system_end, score
    ):
        """
        :param system_at Starting position of system to hit this shot.
        :param transit_start Position of ball at net from previous shot traveling towards the system.
        :param shot_start Position of ball where shot would begin if not missed.
        :param score Score updated as a result of this miss
        :param system_end System's end position if system moved during this but couldn't reach shot_start.
        """
        super().__init__(
            "systemMiss",
            system,
            ball,
            system_at,
            transit_start,
            shot_start,
            actor_end=system_end,
        )
        self._score = score

    def render_event(self, canvas, pristine):
        self.render_miss(canvas, pristine)
        self.render_score(canvas, self._score)
        self.render_frame(canvas)


class PlayerMove(RenderEvent):
    def __init__(self, player, player_at, player_end):
        """
        :param player_at Player with its current position set to the starting position.
        :param player_to Ending position of player.
        """
        super().__init__("playerMove", player, actor_at=player_at, actor_end=player_end)

    def render_event(self, canvas, pristine):
        self.render_actor_move(canvas, pristine)


class SystemMove(RenderEvent):
    def __init__(self, system, system_at, system_end):
        """
        :param system_at System with its current position set to the starting position.
        :param system_end Ending position of system.
        """
        super().__init__("systemMove", system, actor_at=system_at, actor_end=system_end)

    def render_event(self, canvas, pristine):
        self.render_actor_move(canvas, pristine)


class TennisBehavior:
    """
    Tennis behavior interface that enables the user to specify a behavior to be exhibited by the
    environment.
    """

    def is_player_reachable(self, ball_position, player_position):
        """
        Check if the player can reach the shot ending with the ball at specified position while the
        player is at specified position.

        :param ball_position    Tuple (x, y, owner) describing ball's end position.
        :param player_position  Typle (x, y) describing player's current position.
        :return True if the player is reachable
        """
        return True

    def is_system_reachable(self, ball_position, system_position):
        """
        Check if the system can reach the shot ending with the ball at specified position while the
        system is at specified position.

        :param ball_position    Tuple (x, y, owner) describing ball's end position.
        :param system_position  Tuple (x, y) describing system's current position.
        :return True if the system is reachable
        """
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
        return True

    def shot_target(self, actor, actor_position, peer_position):
        """
        Determine the ball target position where shot will end.

        :param actor Actor who is hitting the ball.
        :param actor_position Tuple (x, y) of the actor's current position
        :param peer_position Tuple (x, y) of the peer's current position at the time actor fires the shot.
        :return Target ball position where the shot will end (x, y, owner)
        """
        return None

    def next_system_action(self, fire, env):
        """
        Take the next action on behalf of the system. This method updates the environment as a result of taking
        the step. Additionally, push any renderable events to enable visualization if needed. The method assumes
        that for the system's fire-shot, the shot is a valid shot within the reach of the system.

        :param fire If true, this step is a fire step for the system where the system is expected to hit
            a shot, thereby putting the ball in the player's court.
        :param env Environment to be updated upon taking the system action.
        :return 0 if the current game continues, 1 if current set is over, 2 if the entire episode is completed.
        """
        return 0

    def on_end_game(self):
        """
        Current game (most granular) has ended. Cleanup any behavior before starting the next game in the current set.
        """
        pass

    def compute_reward(self, activity):
        """
        Compute reward value for the specified activity that occured during the step.

        :param activity Activity that occured during the step.
        """
        return 0


class TennisStats:
    def game_result(
        self,
        episode,
        event_type,
        winner,
        player_game,
        system_game,
        player_set,
        system_set,
        player_sets,
        system_sets,
        shots,
    ):
        """
        Process result of a tennis game.

        :param episode 1-based current episode number
        :param event_type Type of result win event 1 - game, 2 - set, 3 - match
        :param winner    Winner entity - 0 player, 1 system
        :param player_game Player's score before the game completion.
        :param system_game System's score before the game completion.
        :param player_set Player's games won in current set on completion of current game.
        :param system_set System's games won in current set on completion of current game.
        :param player_sets Player's total sets won in current match on completion of current game.
        :param system_sets System's total sets won in current match on completion of current game.
        :param shots     Number of shots played during this game.
        """
        pass


class TennisObservedState:
    """
    Tennis state management utility class that allows the users to operate on different components of the
    observed state of the environment typically returned from the reset and step calls.
    """

    @staticmethod
    def get_player_position(state):
        """
        Get the (x, y) position of player: x from 0.0 (left) to 1.0 (right), y from BASE_LINE to NET_LINE

        :param state State to be observed.
        :return Tuple (x, y) for player's position.
        """
        return (
            state[DiscreteTennis.STATE_PLAYER_X],
            state[DiscreteTennis.STATE_PLAYER_Y],
        )

    @staticmethod
    def get_system_position(state):
        """
        Get the (x, y) position of system: x from 0.0 (left) to 1.0 (right), y from BASE_LINE to NET_LINE

        :param state State to be observed.
        :return Tuple (x, y) for system's position.
        """
        return (
            state[DiscreteTennis.STATE_SYSTEM_X],
            state[DiscreteTennis.STATE_SYSTEM_Y],
        )

    @staticmethod
    def get_ball_position(state):
        """
        Get the (x, y, owner) position of ball: owner PLAYER/SYSTEM, x from 0.0 (left) to 1.0 (right), y from BASE_LINE to NET_LINE

        :param state State to be observed.
        :return Tuple (x, y, owner) for ball's position.
        """
        return (
            state[DiscreteTennis.STATE_BALL_X],
            state[DiscreteTennis.STATE_BALL_Y],
            state[DiscreteTennis.STATE_BALL_OWNER],
        )

    @staticmethod
    def get_ball_source_position(state):
        """
        Get the (x, y, owner) position of source-ball: owner PLAYER/SYSTEM, x from 0.0 (left) to 1.0 (right), y from BASE_LINE to NET_LINE

        :param state State to be observed.
        :return Tuple (x, y, owner) for source-ball's position.
        """
        return (
            state[DiscreteTennis.STATE_SRCBALL_X],
            state[DiscreteTennis.STATE_SRCBALL_Y],
            state[DiscreteTennis.STATE_SRCBALL_OWNER],
        )

    @staticmethod
    def get_server(state):
        """
        Get who is serving

        :param state State to be observed.
        :return Serving entity PLAYER/SYSTEM
        """
        return state[DiscreteTennis.STATE_SERVE]

    @staticmethod
    def get_score(state):
        """
        Get the game scores

        :param state State to be observed.
        :return Tuple (game-player-score, game-system-score, set-player-score, set-system-score)
        """
        return (
            state[DiscreteTennis.STATE_GAMESCORE_PLAYER],
            state[DiscreteTennis.STATE_GAMESCORE_SYSTEM],
            state[DiscreteTennis.STATE_SETSCORE_PLAYER],
            state[DiscreteTennis.STATE_SETSCORE_SYSTEM],
            state[DiscreteTennis.STATE_MATCHSCORE_PLAYER],
            state[DiscreteTennis.STATE_MATCHSCORE_SYSTEM],
        )

    @staticmethod
    def get_score_num(state):
        """
        Get the game scores converted to numbers.

        :param state State to be observed.
        :return Tuple (game-player-score, game-system-score, set-player-score, set-system-score) each
            represented as numbers
        """
        return (
            TennisObservedState.to_score_num(
                state[DiscreteTennis.STATE_GAMESCORE_PLAYER]
            ),
            TennisObservedState.to_score_num(
                state[DiscreteTennis.STATE_GAMESCORE_SYSTEM]
            ),
            TennisObservedState.to_score_num(
                state[DiscreteTennis.STATE_SETSCORE_PLAYER]
            ),
            TennisObservedState.to_score_num(
                state[DiscreteTennis.STATE_SETSCORE_SYSTEM]
            ),
            TennisObservedState.to_score_num(
                state[DiscreteTennis.STATE_MATCHSCORE_PLAYER]
            ),
            TennisObservedState.to_score_num(
                state[DiscreteTennis.STATE_MATCHSCORE_SYSTEM]
            ),
        )

    @staticmethod
    def get_current_game_result(state):
        """
        Get the status of the current game's completion.

        :param state State to be observed.
        :return 0 if game continues, +1 if completed with player winning, -1 if completed with system winning.
        """
        return state[DiscreteTennis.STATE_CURRENTGAME_RESULT]

    @staticmethod
    def get_next_server(state):
        """
        Get the entity to serve the next game upon completion of the current game.

        :param state State to be observed.
        :return DiscreteTennis.PLAYER if player is to serve else DiscreteTennis.SYSTEM
        """
        if (
            state[DiscreteTennis.STATE_GAMESCORE_PLAYER] == 0.0
            and state[DiscreteTennis.STATE_GAMESCORE_SYSTEM] == 0.0
        ):  # New set resets game-scores.
            # Alternate serves between player and system at end of each set.
            serving_actor = (
                DiscreteTennis.PLAYER
                if state[DiscreteTennis.STATE_SERVE] == DiscreteTennis.SYSTEM
                else DiscreteTennis.SYSTEM
            )
        else:
            # Current server continues with serve for this new game.
            serving_actor = state[DiscreteTennis.STATE_SERVE]
        return serving_actor

    @staticmethod
    def to_score_num(value):
        return int(round(value * 10, 1))

    @staticmethod
    def fm_score_num(value):
        return float(value) / 10


class TennisState(TennisObservedState):
    @staticmethod
    def set_player_position(state, x, y):
        """
        Set the (x, y) position of player.

        :param state State to be set.
        :param x    Horizontal offset from left edge of court (0.0-1.0).
        :param y    Vertical offset from top edge of court (0.0-1.0).
        """
        if x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0:
            return False
        state[DiscreteTennis.STATE_PLAYER_X] = x
        state[DiscreteTennis.STATE_PLAYER_Y] = y
        return True

    @staticmethod
    def set_system_position(state, x, y):
        """
        Set the (x, y) position of system.

        :param state State to be set.
        :param x    Horizontal offset from left edge of court (0.0-1.0).
        :param y    Vertical offset from top edge of court (0.0-1.0).
        """
        if x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0:
            return False
        state[DiscreteTennis.STATE_SYSTEM_X] = x
        state[DiscreteTennis.STATE_SYSTEM_Y] = y
        return True

    @staticmethod
    def set_ball_position(state, x, y, owner):
        """
        Set the ball's ownership and position of ball.

        :param state State to be set.
        :param owner Ball owner PLAYER/SYSTEM.
        :param x    Horizontal offset from left edge of court (0.0-1.0).
        :param y    Vertical offset from top edge of court (0.0-1.0).
        """
        if (
            (owner != DiscreteTennis.PLAYER and owner != DiscreteTennis.SYSTEM)
            or x < 0.0
            or x > 1.0
            or y < 0.0
            or y > 1.0
        ):
            return False
        state[DiscreteTennis.STATE_BALL_OWNER] = owner
        state[DiscreteTennis.STATE_BALL_X] = x
        state[DiscreteTennis.STATE_BALL_Y] = y
        return True

    @staticmethod
    def set_ball_source_position(state, x, y, owner):
        """
        Set the source ball's ownership and position.

        :param state State to be set.
        :param owner Ball owner PLAYER/SYSTEM.
        :param x    Horizontal offset from left edge of court (0.0-1.0).
        :param y    Vertical offset from top edge of court (0.0-1.0).
        """
        if (
            (owner != DiscreteTennis.PLAYER and owner != DiscreteTennis.SYSTEM)
            or x < 0.0
            or x > 1.0
            or y < 0.0
            or y > 1.0
        ):
            return False
        state[DiscreteTennis.STATE_SRCBALL_OWNER] = owner
        state[DiscreteTennis.STATE_SRCBALL_X] = x
        state[DiscreteTennis.STATE_SRCBALL_Y] = y
        return True

    @staticmethod
    def set_server(state, server):
        """
        Set who is serving

        :param state State to be set.
        :param server Serving entity PLAYER/SYSTEM
        """
        state[DiscreteTennis.STATE_SERVE] = server

    @staticmethod
    def set_score(
        state,
        player_game_score,
        system_game_score,
        playerset_score,
        systemset_score,
        player_match_score,
        system_match_score,
    ):
        """
        Set the score for current game and sets.

        :param state State to be set.
        :param player_game_score    Player's score for current game in current set
        :param system_game_score    System's score for current game in current set
        :param playerset_score     Number of games won in current set by the player
        :param systemset_score     Number of games won in current set won by the system
        :param player_match_score   Number of sets won by the player
        :param system_match_score   Number of sets won by the system
        """
        if (
            player_game_score < DiscreteTennis.GAMESCORE_MINIMUM
            or player_game_score > DiscreteTennis.GAMESCORE_MAXIMUM
            or system_game_score < DiscreteTennis.GAMESCORE_MINIMUM
            or system_game_score > DiscreteTennis.GAMESCORE_MAXIMUM
            or playerset_score < DiscreteTennis.SETSCORE_MINIMUM
            or playerset_score > DiscreteTennis.SETSCORE_MAXIMUM
            or systemset_score < DiscreteTennis.SETSCORE_MINIMUM
            or systemset_score > DiscreteTennis.SETSCORE_MAXIMUM
            or player_match_score < DiscreteTennis.MATCHSCORE_MINIMUM
            or player_match_score > DiscreteTennis.MATCHSCORE_MAXIMUM
            or system_match_score < DiscreteTennis.MATCHSCORE_MINIMUM
            or system_match_score > DiscreteTennis.MATCHSCORE_MAXIMUM
        ):
            return False
        state[DiscreteTennis.STATE_GAMESCORE_PLAYER] = player_game_score
        state[DiscreteTennis.STATE_GAMESCORE_SYSTEM] = system_game_score
        state[DiscreteTennis.STATE_SETSCORE_PLAYER] = playerset_score
        state[DiscreteTennis.STATE_SETSCORE_SYSTEM] = systemset_score
        state[DiscreteTennis.STATE_MATCHSCORE_PLAYER] = player_match_score
        state[DiscreteTennis.STATE_MATCHSCORE_SYSTEM] = system_match_score
        return True

    @staticmethod
    def set_score_num(
        state,
        player_game_score,
        system_game_score,
        playerset_score,
        systemset_score,
        player_match_score,
        system_match_score,
    ):
        """
        Set the score for current game and sets input being provided as numbers

        :param state State to be set.
        :param player_game_score    Player's score for current game in current set
        :param system_game_score    System's score for current game in current set
        :param playerset_score     Number of games won in current set by the player
        :param systemset_score     Number of games won in current set won by the system
        :param player_match_score   Number of sets won by the player
        :param system_match_score   Number of sets won by the system
        """
        player_game_score = TennisObservedState.fm_score_num(player_game_score)
        system_game_score = TennisObservedState.fm_score_num(system_game_score)
        playerset_score = TennisObservedState.fm_score_num(playerset_score)
        systemset_score = TennisObservedState.fm_score_num(systemset_score)
        player_match_score = TennisObservedState.fm_score_num(player_match_score)
        system_match_score = TennisObservedState.fm_score_num(system_match_score)
        return TennisState.set_score(
            state,
            player_game_score,
            system_game_score,
            playerset_score,
            systemset_score,
            player_match_score,
            system_match_score,
        )

    @staticmethod
    def set_current_game_result(state, value):
        """
        Set the status of the current game's completion.

        :param state State to be set.
        :param value 0 if game continues, +1 if completed with player winning, -1 if completed with system winning.
        """
        state[DiscreteTennis.STATE_CURRENTGAME_RESULT] = value
        return True


def main(argv):
    env = DiscreteTennis(TennisBehavior())
    env.reset()

    while True:
        # Take a random action
        action = env.action_space.sample()
        # obs, reward, done, truncated, info = env.step(action)
        _, _, done, _, _ = env.step(action)

        # Render the game
        env.render()

        if done == True:
            break

    env.close()


if __name__ == "__main__":
    main(sys.argv)
