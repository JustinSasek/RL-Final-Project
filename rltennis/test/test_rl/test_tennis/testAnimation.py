from rl.tennis.discreteTennis import DiscreteTennis
from rl.tennis.discreteTennis import PlayerShot
from rl.tennis.discreteTennis import PlayerMiss
from rl.tennis.discreteTennis import PlayerMove
from rl.tennis.discreteTennis import SystemShot
from rl.tennis.discreteTennis import SystemMiss
from rl.tennis.discreteTennis import SystemMove
from rl.tennis.behaviorNondet import SimpleDiscreteTennisBehavior
import sys
import cv2

font = cv2.FONT_HERSHEY_COMPLEX_SMALL 

RENDER_ANIMATION_POSITION = (DiscreteTennis.MARGIN[0], int(DiscreteTennis.MARGIN[1] + DiscreteTennis.COURT[1] + 100))

def render_title(title, canvas):
    pos = RENDER_ANIMATION_POSITION
    canvas[pos[1] - 20:pos[1] + 30, pos[0 ]: pos[0] + 300] = 1
    cv2.putText(canvas, title, RENDER_ANIMATION_POSITION, font, 0.8, (0,255,0), 1, cv2.LINE_AA)
    
# Player Straight-shot animation:
def test_player_straight_shot(env):
    render_title("Player Straight Shot", env.canvas)
    
    transit_start = env._position_to_point(env._to_render_position(env.PLAYER, (env.MIDDLE, env.NET_LINE)))
    shot_start = env._position_to_point(env._to_render_position(env.PLAYER, (env.MIDDLE, env.BASE_LINE)))
    shot_end = env._position_to_point(env._to_render_position(env.SYSTEM, (env.MIDDLE, env.BASE_LINE)))
    env._append_render_event(PlayerShot(env._player, env._ball, shot_start, transit_start, shot_start, shot_end))
    
# Player Cross-shot animation:
def test_player_cross_shot(env):
    render_title("Player Cross Shot", env.canvas)
    
    transit_start = env._position_to_point(env._to_render_position(env.PLAYER, (env.LEFT, env.NET_LINE)))
    shot_start = env._position_to_point(env._to_render_position(env.PLAYER, (env.MIDDLE, env.BASE_LINE)))
    shot_end = env._position_to_point(env._to_render_position(env.SYSTEM, (env.RIGHT, env.BASE_LINE)))
    env._append_render_event(PlayerShot(env._player, env._ball, shot_start, transit_start, shot_start, shot_end))

# Player Cross-shot animation:
def test_player_miss_straight(env):
    render_title("Player Miss Straight", env.canvas)
    
    player_pos = env._position_to_point(env._to_render_position(env.PLAYER, (env.LEFT, env.SERVE_LINE)))
    transit_start = env._position_to_point(env._to_render_position(env.PLAYER, (env.MIDDLE, env.NET_LINE)))
    shot_start = env._position_to_point(env._to_render_position(env.PLAYER, (env.MIDDLE, env.BASE_LINE)))
    player_end = env._position_to_point(env._to_render_position(env.PLAYER, (env.LEFT, env.BASE_LINE)))
    env._append_render_event(PlayerMiss(env._player, env._ball, player_pos, transit_start, shot_start, player_end, (0, 0, 0, 0, 0, 0)))

# Player Cross-shot animation:
def test_player_miss_cross(env):
    render_title("Player Miss Cross", env.canvas)
    
    player_pos = env._position_to_point(env._to_render_position(env.PLAYER, (env.LEFT, env.SERVE_LINE)))
    transit_start = env._position_to_point(env._to_render_position(env.PLAYER, (env.LEFT, env.NET_LINE)))
    shot_start = env._position_to_point(env._to_render_position(env.PLAYER, (env.MIDDLE, env.BASE_LINE)))
    player_end = env._position_to_point(env._to_render_position(env.PLAYER, (env.LEFT, env.BASE_LINE)))
    env._append_render_event(PlayerMiss(env._player, env._ball, player_pos, transit_start, shot_start, player_end, (0, 0, 0, 0, 0, 0)))

# System Straight-shot animation:
def test_system_straight_shot(env):
    render_title("System Straight Shot", env.canvas)
    
    transit_start = env._position_to_point(env._to_render_position(env.SYSTEM, (env.MIDDLE, env.NET_LINE)))
    shot_start = env._position_to_point(env._to_render_position(env.SYSTEM, (env.MIDDLE, env.BASE_LINE)))
    shot_end = env._position_to_point(env._to_render_position(env.PLAYER, (env.MIDDLE, env.BASE_LINE)))
    env._append_render_event(SystemShot(env._system, env._ball, shot_start, transit_start, shot_start, shot_end))
                
        # # System Straight-shot animation:
        # transit_start = Point(self._to_render_x(self.SYSTEM, self.MIDDLE), self._to_render_y(self.SYSTEM, self.NET_LINE))
        # shot_start = Point(self._to_render_x(self.SYSTEM, self.MIDDLE), self._to_render_y(self.SYSTEM, self.BASE_LINE))
        # shot_end = Point(self._to_render_x(self.PLAYER, self.MIDDLE), self._to_render_y(self.PLAYER, self.BASE_LINE))
        # self._render_arr.append(SystemShot(self._system, self._ball, None, transit_start, shot_start, shot_end))

# System Cross-shot animation:
def test_system_cross_shot(env):
    render_title("System Cross Shot", env.canvas)
    
    transit_start = env._position_to_point(env._to_render_position(env.SYSTEM, (env.LEFT, env.NET_LINE)))
    shot_start = env._position_to_point(env._to_render_position(env.SYSTEM, (env.MIDDLE, env.BASE_LINE)))
    shot_end = env._position_to_point(env._to_render_position(env.PLAYER, (env.RIGHT, env.BASE_LINE)))
    env._append_render_event(SystemShot(env._system, env._ball, shot_start, transit_start, shot_start, shot_end))
    
        # # System Cross-shot animation:
        # transit_start = Point(self._to_render_x(self.SYSTEM, self.LEFT), self._to_render_y(self.SYSTEM, self.NET_LINE))
        # shot_start = Point(self._to_render_x(self.SYSTEM, self.MIDDLE), self._to_render_y(self.SYSTEM, self.BASE_LINE))
        # shot_end = Point(self._to_render_x(self.PLAYER, self.RIGHT), self._to_render_y(self.PLAYER, self.BASE_LINE))
        # self._render_arr.append(SystemShot(self._system, self._ball, None, transit_start, shot_start, shot_end))

# System Cross-shot animation:
def test_system_miss_straight(env):
    render_title("System Miss Straight", env.canvas)
    
    system_pos = env._position_to_point(env._to_render_position(env.SYSTEM, (env.LEFT, env.SERVE_LINE)))
    transit_start = env._position_to_point(env._to_render_position(env.SYSTEM, (env.MIDDLE, env.NET_LINE)))
    shot_start = env._position_to_point(env._to_render_position(env.SYSTEM, (env.MIDDLE, env.BASE_LINE)))
    system_end = env._position_to_point(env._to_render_position(env.SYSTEM, (env.LEFT, env.BASE_LINE)))
    env._append_render_event(SystemMiss(env._system, env._ball, system_pos, transit_start, shot_start, system_end, (0, 0, 0, 0, 0, 0)))
            
        ## System Miss straight animation:
        # self._system.set_position(self._to_render_x(self.SYSTEM, self.LEFT), self._to_render_y(self.SYSTEM, self.SERVE_LINE))
        # transit_start = Point(self._to_render_x(self.SYSTEM, self.MIDDLE), self._to_render_y(self.SYSTEM, self.NET_LINE))
        # shot_start = Point(self._to_render_x(self.SYSTEM, self.MIDDLE), self._to_render_y(self.SYSTEM, self.BASE_LINE))
        # system_end = Point(self._to_render_x(self.SYSTEM, self.LEFT), self._to_render_y(self.SYSTEM, self.BASE_LINE))
        # self._render_arr.append(SystemMiss(self._system, self._ball, None, transit_start, shot_start, system_end, (0, 0, 0, 0)))

# System Miss cross-shot animation:
def test_system_miss_cross(env):
    render_title("System Miss Cross", env.canvas)
    
    system_pos = env._position_to_point(env._to_render_position(env.SYSTEM, (env.LEFT, env.SERVE_LINE)))
    transit_start = env._position_to_point(env._to_render_position(env.SYSTEM, (env.LEFT, env.NET_LINE)))
    shot_start = env._position_to_point(env._to_render_position(env.SYSTEM, (env.MIDDLE, env.BASE_LINE)))
    system_end = env._position_to_point(env._to_render_position(env.SYSTEM, (env.LEFT, env.BASE_LINE)))
    env._append_render_event(SystemMiss(env._system, env._ball, system_pos, transit_start, shot_start, system_end, (0, 0, 0, 0, 0, 0)))
    
        # # System Miss Cross-shot animation:
        # self._system.set_position(self._to_render_x(self.SYSTEM, self.LEFT), self._to_render_y(self.SYSTEM, self.SERVE_LINE))
        # transit_start = Point(self._to_render_x(self.SYSTEM, self.LEFT), self._to_render_y(self.SYSTEM, self.NET_LINE))
        # shot_start = Point(self._to_render_x(self.SYSTEM, self.MIDDLE), self._to_render_y(self.SYSTEM, self.BASE_LINE))
        # system_end = Point(self._to_render_x(self.SYSTEM, self.LEFT), self._to_render_y(self.SYSTEM, self.BASE_LINE))
        # self._render_arr.append(SystemMiss(self._system, self._ball, None, transit_start, shot_start, system_end, (0, 0, 0, 0)))

# Player Cross-shot + System Move animation:
def test_player_cross_system_move(env):
    render_title("Player Cross, System Move", env.canvas)
    
    transit_start = env._position_to_point(env._to_render_position(env.PLAYER, (env.LEFT, env.NET_LINE)))
    shot_start = env._position_to_point(env._to_render_position(env.PLAYER, (env.MIDDLE, env.BASE_LINE)))
    shot_end = env._position_to_point(env._to_render_position(env.SYSTEM, (env.RIGHT, env.BASE_LINE)))
    env._append_render_event(PlayerShot(env._player, env._ball, shot_start, transit_start, shot_start, shot_end))
    
    system_pos = env._position_to_point(env._to_render_position(env.SYSTEM, (env.MIDDLE, env.BASE_LINE)))
    actor_end = env._position_to_point(env._to_render_position(env.SYSTEM, (env.RIGHT, env.BASE_LINE)))
    env._append_render_event(SystemMove(env._system, system_pos, actor_end))
    
    
        # # Player Cross-shot animation with System Moving
        # transit_start = Point(self._to_render_x(self.PLAYER, self.LEFT), self._to_render_y(self.PLAYER, self.NET_LINE))
        # shot_start = Point(self._to_render_x(self.PLAYER, self.MIDDLE), self._to_render_y(self.PLAYER, self.BASE_LINE))
        # shot_end = Point(self._to_render_x(self.SYSTEM, self.RIGHT), self._to_render_y(self.SYSTEM, self.BASE_LINE))
        # shot_event = SystemShot(self._player, self._ball, None, transit_start, shot_start, shot_end)
        # self._system.set_position(self._to_render_x(self.SYSTEM, self.MIDDLE), self._to_render_y(self.SYSTEM, self.BASE_LINE))
        # actor_end = Point(self._to_render_x(self.SYSTEM, self.RIGHT), self._to_render_y(self.SYSTEM, self.BASE_LINE))
        # move_event = SystemMove(self._system, None, actor_end)
        # shot_event.set_move_peer_actor_event(move_event)
        # self._render_arr.append(shot_event)

# System Cross-shot + Player Move animation:
def test_system_cross_player_move(env):
    render_title("System Cross, Player Move", env.canvas)
    
    transit_start = env._position_to_point(env._to_render_position(env.SYSTEM, (env.LEFT, env.NET_LINE)))
    shot_start = env._position_to_point(env._to_render_position(env.SYSTEM, (env.MIDDLE, env.BASE_LINE)))
    shot_end = env._position_to_point(env._to_render_position(env.PLAYER, (env.RIGHT, env.BASE_LINE)))
    env._append_render_event(SystemShot(env._system, env._ball, shot_start, transit_start, shot_start, shot_end))
    
    player_pos = env._position_to_point(env._to_render_position(env.PLAYER, (env.MIDDLE, env.BASE_LINE)))
    actor_end = env._position_to_point(env._to_render_position(env.PLAYER, (env.RIGHT, env.BASE_LINE)))
    env._append_render_event(PlayerMove(env._player, player_pos, actor_end))
            
        # # System Cross-shot animation with Player Moving
        # transit_start = Point(self._to_render_x(self.SYSTEM, self.LEFT), self._to_render_y(self.SYSTEM, self.NET_LINE))
        # shot_start = Point(self._to_render_x(self.SYSTEM, self.MIDDLE), self._to_render_y(self.SYSTEM, self.BASE_LINE))
        # shot_end = Point(self._to_render_x(self.PLAYER, self.RIGHT), self._to_render_y(self.PLAYER, self.BASE_LINE))
        # shot_event = SystemShot(self._system, self._ball, None, transit_start, shot_start, shot_end)
        # self._player.set_position(self._to_render_x(self.PLAYER, self.MIDDLE), self._to_render_y(self.PLAYER, self.BASE_LINE))
        # actor_end = Point(self._to_render_x(self.PLAYER, self.RIGHT), self._to_render_y(self.PLAYER, self.BASE_LINE))
        # move_event = PlayerMove(self._player, None, actor_end)
        # shot_event.set_move_peer_actor_event(move_event)
        # self._render_arr.append(shot_event)
        
        # #################################################################################################################
        
        #return self._state, 1, False, False, []
    
animation_tests = [
    test_player_straight_shot,
    test_player_cross_shot,
    test_player_miss_straight,
    test_player_miss_cross,
    test_system_straight_shot,
    test_system_cross_shot,
    test_system_miss_straight,
    test_system_miss_cross,
    test_player_cross_system_move,
    test_system_cross_player_move,
]

def main(argv):
    env = DiscreteTennis(SimpleDiscreteTennisBehavior())
    env._render_view = True
    env.reset()
    env.set_animation_test(animation_tests[0])

    # Do each animation this many times.    
    item_count = 10
    
    count = 0
    while True:
        # Take a random action
        action = env.action_space.sample()
        _, _, done, _, _ = env.step(action)
        
        # Render the game
        env.render()
        
        count = count + 1
        if count % item_count == 0:
            # Change of item 
            next_index = int(count/item_count)
            if next_index >= len(animation_tests):
                break
            env.set_animation_test(animation_tests[next_index])
            cv2.waitKey(2000)
        
        if done == True:
            break
    
    env.close()
    
if __name__ == '__main__':
    main(sys.argv)      