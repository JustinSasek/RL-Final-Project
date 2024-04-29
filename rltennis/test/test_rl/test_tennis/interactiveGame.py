from rl.tennis.discreteTennis import DiscreteTennis
from rl.tennis.discreteTennis import RenderEvent
from rl.util.rtconfig import tc_runtime_configure, tc_closest_filepath
#from rl.tennis.behaviorNondet import SimpleDiscreteTennisBehavior
from rl.tennis.behaviorLearnable import LearnableTennisBehavior
import argparse
import sys
import cv2

font = cv2.FONT_HERSHEY_COMPLEX_SMALL 

RENDER_ANIMATION_POSITION = (DiscreteTennis.MARGIN[0], int(DiscreteTennis.MARGIN[1] + DiscreteTennis.COURT[1] + 100))

RENDER_GAMESCORE_REV = None
RENDER_SETSCORE_REV = None 

def render_title(title, canvas):
    pos = RENDER_ANIMATION_POSITION
    canvas[pos[1] - 20:pos[1] + 30, pos[0 ]: pos[0] + 300] = 1
    cv2.putText(canvas, title, RENDER_ANIMATION_POSITION, font, 0.8, (0,255,0), 1, cv2.LINE_AA)
    
def toMenu(env):
    menu = ""
    for x in env.ACTION_SPEC:
        menu_item = "{}: {}".format(x, env.ACTION_SPEC[x].name)
        menu = menu + menu_item + "\n"
        
    menu = menu + "100: Reset Score\n"
    menu = menu + "101: Replay Actions From File\n"
    menu = menu + "102: Print score\n"
    menu = menu + "103: Exit\n"
    return menu

def to_score_num(score_str, env):
    global RENDER_GAMESCORE_REV
    global RENDER_SETSCORE_REV
    if RENDER_GAMESCORE_REV is None:
        RENDER_GAMESCORE_REV = {}    
        for x in env.RENDER_GAMESCORE:
            RENDER_GAMESCORE_REV[env.RENDER_GAMESCORE[x]] = x
    
    if RENDER_SETSCORE_REV is None:  
        RENDER_SETSCORE_REV = {}  
        for x in env.RENDER_SETSCORE:
            RENDER_SETSCORE_REV[env.RENDER_SETSCORE[x]] = x
            
    score_arr = score_str.split(",")
    score_num = []
    
    for score_elm in score_arr:
        score_elm = score_elm.strip()
        score_num.append(RENDER_GAMESCORE_REV[score_elm] if len(score_num) < 2 else RENDER_SETSCORE_REV[score_elm])
        
    return score_num

def replay_action(action_file, env):
    if action_file is None:
        print("No action file specified.")
        return False
    
    fd = open(action_file, "r")
    action_count = 0
    for line in fd:
        # Ignore lines which start with #
        if line.startswith("#"):
            continue
        
        action = int(line)
        _, _, done, _, _ = env.step(action)
        
        # Render the game
        env.render()
        
        action_count = action_count + 1
        if done:
            return (True, action_count)
        
    return (False, action_count)

def main(argv):
    capture_file = "/tmp/actions.tn"
    ap = argparse.ArgumentParser();
    try:
        ap.add_argument("-s", "--seed", type=int, required=False,
            help="Seed")
        ap.add_argument("-n", "--noview", action="store_true", help="Run without view")
        ap.add_argument("-l", "--logcfg", type=str, required=False,
            help="Log configuration specification path")
        ap.add_argument("-c", "--capture", type=str, required=False, default=(capture_file),
            help="File to capture player actions")
        args = vars(ap.parse_args(argv[1:]))
        
    except Exception as excp:
        print("ERROR: " + str(excp))
        ap.print_help();    
        sys.exit(2)

    log_cfg_file = args.get("logcfg")
    tc_runtime_configure({
        "logConfig": log_cfg_file if log_cfg_file is not None else tc_closest_filepath(__file__, "logConfig.json")
    })
    
    capture_file = args.get("capture")
    seed = args.get("seed")
    #env = DiscreteTennis(SimpleDiscreteTennisBehavior())
    env = DiscreteTennis(LearnableTennisBehavior({"seed": seed}))
    if capture_file is not None and len(capture_file) > 0:
        env.set_capture_action(capture_file)
        
    noview = args.get("noview")
    env._render_view = not noview if noview is not None else True
    env.reset()
    env.render()

    menu = toMenu(env)
    done = False
    while True:
        print("========================================\n")
        print(menu)
        action_str = input("Enter menu choice: ")
        print("________________________________________\n")
        action = int(action_str)
        if action == 100:
            score_str = input("Enter score to be reset (player-game,system-game,player-set,system-set,player-sets,system-sets) :")
            score_num = to_score_num(score_str, env)
            env._set_score_num(*score_num)
            RenderEvent.render_score(env.canvas, env._get_score())
            continue
        elif action == 101:
            action_file = input("Enter action replay file :")
            done, action_count = replay_action(action_file, env)
            if done:
                print("Game done after replaying {} actions from {}".format(action_count, action_file))
                break
            print("Finished replaying all {} actions from {}".format(action_count, action_file))
            continue
        elif action == 102:
            print(env.tostr_score(True))
            continue
        elif action == 103:
            print("Exiting game on user request")
            break
            
        print("Player action={}".format(action))
        
        # Take a random action
        #action = env.action_space.sample()
        _, _, done, _, _ = env.step(action)
        
        # Render the game
        env.render()
        
        if done == True:
            break

    print(env.tostr_score(True))   
    env.close()
    
if __name__ == '__main__':
    main(sys.argv)      