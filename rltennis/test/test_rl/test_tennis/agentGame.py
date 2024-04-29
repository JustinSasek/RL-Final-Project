from rl.tennis.discreteTennis import DiscreteTennis
from rl.tennis.stats import GameStats, ReportGenerator, GraphGenerator,\
    QTableRenderer
from rl.util.rtconfig import tc_runtime_configure, tc_closest_filepath
from rl.tennis.behaviorLearnable import LearnableTennisBehavior
from rl.tennis.agentBase import EGreedyPolicy
import argparse
import sys
import os
import logging
from time import process_time, localtime, strftime, sleep
from threading import Thread
import shutil
import re
from rl.tennis.discreteTennisWrappers import DiscreteTennisEasy, DiscreteTennisHard
#from rl.tennis.behaviorNondet import SimpleDiscreteTennisBehavior

#Profiling
from cProfile import Profile
from pstats import Stats
from rl.tennis.agentFactory import AgentFactory

LOGGER_BASE = "test.rl.tennis.GameControl"

class GameControl:
    _logger = logging.getLogger(LOGGER_BASE)
    _LOG_TAG = {"lgmod": "RLAGT"}
    
    # def __init__(self, seed = None, out_dir = None, model_file = None, stats_file = None, capture_file = None, 
    #     summary_file = None, monitor_file = None, num_episode = 100, 
    #     agent_type = "SARSA", agent_config = None, view = False, model_save_freq = None,
    #     greedy_percent=0.05, learn_rate=0.2, discount_factor=0.95):
    
    def __init__(self, config = {}, agent_config = {}, behavior_config = {}):
        self.config = config
        self.agent_config = agent_config
        # Preempt training and accept the model as of current state.
        self.preempt_training = False 
        
        self.seed = config.get("seed", 20)
        self.agent_type = config.get("agent_type", "SARSA")
        self.out_dir = config.get("out_dir")
        
        self.num_episode = config.get("num_episode")
        
        self.start_time = process_time()
        
        self.model_save_freq = config.get("model_save_freq")
        self.modeldir = None
        if self.model_save_freq is not None and self.model_save_freq > 0:
            self.modeldir = os.path.sep.join([self.out_dir, "models"])
            if os.path.exists(self.modeldir):
                shutil.rmtree(self.modeldir)
            os.mkdir(self.modeldir)
        
        self.behavior = LearnableTennisBehavior(behavior_config)
        #self.behavior = SimpleDiscreteTennisBehavior(seed = self.seed)
        cfg_file = config.get("summary_file")
        self.summary_fd = open(cfg_file, "w")
        
        cfg_file = config.get("monitor_file")
        self.monitor_fd = open(cfg_file, "w")
        
        self.env = DiscreteTennis(self.behavior)
        
        cfg_file = config.get("capture_file")
        if cfg_file is not None and len(cfg_file) > 0:
            self.env.set_capture_action(cfg_file)
        
        self.stats_file = config.get("stats_file")
        if self.stats_file is not None and len(self.stats_file) > 0:
            self.env.set_stats_logger(GameStats(self.stats_file))
            
        self.env._render_view = config.get("view", False)
        self.init_observed_state = self.env.reset()
        
        policy = EGreedyPolicy(config.get("greedy_percent", 0.05), self.behavior.random)
        # An implementation of TennisAgent interface.
        self.agent = AgentFactory.get_default_factory().create_agent(self.agent_type, agent_config, self.behavior, policy, self.init_observed_state);
        
        agent_config = config.get("agent_config")
        if agent_config is not None:
            self.agent.configure(agent_config)
        self._print_launch_config()
        
        self.curr_episode = 0
        self.env.render()
        self.agent_training_active = False
        self.agent_play_active = False
        self.shutdown = False
        
    def _print_launch_config(self):
        launch_cfg_msg = "GameControl Launched At {}:" +\
                "\n\n\tGAME CONTROL CONFIG:" +\
                "\n\tAgent Type={}" +\
                "\n\tEpisodes={}"
        launch_cfg_msg = launch_cfg_msg.format(
            strftime("%Y-%m-%d %H:%M:%S", localtime()),
            self.agent_type,
            self.num_episode)
        
        for cfg_item in self.config:
            seg_msg = "\n\t" + str(cfg_item) + "=" + str(self.config.get(cfg_item))
            launch_cfg_msg = launch_cfg_msg + seg_msg
            
        launch_cfg_msg = launch_cfg_msg + "\n\n\tAGENT CONFIG:"
        for cfg_item in self.agent_config:
            seg_msg = "\n\t" + str(cfg_item) + "=" + str(self.agent_config.get(cfg_item))
            launch_cfg_msg = launch_cfg_msg + seg_msg
            
        if (self._logger.isEnabledFor(logging.ERROR)):
            self._logger.error(launch_cfg_msg, extra=self._LOG_TAG)
        print(launch_cfg_msg)
        
    def _compute_dot_progress(self):
        # Show atleast 100 lines with atmost 100 dots each.
        num_episodes_per_line = int(self.num_episode/100)
        if num_episodes_per_line <= 100:
            num_episodes_per_line = 100
            num_episodes_per_dot = 1
        else:
            num_episodes_per_dot = int(num_episodes_per_line/100)
            if num_episodes_per_dot > 10:
                num_episodes_per_dot = 10
            num_episodes_per_line = num_episodes_per_dot * 100
        return (num_episodes_per_dot, num_episodes_per_line)
    
    def train(self):
        if self.agent_training_active:
            return False    # Training already active
        
        self.agent_training_active = True
        self.train_thread = Thread(target = self._do_train)
        self.train_thread.start()
        
    def _profile_train(self):
        if self.agent_training_active:
            return False    # Training already active
        self.agent_training_active = True
        self.train_thread = Thread(target = self._do_profile_train)
        self.train_thread.start()
    
    def _do_profile_train(self):
        with Profile() as profile:
            self._do_train()
            bin_profile_file = os.path.sep.join([self.out_dir, "agentGame.bin"])
            profile.dump_stats(bin_profile_file)
        
        self._do_dump_profile_stats(bin_profile_file)
    
    def _do_dump_profile_stats(self, bin_profile_file):
        profile_file = os.path.sep.join([self.out_dir, "agentGame.profile"])
        with open(profile_file, 'w') as out_profile_stream:
            stats = Stats(bin_profile_file, stream=out_profile_stream)
            stats.sort_stats("cumulative").print_stats()
        
    def _do_train(self):
        done = False
        truncated = False
        self.curr_episode = 0
        sleep(3)
        num_episodes_per_dot, num_episodes_per_line = self._compute_dot_progress()
        
        training_msg = "\n\nStarting Training at: {}".format(strftime("%Y-%m-%d %H:%M:%S", localtime())) 
        if (self._logger.isEnabledFor(logging.ERROR)):
            self._logger.error(training_msg, extra=self._LOG_TAG)
        print(training_msg)
        
        print("\nEach line shows {} episodes, {} per dot".format(num_episodes_per_line, num_episodes_per_dot))
        print("{:>6d}".format(self.curr_episode), end="")
        start_time = process_time()
        init_time = start_time
        end_time = None
        
        if (self._logger.isEnabledFor(logging.INFO)):
            self._logger.info("Episode-Start: {}".format(self.curr_episode),
                extra=self._LOG_TAG)
        
        model_save_first = 1
        if self.model_save_freq is not None:
            model_save_rate = int(self.num_episode/self.model_save_freq)
            model_save_first = 10 if model_save_rate > 10 else 1
        model_save_count = 0  
              
        step_count = 0
        
        while self.curr_episode < self.num_episode and not self.preempt_training and not self.shutdown:
            action = self.agent.next_action()
            obs, reward, done, truncated, info = self.env.step(action)
            # Render the game
            self.env.render()
            self.agent.reinforce(obs, reward, done, truncated, info, self.curr_episode, step_count)
            step_count = step_count + 1
            
            if done == True:
                end_time = process_time()
                episode_millis = (end_time - start_time) * 1000
                self.summary_fd.write("\nEpisode {} in {:.3f} ms: {}".\
                    format(self.curr_episode, episode_millis, self.env.tostr_score(True)))
                self.summary_fd.flush()
                if (self._logger.isEnabledFor(logging.INFO)):
                    self._logger.info("Episode-End: {}, Time={} ms, Steps={}".format(self.curr_episode, episode_millis, step_count),
                        extra=self._LOG_TAG)
                    
                    if self.curr_episode + 1 < self.num_episode:
                        self._logger.info("Episode-Start: {}".format((self.curr_episode + 1)),
                            extra=self._LOG_TAG)
                
                self.curr_episode = self.curr_episode + 1
                step_count = 0
                
                if self.model_save_freq is not None and ((self.curr_episode == model_save_first) or (self.curr_episode % self.model_save_freq) == 0):
                    self._save_model(model_save_count)
                    model_save_count = model_save_count + 1
                    
                if self.curr_episode % num_episodes_per_line == 0:
                    print("\n{:>6d}".format(self.curr_episode), end ="", flush=True)
                elif self.curr_episode % num_episodes_per_dot == 0:
                    print(".", end ="", flush=True)
                start_time = process_time()
                
        if self.shutdown and not self.preempt_training:
            print("Shutdown without preempt request - exiting without further saving model.")
            return
        
        final_strmodel = "\nEnd Training Model: {}".format(self.agent.tostr_model())
        self.summary_fd.write(final_strmodel)
        self.monitor_fd.write(final_strmodel)
        model_file = self.config.get("model_file")
        self.agent.save_model(model_file)
        
        train_status = "preempted" if self.preempt_training else "completed"
        training_time = end_time - init_time
        if (self._logger.isEnabledFor(logging.INFO)):
            self._logger.info("Training {} for {} episodes, Time={} s".format(train_status, self.curr_episode, training_time),
                extra=self._LOG_TAG)
            
        training_msg = "\n\nTraining {} at {} in {:.3f} s".format(train_status, strftime("%Y-%m-%d %H:%M:%S", localtime()), training_time)
        if (self._logger.isEnabledFor(logging.ERROR)):
            self._logger.error(training_msg, extra=self._LOG_TAG)
        print(training_msg)
        print("Model stored at: {}\nSummary logged at: {}".format(model_file, self.config.get("summary_file")))
        
        episodes_per_report = int(self.num_episode/100) if self.num_episode > 100 else 1 # Need around 100 points in graph.
        self.env.get_stats_logger().flush()
        
        print("Generating reports")
        self._generate_reports(self.stats_file, self.out_dir, episodes_per_report)
        print("Generated reports")
        
        self.agent_training_active = False
        self.train_thread = None
        
    def _save_model(self, model_count):
        """
        Save the agent model for current episode
        
        :param model_count    Counter of model being saved
        """
        file_path = os.path.sep.join([self.modeldir, "tn_{:06d}".format(self.curr_episode)])
        self.agent.save_model(file_path)
        
    def _render_episode_models(self, models_dir, out_dir):
        """
        Render QTables for saved episode models.
        
        :param models_dir    Directory where episode models are saved.
        :param out_dir       Directory where rendered episode models are to be saved.
        """
        pat = re.compile(r"tn_(\d+)*")
        
        count = 0
        tmp_agent = AgentFactory.get_default_factory().create_agent(self.agent_type, self.agent_config, self.behavior,
            EGreedyPolicy(self.config.get("greedy_percent"), self.behavior.random), self.init_observed_state);
        
        try:
            for filename in os.listdir(models_dir):
                mat = pat.match(filename)
                if mat is None:
                    continue    # Not valid model-name
                ep_str = mat.group(1)
                if ep_str is None or len(ep_str) == 0:
                    continue
                episode = int(ep_str)
                outfile = os.path.sep.join([out_dir, "qt_{:06d}.svg".format(episode)])
                infile = os.path.sep.join([models_dir, filename])
                tmp_agent.load_model(infile)
                qtable = tmp_agent.get_qtable()
                if qtable is not None:
                    renderer = QTableRenderer(qtable)
                    renderer.plot_table(outfile, "Q-Table Episode={}".format(episode))
                count = count + 1
            print("Rendered QTable for {} models".format(count))
        finally:
            tmp_agent.close()

    def _launch_play_episode(self):
        if self.agent_training_active or self.agent_play_active:
            return
        self.train_thread = Thread(target = self._agent_play_episode)
        self.train_thread.start()
                    
    def _agent_play_episode(self):
        self.agent_play_active = True
        self.env.reset()
        self.env.render()
        done = False
        while not done and not self.shutdown:
            action = self.agent.next_action()
            obs, _, done, _, _ = self.env.step(action)
            self.agent.prepare_action(obs)
            # Render the new update
            self.env.render()
            
        if self.shutdown:
            print("Exiting episode play due to shutdown request")
            
    def menu(self):
        menu_str = self._build_menu()
        while True:
            print("========================================\n")
            print(menu_str)
            action_str = input("Enter menu choice: ")
            print("________________________________________\n")
            action = None
            try:
                action = int(action_str)
            except:
                continue
            
            if action == 100:
                self.train()
                continue
            if action == 101:
                model_file_path = input("Enter path of model file: ")
                if not os.path.isfile(model_file_path):
                    print("Cannot access specified model file at: {}".format(model_file_path))
                    continue
                try:
                    self.agent.load_model(model_file_path)
                    print("Loaded model from: {}".format(model_file_path))
                except Exception as excp:
                    print("Failed loading model from: {}, reason: {}".format(model_file_path, excp))
                    continue
            elif action == 102:
                current_time = strftime("%H:%M:%S", localtime())
                curr_agent_modelstr = "\nAgent model at {}: {}".format(current_time, self.agent.tostr_model())
                self.monitor_fd.write(curr_agent_modelstr)
                self.monitor_fd.flush()
                print("Model dumped in: {}".format(self.config.get("monitor_file")))
                continue
            elif action == 103:
                self.env._render_view = True
                self.env._render_view_boundcorrect = True
                continue
            elif action == 104:
                self.env._render_view = False
                self.env._render_view_boundcorrect = False
                continue
            elif action == 105:
                self._launch_play_episode()
                continue
            elif action == 106:
                stats_file = input("Enter new statistics file: ")
                new_stats_logger = None
                try:
                    new_stats_logger = GameStats(stats_file)
                except Exception as excp:
                    print("Failed setting statistics file: {}, reason: {}".format(stats_file, excp))
                    continue
                old_stats_logger = self.env.set_stats_logger(new_stats_logger)
                if old_stats_logger is not None:
                    old_stats_logger.close()
                    old_stats_logger = None
                self.stats_file = stats_file
                continue
            elif action == 107:
                try:
                    episodes_per_report = input("Enter episodes/report (Default 100): ")
                    episodes_per_report = episodes_per_report.strip()
                    episodes_per_report = 100 if len(episodes_per_report) == 0 else int(episodes_per_report)
                    stats_file = input("Enter path of statistics input file (Enter for currently active stats): ")
                    stats_file = stats_file.strip()
                    if len(stats_file) == 0:
                        # Current file.
                        stats_file = self.stats_file
                        
                    if not os.path.isfile(stats_file):
                        print("Cannot access statistics file at: {}".format(stats_file))
                        continue
                    out_dir = input("Enter custom output directory (Enter for default): ")
                    out_dir = out_dir.strip()
                    if len(out_dir) == 0:
                        out_dir = self.out_dir
                    self._generate_reports(stats_file, out_dir, episodes_per_report)
                except Exception as excp:
                    print("ERROR: failed generating report: {}".format(excp))
                    print(excp.format_exc())
                continue
            elif action == 108:
                continue
            elif action == 109:
                self._profile_train()
                continue
            elif action == 110:
                bin_file = input("Enter binary profile file: ")
                try:
                    self._do_dump_profile_stats(bin_file)
                except Exception as excp:
                    print("Failed writing profile output: {}, reason: {}".format(bin_file, excp))
                continue
            elif action == 111:
                models_dir = input("Enter episode models directory (Enter for default):")
                models_dir = models_dir.strip()
                if len(models_dir) == 0:
                    models_dir = os.path.sep.join([self.out_dir, "models"])
                elif not os.path.exists(models_dir):
                    print("Cannot access models directory at={}".format(models_dir))
                    continue
                
                render_out_dir = input("Enter custom output directory (Enter for default): ")
                render_out_dir = render_out_dir.strip()
                if len(render_out_dir) == 0:
                    render_out_dir = os.path.sep.join([self.out_dir, "qtable"])
                    if os.path.exists(render_out_dir):
                        shutil.rmtree(render_out_dir)
                    os.mkdir(render_out_dir)
                elif not os.path.exists(render_out_dir):
                    print("Cannot access render output directory={}".format(render_out_dir))
                    continue
                try:
                    self._render_episode_models(models_dir, render_out_dir)
                except Exception as excp:
                    print("Failed rendering models under: {}, reason: {}".format(models_dir, excp))
                    continue
                continue
            elif action == 112:
                self.preempt_training = True
                continue
            elif action == 200:
                self.shutdown = True
                sleep(1)
                self.stop()
                print("Exiting")
                return
            
    def _build_menu(self):
        menu = ""
        menu = menu + "100: Train agent\n"
        menu = menu + "101: Load Model From File\n"
        menu = menu + "102: Print Model\n"
        menu = menu + "103: Show View\n"
        menu = menu + "104: Hide View\n"
        menu = menu + "105: Play Agent Episode\n"
        menu = menu + "106: Change Statistics File\n"
        menu = menu + "107: Reports for Statistics\n"
        menu = menu + "108: Refresh Menu\n"
        menu = menu + "109: Profile Train agent\n"
        menu = menu + "110: Dump profile sorted\n"
        menu = menu + "111: Render QTable for episode models\n"
        menu = menu + "112: Preempt Training (model will be saved)\n"
        menu = menu + "200: Exit\n"
        return menu
    
    def _generate_reports(self, stats_file, out_dir, episodes_per_report):
        report_generator = ReportGenerator(stats_file, episodes_per_report)
        _, report_list = report_generator.generate()
        
        graph_generator = GraphGenerator(out_dir, report_list)
        graph_generator.generate()
        
        qtable = self.agent.get_qtable()
        if qtable is not None: 
            qrenderer = QTableRenderer(qtable)
            qrenderer.plot_table(os.path.sep.join([self.out_dir, "qtableTrained.svg"]))
        
        print("Reports are now available under: {}".format(out_dir))
        summary = report_generator.tostr_whole_summary()
        print("\n------------------------------------------------------------------")
        print(summary)
        print("------------------------------------------------------------------")
    
    def stop(self):
        if self.summary_fd is not None:
            self.summary_fd.close()
            self.summary_fd = None
        if self.monitor_fd is not None:
            self.monitor_fd.close()
            self.monitor_fd = None
        if self.env is not None:
            self.env.close()
    
def percent_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("{} not a floating-point literal".format(x))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("{} not a percent in allowed range [0.0, 1.0]".format(x))
    return x

def add_config_item(name, cfg_name, cfg_dict, args = None, out_dir = None, override_value = None):
    config_value = None
    if override_value is not None:
        config_value = override_value
    
    elif args is not None:
        config_value = args.get(name)
    
    if config_value is not None:
        if out_dir is not None:
            config_value = config_value if os.path.isabs(config_value) else os.path.sep.join([out_dir, config_value])
        
        cfg_name = cfg_name if cfg_name is not None else name
        cfg_dict[cfg_name] = config_value
        
def main(argv):
    data_dir = tc_closest_filepath(__file__, "data")
    out_dir = os.path.sep.join([data_dir, "rl", "output"])
    ap = argparse.ArgumentParser()
    try:
        # Game Control Options
        ap.add_argument("-o", "--outdir", type=str, required=False, default=(out_dir),
            help="Output directory")
        ap.add_argument("-a", "--agentType", type=str, required=False, default=("SARSA"),
            help="Agent type: SARSA, DQN")
        ap.add_argument("-A", "--agentConfig", type=str, required=False,
            help="Agent configuration file")
        ap.add_argument("-S", "--statisticsFile", type=str, required=False, default=("tnstats.csv"),
            help="Statistics file")
        ap.add_argument("-m", "--modelFile", type=str, required=False, default=("tnmodel"),
            help="Model file")
        ap.add_argument("-M", "--monitorLog", type=str, required=False, default=("tnmonitor.log"),
            help="Monitor log file")
        ap.add_argument("-L", "--summaryLog", type=str, required=False, default=("tnsummary.log"),
            help="Summary log file")
        ap.add_argument("-c", "--capture", type=str, required=False, default=("actions.tn"),
            help="File to capture player actions")
        ap.add_argument("-E", "--episodeModelFreq", type=int, required=False, help="Save model every x episodes")
        ap.add_argument("-s", "--seed", type=int, required=False, help="Seed")
        ap.add_argument("-e", "--episode", type=int, required=False, help="Episode")
        ap.add_argument("-v", "--view", action="store_true", help="Run with animated view")
        ap.add_argument("-n", "--noninteractive", action="store_true", help="Non-interactive launch")
        ap.add_argument("-l", "--logcfg", type=str, required=False,
            help="Log configuration specification path")
        
        # Agent Options
        ap.add_argument("--greed", type=percent_float, required=False,
            help="Greed percent 0.0 to 1.0 of exploration, e.g. 0.05")
        ap.add_argument("--learnRate", type=percent_float, required=False,
            help="Agent's learn rate (alpha) 0.0 to 1.0 of exploration, e.g. 0.2")
        ap.add_argument("-d", "--discount", type=percent_float, required=False,
            help="Agent's discount factor (gamma) percent 0.0 to 1.0, e.g. 0.95")
        ap.add_argument("--dqnTau", type=percent_float, required=False,
            help="Tau for DQN model sync update percent 0.0 to 1.0, e.g. 0.15")
        ap.add_argument("--batchSize", type=int, required=False, 
            help="Batch size for batch-based model updates, e.g. 32")
        ap.add_argument("--replayBuffer", type=int, required=False, 
            help="Size of replay buffer for replay based model updating, e.g. 4000")
        
        # Behavior Options
        ap.add_argument("--difficulty", type=percent_float, required=False,
            help="Game difficulty level for adjusting system behavior 0.0 to 1.0, e.g. 0.95")
        args = vars(ap.parse_args(argv[1:]))
        
    except Exception as excp:
        print("ERROR: " + str(excp))
        ap.print_help();    
        sys.exit(2)

    log_cfg_file = args.get("logcfg")
    tc_runtime_configure({
        "logConfig": log_cfg_file if log_cfg_file is not None else tc_closest_filepath(__file__, "logConfig.json")
    })
    
    config_dict = {}
    out_dir = args.get("outdir")
    if not os.path.exists(out_dir):
        print("ERROR: Output directory does not exist - please create it and retry: {}".format(out_dir))
        return
    config_dict["out_dir"] = out_dir
    
    add_config_item("agentType", "agent_type", config_dict, args)
    add_config_item("agentConfig", "agent_config", config_dict, args)
    add_config_item("modelFile", "model_file", config_dict, args, out_dir=out_dir)
    add_config_item("capture", "capture_file", config_dict, args, out_dir=out_dir)
    add_config_item("summaryLog", "summary_file", config_dict, args, out_dir=out_dir)
    add_config_item("monitorLog", "monitor_file", config_dict, args, out_dir=out_dir)
    add_config_item("statisticsFile", "stats_file", config_dict, args, out_dir=out_dir)
    add_config_item("greed", "greedy_percent", config_dict, args)
    add_config_item("episodeModelFreq", "model_save_freq", config_dict, args)
    add_config_item("seed", "seed", config_dict, args)
    add_config_item("episode", "num_episode", config_dict, args)
    add_config_item("view", "view", config_dict, args)
    add_config_item("noninteractive", "noninteractive", config_dict, args)

    agent_config_dict = {}
    add_config_item("agentType", "agent_type", agent_config_dict, args)
    add_config_item("learnRate", "learn_rate", agent_config_dict, args)
    add_config_item("discount", "discount_factor", agent_config_dict, args)
    add_config_item("seed", "seed", agent_config_dict, args)
    add_config_item("greed", "greedy_percent", agent_config_dict, args)
    add_config_item("dqnTau", "dqn_tau", agent_config_dict, args)
    add_config_item("batchSize", "batch_size", agent_config_dict, args)
    add_config_item("replayBuffer", "replay_buffer", agent_config_dict, args)
    
    behavior_dict = {}
    add_config_item("seed", "seed", agent_config_dict, args)
    add_config_item("difficulty", "difficulty", behavior_dict, args)

    
    game_control = GameControl(config_dict, agent_config=agent_config_dict, behavior_config = behavior_dict)
    
    if config_dict.get("noninteractive"):
        game_control.train()
        game_control.train_thread.join()
    else:
        game_control.menu()
    
if __name__ == '__main__':
    main(sys.argv)      
