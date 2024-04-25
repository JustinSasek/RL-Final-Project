import argparse
import logging
import os
import random
import re
import shutil
import sys

# Profiling
from cProfile import Profile
from pstats import Stats
from threading import Thread
from time import localtime, process_time, sleep, strftime

import numpy as np
from rl.tennis.agentTrain import EGreedyPolicy, SarsaTennisAgent
from rl.tennis.discreteTennisWrappers import *
from rl.tennis.stats import GameStats, GraphGenerator, QTableRenderer, ReportGenerator
from rl.util.rtconfig import tc_closest_filepath, tc_runtime_configure


LOGGER_BASE = "test.rl.tennis.GameControl"


class GameControl:
    _logger = logging.getLogger(LOGGER_BASE)
    _LOG_TAG = {"lgmod": "RLAGT"}

    def __init__(
        self,
        seed,
        out_dir,
        model_file,
        stats_file,
        capture_file,
        summary_file,
        monitor_file,
        num_episode,
        view=False,
        model_save_freq=None,
    ):
        self.seed = seed
        self.out_dir = out_dir
        self.model_file = model_file
        self.stats_file = stats_file
        self.capture_file = capture_file
        self.summary_file = summary_file
        self.monitor_file = monitor_file
        self.num_episode = num_episode
        self.curr_episode = 0
        self.start_time = process_time()
        self.model_save_freq = model_save_freq
        self.modeldir = None
        if model_save_freq is not None and model_save_freq > 0:
            self.modeldir = os.path.sep.join([self.out_dir, "models"])
            if os.path.exists(self.modeldir):
                shutil.rmtree(self.modeldir)
            os.mkdir(self.modeldir)

        self.summary_fd = open(self.summary_file, "w")
        self.monitor_fd = open(self.monitor_file, "w")

        self.env = DiscreteTennisEasy(seed)
        if capture_file is not None and len(capture_file) > 0:
            self.env.set_capture_action(capture_file)

        if stats_file is not None and len(stats_file) > 0:
            self.env.set_stats_logger(GameStats(stats_file))

        self.env._render_view = view
        init_observed_state = self.env.reset()

        self.agent = SarsaTennisAgent(
            0.125, 0.125, EGreedyPolicy(0.05, random.Random(seed)), init_observed_state
        )

        self.env.render()
        self.agent_training_active = False

    def train(self):
        if self.agent_training_active:
            return False  # Training already active

        self.agent_training_active = True
        self.train_thread = Thread(target=self._do_train)
        self.train_thread.start()

    def _profile_train(self):
        if self.agent_training_active:
            return False  # Training already active
        self.agent_training_active = True
        self.train_thread = Thread(target=self._do_profile_train)
        self.train_thread.start()

    def _do_profile_train(self):
        with Profile() as profile:
            self._do_train()
            bin_profile_file = os.path.sep.join([self.out_dir, "agentGame.bin"])
            profile.dump_stats(bin_profile_file)

        self._do_dump_profile_stats(bin_profile_file)

    def _do_dump_profile_stats(self, bin_profile_file):
        profile_file = os.path.sep.join([self.out_dir, "agentGame.profile"])
        with open(profile_file, "w") as out_profile_stream:
            stats = Stats(bin_profile_file, stream=out_profile_stream)
            stats.sort_stats("cumulative").print_stats()

    def _do_train(self):
        done = False
        truncated = False
        self.curr_episode = 0
        sleep(3)
        print("\n\nStarting Training at: {}".format(strftime("%H:%M:%S", localtime())))
        print("{:>6d}".format(self.curr_episode), end="")
        start_time = process_time()
        init_time = start_time
        end_time = None

        if self._logger.isEnabledFor(logging.INFO):
            self._logger.info(
                "Episode-Start: {}".format(self.curr_episode), extra=self._LOG_TAG
            )

        model_save_first = 1
        if self.model_save_freq is not None:
            model_save_rate = int(self.num_episode / self.model_save_freq)
            model_save_first = 10 if model_save_rate > 10 else 1
        model_save_count = 0

        while self.curr_episode < self.num_episode:
            action = self.agent.next_action()
            obs, reward, done, truncated, info = self.env.step(action)
            # Render the game
            self.env.render()
            self.agent.reinforce(obs, reward, done, truncated, info)
            if done == True:
                end_time = process_time()
                episode_millis = (end_time - start_time) * 1000
                self.summary_fd.write(
                    "\nEpisode {} in {:.3f} ms: {}".format(
                        self.curr_episode, episode_millis, self.env.tostr_score(True)
                    )
                )
                self.summary_fd.flush()
                if self._logger.isEnabledFor(logging.INFO):
                    self._logger.info(
                        "Episode-End: {}, Time={} ms".format(
                            self.curr_episode, episode_millis
                        ),
                        extra=self._LOG_TAG,
                    )

                    if self.curr_episode + 1 < self.num_episode:
                        self._logger.info(
                            "Episode-Start: {}".format((self.curr_episode + 1)),
                            extra=self._LOG_TAG,
                        )

                self.curr_episode = self.curr_episode + 1

                if self.model_save_freq is not None and (
                    (self.curr_episode == model_save_first)
                    or (self.curr_episode % self.model_save_freq) == 0
                ):
                    self._save_model(model_save_count)
                    model_save_count = model_save_count + 1

                if self.curr_episode % 1000 == 0:
                    print("\n{:>6d}".format(self.curr_episode), end="", flush=True)
                elif self.curr_episode % 10 == 0:
                    print(".", end="", flush=True)
                start_time = process_time()

        final_qtable = "\nEnd Training Q-Table: {}".format(
            self.agent.tostr_qtable(self.agent.get_qtable())
        )
        self.summary_fd.write(final_qtable)
        self.monitor_fd.write(final_qtable)
        self.agent.save_model(self.model_file)

        training_time = end_time - init_time
        if self._logger.isEnabledFor(logging.INFO):
            self._logger.info(
                "Training completed for {} episodes, Time={} s".format(
                    self.num_episode, training_time
                ),
                extra=self._LOG_TAG,
            )

        print(
            "\n\nTraining completed at {} in {:.3f} s".format(
                strftime("%H:%M:%S", localtime()), training_time
            )
        )
        print(
            "Model stored at: {}\nSummary logged at: {}".format(
                self.model_file, self.summary_file
            )
        )

        episodes_per_report = (
            int(self.num_episode / 100) if self.num_episode > 100 else 1
        )  # Need around 100 points in graph.
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
        file_path = os.path.sep.join(
            [self.modeldir, "tn_{:06d}.npy".format(self.curr_episode)]
        )
        self.agent.save_model(file_path)

    def _render_episode_models(self, models_dir, out_dir):
        """
        Render QTables for saved episode models.

        :param models_dir    Directory where episode models are saved.
        :param out_dir       Directory where rendered episode models are to be saved.
        """
        pat = re.compile(r"tn_(\d+).npy")

        count = 0
        for filename in os.listdir(models_dir):
            mat = pat.match(filename)
            if mat is None:
                continue  # Not valid model-name
            ep_str = mat.group(1)
            if ep_str is None or len(ep_str) == 0:
                continue
            episode = int(ep_str)
            outfile = os.path.sep.join([out_dir, "qt_{:06d}.svg".format(episode)])
            infile = os.path.sep.join([models_dir, filename])
            qtable = np.load(infile)
            renderer = QTableRenderer(qtable)
            renderer.plot_table(outfile, "Q-Table Episode={}".format(episode))
            count = count + 1
        print("Rendered QTable for {} models".format(count))

    def _agent_play_episode(self):
        if self.agent_training_active:
            return

        done = False
        while not done:
            action = self.agent.next_action()
            _, _, done, _, _ = self.env.step(action)

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
                    print(
                        "Cannot access specified model file at: {}".format(
                            model_file_path
                        )
                    )
                    continue
                try:
                    self.agent.load_model(model_file_path)
                    print("Loaded model from: {}".format(model_file_path))
                except Exception as excp:
                    print(
                        "Failed loading model from: {}, reason: {}".format(
                            model_file_path, excp
                        )
                    )
                    continue
            elif action == 102:
                current_time = strftime("%H:%M:%S", localtime())
                curr_qtable = "\nQ-Table at {}: {}".format(
                    current_time, self.agent.tostr_qtable(self.agent.get_qtable())
                )
                self.monitor_fd.write(curr_qtable)
                self.monitor_fd.flush()
                print("Model dumped in: {}".format(self.monitor_file))
                continue
            elif action == 103:
                self.env._render_view = True
                continue
            elif action == 104:
                self.env._render_view = False
                continue
            elif action == 105:
                self._agent_play_episode()
                continue
            elif action == 106:
                stats_file = input("Enter new statistics file: ")
                new_stats_logger = None
                try:
                    new_stats_logger = GameStats(stats_file)
                except Exception as excp:
                    print(
                        "Failed setting statistics file: {}, reason: {}".format(
                            stats_file, excp
                        )
                    )
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
                    episodes_per_report = (
                        100
                        if len(episodes_per_report) == 0
                        else int(episodes_per_report)
                    )
                    stats_file = input(
                        "Enter path of statistics input file (Enter for currently active stats): "
                    )
                    stats_file = stats_file.strip()
                    if len(stats_file) == 0:
                        # Current file.
                        stats_file = self.stats_file

                    if not os.path.isfile(stats_file):
                        print("Cannot access statistics file at: {}".format(stats_file))
                        continue
                    out_dir = input(
                        "Enter custom output directory (Enter for default): "
                    )
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
                    print(
                        "Failed writing profile output: {}, reason: {}".format(
                            bin_file, excp
                        )
                    )
                continue
            elif action == 111:
                models_dir = input(
                    "Enter episode models directory (Enter for default):"
                )
                models_dir = models_dir.strip()
                if len(models_dir) == 0:
                    models_dir = os.path.sep.join([self.out_dir, "models"])
                elif not os.path.exists(models_dir):
                    print("Cannot access models directory at={}".format(models_dir))
                    continue

                render_out_dir = input(
                    "Enter custom output directory (Enter for default): "
                )
                render_out_dir = render_out_dir.strip()
                if len(render_out_dir) == 0:
                    render_out_dir = os.path.sep.join([self.out_dir, "qtable"])
                    if os.path.exists(render_out_dir):
                        shutil.rmtree(render_out_dir)
                    os.mkdir(render_out_dir)
                elif not os.path.exists(render_out_dir):
                    print(
                        "Cannot access render output directory={}".format(
                            render_out_dir
                        )
                    )
                    continue
                try:
                    self._render_episode_models(models_dir, render_out_dir)
                except Exception as excp:
                    print(
                        "Failed rendering models under: {}, reason: {}".format(
                            models_dir, excp
                        )
                    )
                    continue
                continue
            elif action == 200:
                self.stop()
                print("Exiting")
                return

    def _build_menu(self):
        menu = ""
        menu = menu + "100: Train agent\n"
        menu = menu + "101: Load Model From File\n"
        menu = menu + "102: Print Q-Table\n"
        menu = menu + "103: Show View\n"
        menu = menu + "104: Hide View\n"
        menu = menu + "105: Play Agent Episode\n"
        menu = menu + "106: Change Statistics File\n"
        menu = menu + "107: Reports for Statistics\n"
        menu = menu + "108: Refresh Menu\n"
        menu = menu + "109: Profile Train agent\n"
        menu = menu + "110: Dump profile sorted\n"
        menu = menu + "111: Render QTable for episode models\n"
        menu = menu + "200: Exit\n"
        return menu

    def _generate_reports(self, stats_file, out_dir, episodes_per_report):
        report_generator = ReportGenerator(stats_file, episodes_per_report)
        _, report_list = report_generator.generate()

        graph_generator = GraphGenerator(out_dir, report_list)
        graph_generator.generate()

        qrenderer = QTableRenderer(self.agent.get_qtable())
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


def to_path(args, purpose, out_dir):
    file_path = args.get(purpose)
    file_path = (
        file_path
        if os.path.isabs(file_path)
        else os.path.sep.join([out_dir, file_path])
    )
    return file_path


def main(argv):
    data_dir = tc_closest_filepath(__file__, "data")
    out_dir = os.path.sep.join([data_dir, "rl", "output"])
    ap = argparse.ArgumentParser()
    try:
        ap.add_argument(
            "-o",
            "--outdir",
            type=str,
            required=False,
            default=(out_dir),
            help="Output directory",
        )
        ap.add_argument(
            "-S",
            "--statisticsFile",
            type=str,
            required=False,
            default=("tnstats.csv"),
            help="Statistics file",
        )
        ap.add_argument(
            "-m",
            "--modelFile",
            type=str,
            required=False,
            default=("tnmodel.npy"),
            help="Model file",
        )
        ap.add_argument(
            "-M",
            "--monitorLog",
            type=str,
            required=False,
            default=("tnmonitor.log"),
            help="Monitor log file",
        )
        ap.add_argument(
            "-L",
            "--summaryLog",
            type=str,
            required=False,
            default=("tnsummary.log"),
            help="Summary log file",
        )
        ap.add_argument(
            "-c",
            "--capture",
            type=str,
            required=False,
            default=("actions.tn"),
            help="File to capture player actions",
        )
        ap.add_argument(
            "-E",
            "--episodeModelFreq",
            type=int,
            required=False,
            help="Save model every x episodes",
        )
        ap.add_argument("-s", "--seed", type=int, required=False, help="Seed")
        ap.add_argument("-e", "--episode", type=int, required=False, help="Episode")
        ap.add_argument(
            "-v", "--view", action="store_true", help="Run with animated view"
        )
        ap.add_argument(
            "-n", "--noninteractive", action="store_true", help="Non-interactive launch"
        )
        ap.add_argument(
            "-l",
            "--logcfg",
            type=str,
            required=False,
            help="Log configuration specification path",
        )

        args = vars(ap.parse_args(argv[1:]))

    except Exception as excp:
        print("ERROR: " + str(excp))
        ap.print_help()
        sys.exit(2)

    log_cfg_file = args.get("logcfg")
    tc_runtime_configure(
        {
            "logConfig": (
                log_cfg_file
                if log_cfg_file is not None
                else tc_closest_filepath(__file__, "logConfig.json")
            )
        }
    )

    out_dir = args.get("outdir")
    if not os.path.exists(out_dir):
        print(
            "ERROR: Output directory does not exist - please create it and retry: {}".format(
                out_dir
            )
        )
        return

    model_file = to_path(args, "modelFile", out_dir)
    capture_file = to_path(args, "capture", out_dir)
    summary_file = to_path(args, "summaryLog", out_dir)
    monitor_file = to_path(args, "monitorLog", out_dir)
    statistics_file = to_path(args, "statisticsFile", out_dir)
    model_save_freq = args.get("episodeModelFreq")
    seed = args.get("seed")
    num_episode = args.get("episode")
    show_animated_view = args.get("view")
    noninteractive = args.get("noninteractive")
    game_control = GameControl(
        seed,
        out_dir,
        model_file,
        statistics_file,
        capture_file,
        summary_file,
        monitor_file,
        num_episode,
        view=show_animated_view,
        model_save_freq=model_save_freq,
    )

    if noninteractive:
        game_control.train()
        game_control.train_thread.join()
    else:
        game_control.menu()


if __name__ == "__main__":
    main(sys.argv)
