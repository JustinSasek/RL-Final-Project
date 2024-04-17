import logging
import os
import sys
import time

import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Matplotlib normally cannot be launched outside main thread - to prevent this, the following workaround is used as
# suggested at: https://stackoverflow.com/questions/69924881/userwarning-starting-a-matplotlib-gui-outside-of-the-main-thread-will-likely-fa
# and https://matplotlib.org/stable/users/explain/figure/backends.html
from matplotlib import use as matplotuse
from rl.tennis.agentTrain import SarsaTennisAgent
from rl.tennis.discreteTennis import DiscreteTennis, TennisStats
from rl.util.rtconfig import tc_closest_filepath, tc_runtime_configure

matplotuse("agg")

LOGGER_BASE = "rl.tennis.stats"


def safe_div(numerator, denominator, on_zero=None, ret_zero=-1):
    """
    Safe division

    :param numerator    Numerator for division.
    :param denominator  Denominator for division.
    :param on_zero      Denominator to be used if denominator argument is zero.
    :param ret_zero     Value to be returned if denominator is zero
    """
    return (
        (numerator / denominator)
        if denominator != 0
        else (ret_zero if on_zero is None else (numerator / on_zero))
    )


class GameStats(TennisStats):
    def __init__(self, stats_file):
        self.stats_fd = open(stats_file, "w")
        self.stats_fd.write(
            "time,episode,event,winner,player_game,system_game,player_set,system_set,player_sets,system_sets,shots\n"
        )

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
        Log a game-result event.

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
        at_time = time.time_ns()
        buf = (
            str(at_time)
            + ","
            + str(episode)
            + ","
            + str(event_type)
            + ","
            + str(winner)
            + ","
            + str(player_game)
            + ","
            + str(system_game)
            + ","
            + str(player_set)
            + ","
            + str(system_set)
            + ","
            + str(player_sets)
            + ","
            + str(system_sets)
            + ","
            + str(shots)
            + "\n"
        )
        self.stats_fd.write(buf)

    def flush(self):
        self.stats_fd.flush()

    def close(self):
        self.stats_fd.close()


class SummaryReport:
    """
    Summary report for a set of episodes.
    """

    def __init__(self):
        # 0-based row number in the input statistics file where this report begins.
        self.start_row = 0
        self.end_row = None
        self.start_episode = None
        self.end_episode = None
        self._curr_episode = None

        self._episode_start_time = None
        self.episode_maxtime = -sys.maxsize - 1
        self.episode_max = None
        self.episode_mintime = sys.maxsize
        self.episode_min = None
        self.episode_sumtime = 0
        self.num_episode = 0

        self.player_game_win = 0
        self.system_game_win = 0
        self.player_set_win = 0
        self.system_set_win = 0
        self.player_match_win = 0
        self.system_match_win = 0

        self.player_game_sumwinmargin = 0
        self.system_game_sumwinmargin = 0
        self.player_set_sumwinmargin = 0
        self.system_set_sumwinmargin = 0
        self.player_match_sumwinmargin = 0
        self.system_match_sumwinmargin = 0

        self.total_shots = 0


class ReportGenerator:
    """
    Report generator that takes a training statistics CSV file and generates reports for these training episodes.
    The summary is organized by a sequence of reports, each representing a contiguous episode sequence, so as to
    illustrate how each subsequent report captures the effect of training. Additionally, a comprehensive report
    is also generated to represent all training episodes.
    """

    def __init__(self, stats_file, reportsize=1000):
        """
        Report generator constructor.

        :param stats_file Training statistics CSV file generated by a GameStats instance during agent training.
        :param reportsize    Number of episodes to be included in a report
        """
        # Statistics file
        self.stats_file = stats_file
        self._report = None
        self._whole_report = None
        self._report_list = []
        self._reportsize = reportsize

    def generate(self):
        """
        Generate the reports.

        :return Return a tuple comprising the whole report and a list of report sequence.
        """
        self._report = SummaryReport()
        self._report_list.append(self._report)

        chunksize = 1000  # process DF with 1000 rows at a time.
        row_index = None  # 0-based row index
        prev_row = None
        try:
            stats_fd = open(self.stats_file, "r")
            with pd.read_csv(stats_fd, chunksize=chunksize) as reader:
                for df in reader:
                    for row in df.itertuples():
                        row_index = row.Index
                        row_episode = row.episode
                        if row_episode != self._report._curr_episode:
                            self._new_episode(row_index, row_episode, row)

                        player_win = row.winner == DiscreteTennis.PLAYER
                        if player_win:
                            self._report.player_game_win = (
                                self._report.player_game_win + 1
                            )
                            self._report.player_game_sumwinmargin = (
                                self._report.player_game_sumwinmargin
                                + row.player_game
                                - row.system_game
                                + 1
                            )
                        else:
                            self._report.system_game_win = (
                                self._report.system_game_win + 1
                            )
                            self._report.system_game_sumwinmargin = (
                                self._report.system_game_sumwinmargin
                                + row.system_game
                                - row.player_game
                                + 1
                            )

                        if row.event == 2:  # Set win
                            if player_win:
                                self._report.player_set_win = (
                                    self._report.player_set_win + 1
                                )
                                self._report.player_set_sumwinmargin = (
                                    self._report.player_set_sumwinmargin
                                    + row.player_set
                                    - row.system_set
                                )
                            else:
                                self._report.system_set_win = (
                                    self._report.system_set_win + 1
                                )
                                self._report.system_set_sumwinmargin = (
                                    self._report.system_set_sumwinmargin
                                    + row.system_set
                                    - row.player_set
                                )
                        elif row.event == 3:  # Match win
                            if player_win:
                                self._report.player_match_win = (
                                    self._report.player_match_win + 1
                                )
                                self._report.player_match_sumwinmargin = (
                                    self._report.player_match_sumwinmargin
                                    + row.player_sets
                                    - row.system_sets
                                )
                            else:
                                self._report.system_match_win = (
                                    self._report.system_match_win + 1
                                )
                                self._report.system_match_sumwinmargin = (
                                    self._report.system_match_sumwinmargin
                                    + row.system_sets
                                    - row.player_sets
                                )

                        self._report.total_shots = self._report.total_shots + row.shots
                        prev_row = row
                        # print("{} {}".format(row.time, row.episode))
        finally:
            self._end_summary(stats_fd, row_index, prev_row)
        self._generate_whole_summary()
        return self.get_reports()

    def get_reports(self):
        """
        Get the generated reports

        :return Return a tuple comprising the whole report and a list of report sequence.
        """
        return (self._whole_report, self._report_list)

    def _new_episode(self, row_number, row_episode, row, is_last=False):
        """
        Handle a new episode encountered while iterating through each row of the input training statistics file.

        :param row_number 0-based index number of the row in input training statistics file.
        :param row_episode Episode number for which the row is being processed.
        :param row Row data obtained from the statistics file
        :param is_last Represents if this is the last episode.
        """
        if self._report.start_episode is None:
            self._report.start_episode = row_episode
            self._report.start_row = row_number

        row_episode_time = int(row.time / 1000000)
        if self._report._curr_episode is not None:
            # End of current, start of new.
            episode_duration = row_episode_time - self._report._episode_start_time
            if episode_duration > self._report.episode_maxtime:
                self._report.episode_maxtime = episode_duration
                self._report.episode_max = self._report._curr_episode

            if episode_duration < self._report.episode_mintime:
                self._report.episode_mintime = episode_duration
                self._report.episode_min = self._report._curr_episode

            self._report.episode_sumtime = (
                self._report.episode_sumtime + episode_duration
            )
            self._report.num_episode = (
                self._report.num_episode + 1
            )  # Completed this many episodes in current item.

        if self._report.num_episode >= self._reportsize or is_last:
            self._report.end_row = row_number
            self._report.end_episode = row_episode
            if not is_last:
                self.end_episode = row_episode
                # Create a new item for subsequent episodes.
                self._report = SummaryReport()
                self._report_list.append(self._report)
                self._report.start_episode = row_episode
                self._report.start_row = row_number

        if not is_last:
            self._report._curr_episode = row_episode
            self._report._episode_start_time = row_episode_time

    def _generate_whole_summary(self):
        """
        Generate a comprehensive summary report encompassing all episodes captured in the input training file.
        """
        if len(self._report_list) == 0:
            return None

        first_report = self._report_list[0]
        whole = SummaryReport()
        whole.start_episode = first_report.start_episode
        whole.start_row = first_report.start_row
        whole.episode_max = first_report.episode_max
        whole.episode_maxtime = first_report.episode_maxtime
        whole.episode_min = first_report.episode_min
        whole.episode_mintime = first_report.episode_mintime

        for report in self._report_list:
            if report.episode_max > whole.episode_max:
                whole.episode_max = report.episode_max
                whole.episode_maxtime = report.episode_maxtime

            if report.episode_min > whole.episode_min:
                whole.episode_min = report.episode_min
                whole.episode_mintime = report.episode_mintime

            whole.episode_sumtime = whole.episode_sumtime + report.episode_sumtime

            whole.player_game_win = whole.player_game_win + report.player_game_win
            whole.player_game_sumwinmargin = whole.player_game_sumwinmargin = (
                +report.player_game_sumwinmargin
            )

            whole.system_game_win = whole.system_game_win + report.system_game_win
            whole.system_game_sumwinmargin = whole.system_game_sumwinmargin = (
                +report.system_game_sumwinmargin
            )

            whole.player_set_win = whole.player_set_win + report.player_set_win
            whole.player_set_sumwinmargin = whole.player_set_sumwinmargin = (
                +report.player_set_sumwinmargin
            )

            whole.system_set_win = whole.system_set_win + report.system_set_win
            whole.system_set_sumwinmargin = whole.system_set_sumwinmargin = (
                +report.system_set_sumwinmargin
            )

            whole.player_match_win = whole.player_match_win + report.player_match_win
            whole.player_match_sumwinmargin = whole.player_match_sumwinmargin = (
                +report.player_match_sumwinmargin
            )

            whole.system_match_win = whole.system_match_win + report.system_match_win
            whole.system_match_sumwinmargin = whole.system_match_sumwinmargin = (
                +report.system_match_sumwinmargin
            )
            whole.num_episode = whole.num_episode + report.num_episode

        whole.end_episode = self._report_list[-1].end_episode
        whole.end_row = self._report_list[-1].end_row
        self._whole_report = whole

    def tostr_whole_summary(self):
        rp = self._whole_report
        buf = ""
        tot = rp.player_match_win + rp.system_match_win
        buf = (
            buf
            + "Match Won:  Player={:<4.1f}% {:<8d} System={:<4.1f}% {:<8d}\n".format(
                safe_div(rp.player_match_win, tot) * 100,
                rp.player_match_win,
                safe_div(rp.system_match_win, tot) * 100,
                rp.system_match_win,
            )
        )
        tot = rp.player_set_win + rp.system_set_win
        buf = (
            buf
            + "Sets Won:   Player={:<4.1f}% {:<8d} System={:<4.1f}% {:<8d}\n".format(
                safe_div(rp.player_set_win, tot) * 100,
                rp.player_set_win,
                safe_div(rp.system_set_win, tot) * 100,
                rp.system_set_win,
            )
        )
        tot = rp.player_game_win + rp.system_game_win
        buf = (
            buf
            + "Games Won:  Player={:<4.1f}% {:<8d} System={:<4.1f}% {:<8d}\n".format(
                safe_div(rp.player_game_win, tot) * 100,
                rp.player_game_win,
                safe_div(rp.system_game_win, tot) * 100,
                rp.system_game_win,
            )
        )
        buf = (
            buf
            + "Episodes:   {:<8d} (Msec) Avg={:<8.2f} Min={:<8.2f}@{} Max={:<8.2f}@{}\n".format(
                rp.num_episode,
                safe_div(rp.episode_sumtime, rp.num_episode),
                rp.episode_mintime,
                rp.episode_min,
                rp.episode_maxtime,
                rp.episode_max,
            )
        )
        buf = buf + "Margins:\n"
        buf = buf + "Game Won:   Player={:<8.2f} System={:<8.2f}\n".format(
            safe_div(rp.player_game_sumwinmargin, rp.player_game_win),
            safe_div(rp.system_game_sumwinmargin, rp.system_game_win),
        )
        buf = buf + "Set Won:    Player={:<8.2f} System={:<8.2f}\n".format(
            safe_div(rp.player_set_sumwinmargin, rp.player_set_win),
            safe_div(rp.system_set_sumwinmargin, rp.system_set_win),
        )
        buf = buf + "Match Won:  Player={:<8.2f} System={:<8.2f}\n".format(
            safe_div(rp.player_game_sumwinmargin, rp.player_match_win),
            safe_div(rp.system_game_sumwinmargin, rp.system_match_win),
        )
        return buf

    def _end_summary(self, stats_fd, last_row_index, last_row):
        """
        End summary report generation.

        :param stats_fd         Statistics file
        :param last_row_index   Index of the last row in the file.
        :param last_row         Last row data in the statistics file.
        """
        try:
            stats_fd.close()
        except:
            pass

        if self._report is not None:
            if self._report.num_episode == 0:
                self._report_list.pop()  # Remove just created list as no more elements are present.
            else:
                self._report.end_row = last_row_index + 1
                self._new_episode(last_row_index, last_row.episode, last_row, True)


class GraphGenerator:
    """
    Generate graphs that represents the training statistics in a graphical format.
    """

    def __init__(self, graph_dir, report_list):
        """
        Graph generator constructor

        :param graph_dir    Directory where graph output will be stored.
        :param report_list  List of reports to be graphed.
        """
        self._graph_dir = graph_dir
        self._report_list = report_list
        self._num_report = len(report_list)
        index = 0
        self._episode_arr = np.empty(self._num_report, dtype=int)
        for report in self._report_list:
            self._episode_arr[index] = report.end_episode
            index = index + 1

    def generate(self):
        """
        Generate the graphs.
        """
        self._generate_wins()
        self._generate_episodes()
        self._generate_win_margins()

    def _generate_episodes(self):
        """
        Generate graph of report's last episode v/s the average, min and max time of episodes in each report.
        """
        avg_arr = np.empty(self._num_report, dtype=int)
        min_arr = np.empty(self._num_report, dtype=int)
        max_arr = np.empty(self._num_report, dtype=int)

        index = 0
        for report in self._report_list:
            avg_arr[index] = int(safe_div(report.episode_sumtime, report.num_episode))
            min_arr[index] = report.episode_mintime
            max_arr[index] = report.episode_maxtime
            index = index + 1

        _, ax1 = plt.subplots()
        ax1.plot(
            self._episode_arr,
            avg_arr,
            color="blue",
            marker="o",
            linestyle="solid",
            linewidth=1,
            markersize=2,
            label="Average",
        )
        ax1.plot(
            self._episode_arr,
            max_arr,
            color="red",
            marker="o",
            linestyle="solid",
            linewidth=1,
            markersize=2,
            label="Maximum",
        )
        ax1.plot(
            self._episode_arr,
            min_arr,
            color="green",
            marker="o",
            linestyle="solid",
            linewidth=1,
            markersize=2,
            label="Minimum",
        )

        ax1.legend()
        ax1.grid(
            visible=True,
            which="major",
            axis="both",
            color="#909090",
            linestyle="dotted",
            linewidth=0.8,
        )

        ax1.set(xlabel="Episodes")
        ax1.set(ylabel="Time (msec)")
        ax1.set(title="Episode Times")

        fileName = os.path.sep.join([self._graph_dir, "episodes.svg"])
        plt.savefig(fileName)

    def _generate_wins(self):
        """
        Generate graph of report's last episode v/s the number of game, set and match wins of player and system in each report.
        """
        player_arr = np.empty(self._num_report, dtype=int)
        system_arr = np.empty(self._num_report, dtype=int)
        index = 0
        for report in self._report_list:
            player_arr[index] = report.player_game_win
            system_arr[index] = report.system_game_win
            index = index + 1

        _, ((ax1), (ax2), (ax3)) = plt.subplots(3, 1)

        ax1.plot(
            self._episode_arr,
            player_arr,
            color="blue",
            marker="o",
            linestyle="solid",
            linewidth=1,
            markersize=2,
            label="Player",
        )
        ax1.plot(
            self._episode_arr,
            system_arr,
            color="red",
            marker="o",
            linestyle="solid",
            linewidth=1,
            markersize=2,
            label="System",
        )

        ax1.legend()
        ax1.grid(
            visible=True,
            which="major",
            axis="both",
            color="#909090",
            linestyle="dotted",
            linewidth=0.8,
        )

        ax1.set(xlabel="Episodes")
        ax1.set(ylabel="Wins")
        ax1.set(title="Game Wins")

        index = 0
        for report in self._report_list:
            player_arr[index] = report.player_set_win
            system_arr[index] = report.system_set_win
            index = index + 1

        ax2.plot(
            self._episode_arr,
            player_arr,
            color="blue",
            marker="o",
            linestyle="solid",
            linewidth=1,
            markersize=2,
            label="Player",
        )
        ax2.plot(
            self._episode_arr,
            system_arr,
            color="red",
            marker="o",
            linestyle="solid",
            linewidth=1,
            markersize=2,
            label="System",
        )

        ax2.legend()
        ax2.grid(
            visible=True,
            which="major",
            axis="both",
            color="#909090",
            linestyle="dotted",
            linewidth=0.8,
        )

        ax2.set(xlabel="Episodes")
        ax2.set(ylabel="Wins")
        ax2.set(title="Set Wins")

        index = 0
        for report in self._report_list:
            player_arr[index] = report.player_match_win
            system_arr[index] = report.system_match_win
            index = index + 1

        ax3.plot(
            self._episode_arr,
            player_arr,
            color="blue",
            marker="o",
            linestyle="solid",
            linewidth=1,
            markersize=2,
            label="Player",
        )
        ax3.plot(
            self._episode_arr,
            system_arr,
            color="red",
            marker="o",
            linestyle="solid",
            linewidth=1,
            markersize=2,
            label="System",
        )

        ax3.legend()
        ax3.grid(
            visible=True,
            which="major",
            axis="both",
            color="#909090",
            linestyle="dotted",
            linewidth=0.8,
        )

        ax3.set(xlabel="Episodes")
        ax3.set(ylabel="Wins")
        ax3.set(title="Match Wins")

        fileName = os.path.sep.join([self._graph_dir, "wins.svg"])
        plt.savefig(fileName)

    def _generate_win_margins(self):
        """
        Generate graph of report's last episode v/s the average win margin for game, set and match for player and system in each report.
        """
        player_arr = np.empty(self._num_report, dtype=np.float32)
        system_arr = np.empty(self._num_report, dtype=np.float32)
        index = 0
        for report in self._report_list:
            player_arr[index] = safe_div(
                report.player_game_sumwinmargin, report.player_game_win
            )
            system_arr[index] = safe_div(
                report.system_game_sumwinmargin, report.system_game_win
            )
            index = index + 1

        _, ((ax1), (ax2), (ax3)) = plt.subplots(3, 1)

        ax1.plot(
            self._episode_arr,
            player_arr,
            color="blue",
            marker="o",
            linestyle="solid",
            linewidth=1,
            markersize=2,
            label="Player",
        )
        ax1.plot(
            self._episode_arr,
            system_arr,
            color="red",
            marker="o",
            linestyle="solid",
            linewidth=1,
            markersize=2,
            label="System",
        )

        ax1.legend()
        ax1.grid(
            visible=True,
            which="major",
            axis="both",
            color="#909090",
            linestyle="dotted",
            linewidth=0.8,
        )

        ax1.set(xlabel="Episodes")
        ax1.set(ylabel="Win Margin")
        ax1.set(title="Game Win Margins")

        index = 0
        for report in self._report_list:
            player_arr[index] = safe_div(
                report.player_set_sumwinmargin, report.player_set_win
            )
            system_arr[index] = safe_div(
                report.system_set_sumwinmargin, report.system_set_win
            )
            index = index + 1

        ax2.plot(
            self._episode_arr,
            player_arr,
            color="blue",
            marker="o",
            linestyle="solid",
            linewidth=1,
            markersize=2,
            label="Player",
        )
        ax2.plot(
            self._episode_arr,
            system_arr,
            color="red",
            marker="o",
            linestyle="solid",
            linewidth=1,
            markersize=2,
            label="System",
        )

        ax2.legend()
        ax2.grid(
            visible=True,
            which="major",
            axis="both",
            color="#909090",
            linestyle="dotted",
            linewidth=0.8,
        )

        ax2.set(xlabel="Episodes")
        ax2.set(ylabel="Win Margin")
        ax2.set(title="Set Win Margins")

        index = 0
        for report in self._report_list:
            player_arr[index] = safe_div(
                report.player_match_sumwinmargin, report.player_match_win
            )
            system_arr[index] = safe_div(
                report.system_match_sumwinmargin, report.system_match_win
            )
            index = index + 1

        ax3.plot(
            self._episode_arr,
            player_arr,
            color="blue",
            marker="o",
            linestyle="solid",
            linewidth=1,
            markersize=2,
            label="Player",
        )
        ax3.plot(
            self._episode_arr,
            system_arr,
            color="red",
            marker="o",
            linestyle="solid",
            linewidth=1,
            markersize=2,
            label="System",
        )

        ax3.legend()
        ax3.grid(
            visible=True,
            which="major",
            axis="both",
            color="#909090",
            linestyle="dotted",
            linewidth=0.8,
        )

        ax3.set(xlabel="Episodes")
        ax3.set(ylabel="Wins")
        ax3.set(title="Match Win Margins")

        fileName = os.path.sep.join([self._graph_dir, "winmargins.svg"])
        plt.savefig(fileName)


class QTableRenderer:
    _logger = logging.getLogger(LOGGER_BASE)
    _LOG_TAG = {"lgmod": "TNSTATS"}

    @staticmethod
    def create_custom_colormap(name, red, green, blue, reverse_map=False):
        num = 256
        vals = np.ones((num, 4))
        vals[:, 0] = np.linspace(red / 256, 1, num)
        vals[:, 1] = np.linspace(green / 256, 1, num)
        vals[:, 2] = np.linspace(blue / 256, 1, num)
        # 0 is lighter, 1 is
        cm = colors.ListedColormap(vals, name=name)
        if reverse_map:
            return cm.reversed(name=name)
        return cm

    COL_MAP1 = plt.get_cmap("Reds")  # Back, Front
    COL_MAP2 = plt.get_cmap("Blues")  # Left, Right
    COL_MAP3 = plt.get_cmap("Greens")  # Back-Right, Front-Left
    COL_MAP4 = plt.get_cmap("Greys")  # Back-Left,  Front-Right
    COL_MAP5 = create_custom_colormap(
        "Yellows", 240, 220, 0, reverse_map=True
    )  # No-Movement

    # Maps direction to color-map and whether direction points in positive direction.
    DIR_COLOR_MAP = {
        DiscreteTennis.DIR_FRONT: (COL_MAP1, True, COL_MAP1(1.0), " F"),
        DiscreteTennis.DIR_LEFT: (COL_MAP2, False, COL_MAP2(1.0), " L"),
        DiscreteTennis.DIR_BACK: (COL_MAP1, False, COL_MAP1(1.0), " B"),
        DiscreteTennis.DIR_RIGHT: (COL_MAP2, True, COL_MAP2(1.0), " R"),
        DiscreteTennis.DIR_FRONTLEFT: (COL_MAP3, True, COL_MAP3(1.0), "FL"),
        DiscreteTennis.DIR_BACKLEFT: (COL_MAP4, False, COL_MAP4(1.0), "BL"),
        DiscreteTennis.DIR_BACKRIGHT: (COL_MAP3, False, COL_MAP3(1.0), "BR"),
        DiscreteTennis.DIR_FRONTRIGHT: (COL_MAP4, True, COL_MAP4(1.0), "FR"),
        DiscreteTennis.DIR_NONE: (
            COL_MAP5,
            True,
            (139 / 256, 128 / 256, 0.0, 1.0),
            "NO",
        ),
    }

    DIR_COLOR_ORDER = [
        DiscreteTennis.DIR_BACK,
        DiscreteTennis.DIR_FRONT,
        DiscreteTennis.DIR_LEFT,
        DiscreteTennis.DIR_RIGHT,
        DiscreteTennis.DIR_BACKRIGHT,
        DiscreteTennis.DIR_FRONTLEFT,
        DiscreteTennis.DIR_BACKLEFT,
        DiscreteTennis.DIR_FRONTRIGHT,
        DiscreteTennis.DIR_NONE,
    ]

    GCELLPOS_TO_DIR = np.array(
        [
            [
                DiscreteTennis.DIR_FRONTRIGHT,
                DiscreteTennis.DIR_FRONT,
                DiscreteTennis.DIR_FRONTLEFT,
            ],  # Row-0 Bottom-most
            [
                DiscreteTennis.DIR_RIGHT,
                DiscreteTennis.DIR_NONE,
                DiscreteTennis.DIR_LEFT,
            ],  # Row-1
            [
                DiscreteTennis.DIR_BACKRIGHT,
                DiscreteTennis.DIR_BACK,
                DiscreteTennis.DIR_BACKLEFT,
            ],  # Row-2 Top-most
        ]
    )

    MODE_NEXT_DELTA = (1,)
    MODE_MAX_QVALUE = (2,)

    def __init__(self, qarr, mode=MODE_NEXT_DELTA):
        """
        Q-Table visualizer construction. The table is assumed to be of shape (court-row, court-column, drift-dir, action-dir)
        with  DiscreteTennis.DIR_MAX + 1 drift directions and action movements for each court position. The court-position
        rows and columns can be any numbers and typically depends on the cell-resolution that divides court into multiple
        discrete grid cells with 0-based positions at intersection of each grid line and origin centered at top-right corner
        showing a perspective relative to the human/agent-player, left & back directions towards the origin and right & front
        directions away from origin.

        :param qarr    Q-Table to be visualized.
        :param graph_file    Path of the Q-Table graph file where output is to be rendered.
        :param mode    Visualization mode - MODE_NEXT_DELTA shows offset to next highest q-value whereas
         MODE_MAX_QVALUE shows highest q-value for a court position's drift direction.
        """
        self._shape = qarr.shape
        self._rows = self._shape[0]
        self._cols = self._shape[1]
        self._qtable = qarr

        # self._fig, self._ax = plt.subplots()
        self._cell_x = 0.5
        self._cell_y = 0.5
        self._cell_valx = self._cell_x / 2
        self._cell_diry = self._cell_y / 6
        self._cell_valy = self._cell_y / 2
        self._cell_line = 0.25
        self._mode = mode

        self._width = self._cols * 3 * self._cell_x
        self._height = self._rows * 3 * self._cell_y

        self._qrange = None
        self._qtotal = None
        # Color value min-delta is established as percentage w.r.t. _qnormal_delta

    def plot_table(self, output_file, title=None):
        """
        Render the q-table to the current output file.
        """
        self._title = title
        std_dev = np.std(self._qtable, axis=None)
        std_dev = std_dev if std_dev != 0.0 else 0.1
        avg = np.average(self._qtable, axis=None)
        self._qrange = [avg - 2 * std_dev, avg + 2 * std_dev]
        self._qtotal = 4 * std_dev

        self._fig, self._ax = plt.subplots(figsize=(14, 15))
        try:
            self._ax.set_xlim(self._width)
            self._ax.set_ylim(self._height + 4 * self._cell_y)

            if self._logger.isEnabledFor(logging.INFO):
                self._logger.info(
                    "QTable Rendering qrange=[{}, {}], qtotal={}".format(
                        self._qrange[0], self._qrange[1], self._qtotal
                    ),
                    extra=self._LOG_TAG,
                )

            for pos_y in range(self._shape[0]):  # Rows
                for pos_x in range(self._shape[1]):  # Columns
                    # print("x={}, y={}".format(pos_x, pos_y))
                    self._plot_point(pos_x, pos_y, self._qtable[pos_y][pos_x])

            self._plot_key()
            self._ax.set_xlim(0, self._width + 1)
            self._ax.set_ylim(0, self._height + 3)
            plt.savefig(output_file)
            # plt.show()
        finally:
            plt.close(self._fig)

    def _plot_key(self):
        """
        Plot the legend key and additional q-table information.
        """
        self._width = self._cols * 3 * self._cell_x
        self._height = self._rows * 3 * self._cell_y

        for index in range(DiscreteTennis.DIR_MAX + 1):
            action_dir = QTableRenderer.DIR_COLOR_ORDER[index]
            kcell_x = index * self._cell_x
            kcell_y = self._height + (self._cell_y * 0.5)

            cm, _, full_color, dir_name = QTableRenderer.DIR_COLOR_MAP[action_dir]
            color = cm(0.5)
            kcell = patches.Rectangle(
                (kcell_x, kcell_y),
                self._cell_x,
                self._cell_y,
                facecolor=color,
                edgecolor="black",
                lw=self._cell_line,
            )
            self._ax.add_patch(kcell)

            rx, ry = kcell.get_xy()
            self._ax.annotate(
                dir_name,
                (rx + self._cell_valx, ry + self._cell_valy),
                color=full_color,
                weight="bold",
                fontsize=8,
                ha="center",
                va="center",
            )

            title = (
                self._title
                if self._title is not None
                else "Q-Table, Mode={}".format(
                    "Next-Delta"
                    if self._mode == QTableRenderer.MODE_NEXT_DELTA
                    else "Max-QVal"
                )
            )
            self._ax.annotate(
                title,
                (self._width / 2, ry + 1.5 * self._cell_valy),
                weight="bold",
                fontsize=11,
                ha="center",
                va="center",
            )

        # self._fig.text(0.5, 0.05, "Q-Table Mode={}".format("Next-Delta" if self._mode == QTableRenderer.MODE_NEXT_DELTA else "Max-QVal"), color="black",
        #    weight="bold", fontsize=11, ha="center", va="center")

    def _to_min_max(self):
        """
        Compute the q-value range of interest.
        """
        min_qval = -sys.maxsize + 1
        max_qval = sys.maxsize
        for x in range(self._shape[0]):
            for y in range(self._shape[1]):
                for drift_dir in range(self._shape[2]):
                    qaction_arr = self._qtable[x][y][drift_dir]
                    for action in range(self._shape[3]):
                        if qaction_arr[action] < min_qval:
                            min_qval = qaction_arr[action]
                        if qaction_arr[action] > max_qval:
                            max_qval = qaction_arr[action]
        self._qrange = [min_qval, max_qval]
        self._qmaxdelta = max_qval - min_qval

    def _to_max_qaction(self, action_arr):
        """
        Find the action with maximum q-value.

        :param action_arr Array with q-values for actions, indexed by move direction of action.
        :return Tuple (max-action-dir, min-delta, max-action-qvalue) where max-action-dir is the
        move direction that has the maximum action q-value, min-delta is the delta from next maximum
        action q-value, and max-action-qvalue is the maximum q-value for the action.
        """
        max_action = action_arr[0]
        max_action_dir = 0
        min_delta = sys.maxsize
        for action in range(1, self._shape[3]):
            curr_action = action_arr[action]
            action_delta = curr_action - max_action
            if action_delta > 0:
                max_action = curr_action
                max_action_dir = action
                min_delta = action_delta
            elif -action_delta < min_delta:
                min_delta = -action_delta
        return (max_action_dir, min_delta, max_action)

    def _plot_point(self, pt_x, pt_y, qpt):
        """
        Plot the view for a single court position.

        :param pt_x    X position in court.
        :param pt_y    Y position in court.
        :param qpt     Q-Table 2-d entry (drift-dir, action-dir) describing the q-values at the specified court position.
        """
        # bottom-left origin for this pt.
        pt_posx = self._width - (pt_x * 3 * self._cell_x) - (self._cell_x * 3)
        pt_posy = self._height - (pt_y * 3 * self._cell_y) - (self._cell_y * 3)

        color = QTableRenderer.COL_MAP2(0.5)
        rect_list = []
        debug_enabled = self._logger.isEnabledFor(logging.DEBUG)
        for pt_row in range(3):
            row_list = []
            rect_list.append(row_list)
            for pt_col in range(3):
                drift_dir = QTableRenderer.GCELLPOS_TO_DIR[pt_row][pt_col]
                color = (1.0, 1.0, 1.0, 1.0)
                val_color = (0.0, 0.0, 0.0, 1.0)
                actdir_name = None

                action_arr = qpt[drift_dir]
                max_action_dir, min_delta, max_action = self._to_max_qaction(action_arr)

                debug_message = None
                if drift_dir != max_action_dir:
                    cmap, _, val_color, actdir_name = QTableRenderer.DIR_COLOR_MAP[
                        max_action_dir
                    ]
                    qaction = max(
                        self._qrange[0], min(max_action, self._qrange[1])
                    )  # Clamp max_action between qrange[0] and qrange[1]
                    percent = (
                        1.0 - (qaction - self._qrange[0]) / self._qtotal
                    )  # Max q gets lightest color.
                    color = cmap(percent)

                    if debug_enabled:
                        debug_message = "Non At=({}, {}, {}): Action=({}, q={:.2f}, n={:.2f}), Color=({}, {:.3f}, {})".format(
                            pt_x,
                            pt_y,
                            DiscreteTennis.DIR_NAME_MAP[drift_dir][1],
                            actdir_name,
                            max_action,
                            min_delta,
                            cmap.name,
                            percent,
                            colors.rgb2hex(color, keep_alpha=True),
                        )
                else:
                    actdir_name = DiscreteTennis.DIR_NAME_MAP[max_action_dir][1]
                    if debug_enabled:
                        debug_message = "Opt At=({}, {}, {}): Action=({}, q={:.2f}, n={:.2f})".format(
                            pt_x,
                            pt_y,
                            DiscreteTennis.DIR_NAME_MAP[drift_dir][1],
                            actdir_name,
                            max_action,
                            min_delta,
                        )

                gcell_x = pt_posx + (pt_col * self._cell_x)
                gcell_y = pt_posy + (pt_row * self._cell_y)

                if debug_enabled:
                    debug_message = (
                        debug_message
                        + ", Layout=(pt=({}, {}), pos=({}, {}))".format(
                            pt_row, pt_col, gcell_x, gcell_y
                        )
                    )

                gcell = patches.Rectangle(
                    (gcell_x, gcell_y),
                    self._cell_x,
                    self._cell_y,
                    facecolor=color,
                    edgecolor="black",
                    lw=self._cell_line,
                )
                self._ax.add_patch(gcell)

                rx, ry = gcell.get_xy()
                gcell_val = (
                    min_delta if self._mode == self.MODE_NEXT_DELTA else max_action
                )
                if pt_row == 1 and pt_col == 1:
                    self._ax.annotate(
                        "{},{}".format(pt_x, pt_y),
                        (rx + self._cell_valx, ry + self._cell_valy),
                        color="black",
                        weight="bold",
                        fontsize=8,
                        ha="center",
                        va="center",
                    )
                    self._ax.annotate(
                        "{}: {:.2f}".format(actdir_name, gcell_val),
                        (rx + self._cell_valx, ry + self._cell_diry),
                        color=val_color,
                        fontsize=6,
                        ha="center",
                        va="center",
                    )
                else:
                    if drift_dir != max_action_dir:
                        self._ax.annotate(
                            actdir_name,
                            (rx + self._cell_valx, ry + self._cell_diry),
                            color=val_color,
                            weight="bold",
                            fontsize=6,
                            ha="center",
                            va="center",
                        )
                    self._ax.annotate(
                        "{:.2f}".format(gcell_val),
                        (rx + self._cell_valx, ry + self._cell_valy),
                        color=val_color,
                        fontsize=6,
                        ha="center",
                        va="center",
                    )
                row_list.append(gcell)

                if debug_enabled:
                    self._logger.debug(debug_message, extra=self._LOG_TAG)

        pt_cell = patches.Rectangle(
            (pt_posx, pt_posy),
            (self._cell_x * 3),
            self._cell_y * 3,
            alpha=1,
            facecolor="none",
            edgecolor="black",
            lw=2,
        )
        self._ax.add_patch(pt_cell)

    @staticmethod
    def show_color_map(cm):
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))

        fig, ax = plt.subplots(figsize=(10, 2))
        fig.subplots_adjust(top=0.5, bottom=0.4)
        ax.imshow(gradient, aspect="auto", cmap=cm)
        ax.set(title="{} Colormap".format(cm.name))
        plt.show()

    @staticmethod
    def show_color_map_tiles(cm, num_tiles):
        cell_x = 0.5
        cell_y = 0.5

        # _, ax = plt.subplots(figsize=(cell_y * num_tiles * 1.2, 2))
        xlim = cell_x * num_tiles + 1
        _, ax = plt.subplots(figsize=(xlim, 2))
        ax.set_xlim([0, xlim])
        percent = 1.0 / num_tiles
        for col in range(num_tiles):
            pos_x = col * cell_y
            print("{}, {}, {}".format(0, col, pos_x))
            color_percent = percent * col
            color = cm(color_percent)
            cell = patches.Rectangle(
                (pos_x, 0), cell_x, cell_y, facecolor=color, edgecolor="black", lw=1
            )
            ax.add_patch(cell)

            rx, ry = cell.get_xy()
            ax.annotate(
                "{:.2f}".format(color_percent),
                (rx + cell_x / 2, ry + cell_y / 2),
                color="black",
                weight="bold",
                fontsize=6,
                ha="center",
                va="center",
            )

        plt.show()


def main(argv):
    tc_runtime_configure({"logConfig": tc_closest_filepath(__file__, "logConfig.json")})
    logger = logging.getLogger(LOGGER_BASE)
    # QTableRenderer.show_color_map_tiles(QTableRenderer.create_custom_colormap("Yellows", 240, 220, 0, reverse_map=True), 20)
    # cmap = plt.get_cmap("Reds")
    # rgba = cmap(0.5)
    # print(rgba)
    np.random.seed(20)
    # qtable = np.array([[1, 2, 3],[3, 2.5, 1.5]])
    qtable = np.random.rand(
        2, 3, DiscreteTennis.DIR_MAX + 1, DiscreteTennis.DIR_MAX + 1
    )
    if logger.isEnabledFor(logging.INFO):
        logger.info("Qtable Shape={}".format(qtable.shape))

    np.set_printoptions(precision=2)
    qtableRenderer = QTableRenderer(qtable, mode=QTableRenderer.MODE_MAX_QVALUE)
    qtableRenderer.plot_table("/tmp/statsTest.svg")

    if logger.isEnabledFor(logging.INFO):
        logger.info("Qtable=" + SarsaTennisAgent.tostr_qtable(qtable))


if __name__ == "__main__":
    main(sys.argv)
