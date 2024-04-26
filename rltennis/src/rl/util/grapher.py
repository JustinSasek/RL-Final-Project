from dataclasses import dataclass
from glob import glob

import pandas as pf
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm


@dataclass()
class Graph:
    title: str
    W: int = 8
    H: int = 6
    moving_avg_window: int = 100

    def __post_init__(self):
        plt.figure(figsize=(self.W, self.H))
        plt.title(self.title)
        plt.xlabel("Episode")
        plt.ylabel("Return")

    def add_series(self, data: list[pd.DataFrame], label: str):
        # data = [df.copy() for df in data]
        data = [df.copy() for df in data if len(df) == 10000]
        if not data:
            return
        for df in data:
            df["Returns"] = df["Returns"].rolling(window=self.moving_avg_window).mean()
            df["Episode"] = df.index

        df = pd.concat(data, ignore_index=True)
        sns.lineplot(
            x="Episode", y="Returns", data=df, label=label, errorbar=("ci", 95)
        )

    def show(self):
        plt.show()


if __name__ == "__main__":
    g = Graph("Tennis")
    # selection for hyper params
    # series = glob("rltennis/data/rl/output/results/REINFORCE/easy_seq_len1*", recursive=True)
    # series = glob("rltennis/data/rl/output/results/REINFORCE/easy_seq_len2*", recursive=True)
    series = glob(
        "rltennis/data/rl/output/results/REINFORCE Memory/easy*", recursive=True
    )
    labels = ["/".join(s.split("/")[-2:]) for s in series]
    path_lists = [glob(f"{s}/*.csv") for s in series]
    data = [[pd.read_csv(p) for p in path_list] for path_list in path_lists]
    for d, l in tqdm(zip(data, labels), colour="green"):
        g.add_series(d, l)
    g.show()
