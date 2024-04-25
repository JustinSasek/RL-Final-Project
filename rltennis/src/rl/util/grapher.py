from dataclasses import dataclass

import pandas as pf
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from glob import glob


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
        
    def add_series(self, data:list[pd.DataFrame], label:str):
        data = [df.copy() for df in data]
        for df in data:
            df['Returns'] = df['Returns'].rolling(window=self.moving_avg_window).mean()
            df["Episode"] = df.index
            
        df = pd.concat(data, ignore_index=True)
        sns.lineplot(x="Episode", y='Returns', data=df, label=label, errorbar=('ci', 95))
        
    def show(self):
        plt.show()

if __name__ == "__main__":
    g = Graph("Tennis")
    paths = glob("rltennis/data/rl/output/results/REINFORCE/easy_seq_len2_N2_d32_lr0.001_pi0.3/**/*.csv", recursive=True)
    data = [pd.read_csv(path) for path in paths]
    g.add_series(data, "REINFORCE w/ mem")
    paths = glob("rltennis/data/rl/output/results/REINFORCE/easy_seq_len1_N2_d32_lr0.001_pi0.3/**/*.csv", recursive=True)
    data = [pd.read_csv(path) for path in paths]
    g.add_series(data, "REINFORCE no mem")
    paths = glob("rltennis/data/rl/output/results/REINFORCE Memory/**/*.csv", recursive=True)
    data = [pd.read_csv(path) for path in paths]
    g.add_series(data, "REINFORCE transformer")
    g.show()