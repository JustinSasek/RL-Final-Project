from dataclasses import dataclass
from glob import glob
from typing import Optional as Op

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
        plt.xlabel("Match (Episode)")
        plt.ylabel("Hits per Match (Episode)")
        plt.grid()

    def add_series(self, data: list[pd.DataFrame], label: str, max_len: Op[int] = None):
        data = [df.copy() for df in data]
        # data = [df.copy() for df in data if len(df) == 10000]
        if not data:
            return
        for i, df in enumerate(data):
            df = df.iloc[:max_len] if max_len else df
            df["Returns"] = df["Returns"].rolling(window=self.moving_avg_window).mean()
            df["Episode"] = df.index
            data[i] = df

        df = pd.concat(data, ignore_index=True)
        sns.lineplot(
            x="Episode", y="Returns", data=df, label=label, errorbar=("ci", 95)
        )

    def show(self):
        # plt.show()
        pass


# REINFORCE without Memory ############################################################################################################
g = Graph(
    "Hits Per Match vs Number of Hidden Layers for REINFORCE without Memory"
)
series = glob(
    "rltennis/data/rl/output/results/REINFORCE/easy_seq_len1_N*_d16_lr0.01_pi0.5",
    recursive=True,
)
series = sorted(series, key=lambda x: int(x.split("_N")[-1].split("_")[0]))
labels = ["N = " + str(int(s.split("_N")[-1].split("_")[0])) for s in series]
path_list = [glob(f"{s}/*.csv")[0] for s in series]
data = [pd.read_csv(p) for p in path_list]
for d, l in tqdm(zip(data, labels), colour="green"):
    g.add_series([d], l)
plt.savefig("rltennis/data/rl/output/results/Figures/REINFORCE_nomem_N.png")
g.show()


g = Graph("Hits Per Match vs Hidden Layer Size for REINFORCE without Memory")
series = glob(
    "rltennis/data/rl/output/results/REINFORCE/easy_seq_len1_N1_d*_lr0.01_pi0.5",
    recursive=True,
)
series = sorted(series, key=lambda x: int(x.split("_d")[-1].split("_")[0]))
labels = ["d = " + str(int(s.split("_d")[-1].split("_")[0])) for s in series]
path_list = [glob(f"{s}/*.csv")[0] for s in series]
data = [pd.read_csv(p) for p in path_list]
for d, l in tqdm(zip(data, labels), colour="green"):
    g.add_series([d], l)
plt.savefig("rltennis/data/rl/output/results/Figures/REINFORCE_nomem_d.png")
g.show()


g = Graph("Hits Per Match vs Learning Rate for REINFORCE without Memory")
series = glob(
    "rltennis/data/rl/output/results/REINFORCE/easy_seq_len1_N1_d16_lr*_pi0.5",
    recursive=True,
)
series = sorted(series, key=lambda x: float(x.split("_lr")[-1].split("_")[0]))
labels = ["lr = " + str(float(s.split("_lr")[-1].split("_")[0])) for s in series]
path_list = [glob(f"{s}/*.csv")[0] for s in series]
data = [pd.read_csv(p) for p in path_list]
for d, l in tqdm(zip(data, labels), colour="green"):
    g.add_series([d], l)
plt.savefig("rltennis/data/rl/output/results/Figures/REINFORCE_nomem_lr.png")
g.show()


g = Graph("Hits Per Match vs Policy Loss Scale for REINFORCE without Memory")
series = glob(
    "rltennis/data/rl/output/results/REINFORCE/easy_seq_len1_N1_d16_lr0.01_pi*",
    recursive=True,
)
series = sorted(series, key=lambda x: float(x.split("_pi")[-1].split("_")[0]))
labels = ["pi = " + str(float(s.split("_pi")[-1].split("_")[0])) for s in series]
path_list = [glob(f"{s}/*.csv")[0] for s in series]
data = [pd.read_csv(p) for p in path_list]
for d, l in tqdm(zip(data, labels), colour="green"):
    g.add_series([d], l)
plt.savefig("rltennis/data/rl/output/results/Figures/REINFORCE_nomem_pi.png")
g.show()


# REINFORCE with Memory ############################################################################################################
g = Graph(
    "Hits Per Match vs Number of Hidden Layers for REINFORCE with Memory"
)
series = glob(
    "rltennis/data/rl/output/results/REINFORCE/easy_seq_len2_N*_d16_lr0.01_pi0.5",
    recursive=True,
)
series = sorted(series, key=lambda x: int(x.split("_N")[-1].split("_")[0]))
labels = ["N = " + str(int(s.split("_N")[-1].split("_")[0])) for s in series]
path_list = [glob(f"{s}/*.csv")[0] for s in series]
data = [pd.read_csv(p) for p in path_list]
for d, l in tqdm(zip(data, labels), colour="green"):
    g.add_series([d], l)
plt.savefig("rltennis/data/rl/output/results/Figures/REINFORCE_mem_N.png")
g.show()


g = Graph("Hits Per Match vs Hidden Layer Size for REINFORCE with Memory")
series = glob(
    "rltennis/data/rl/output/results/REINFORCE/easy_seq_len2_N1_d*_lr0.01_pi0.5",
    recursive=True,
)
series = sorted(series, key=lambda x: int(x.split("_d")[-1].split("_")[0]))
labels = ["d = " + str(int(s.split("_d")[-1].split("_")[0])) for s in series]
path_list = [glob(f"{s}/*.csv")[0] for s in series]
data = [pd.read_csv(p) for p in path_list]
for d, l in tqdm(zip(data, labels), colour="green"):
    g.add_series([d], l)
plt.savefig("rltennis/data/rl/output/results/Figures/REINFORCE_mem_d.png")
g.show()


g = Graph("Hits Per Match vs Learning Rate for REINFORCE with Memory")
series = glob(
    "rltennis/data/rl/output/results/REINFORCE/easy_seq_len2_N1_d16_lr*_pi0.5",
    recursive=True,
)
series = sorted(series, key=lambda x: float(x.split("_lr")[-1].split("_")[0]))
labels = ["lr = " + str(float(s.split("_lr")[-1].split("_")[0])) for s in series]
path_list = [glob(f"{s}/*.csv")[0] for s in series]
data = [pd.read_csv(p) for p in path_list]
for d, l in tqdm(zip(data, labels), colour="green"):
    g.add_series([d], l)
plt.savefig("rltennis/data/rl/output/results/Figures/REINFORCE_mem_lr.png")
g.show()


g = Graph("Hits Per Match vs Policy Loss Scale for REINFORCE with Memory")
series = glob(
    "rltennis/data/rl/output/results/REINFORCE/easy_seq_len2_N1_d16_lr0.01_pi*",
    recursive=True,
)
series = sorted(series, key=lambda x: float(x.split("_pi")[-1].split("_")[0]))
labels = ["pi = " + str(float(s.split("_pi")[-1].split("_")[0])) for s in series]
path_list = [glob(f"{s}/*.csv")[0] for s in series]
data = [pd.read_csv(p) for p in path_list]
for d, l in tqdm(zip(data, labels), colour="green"):
    g.add_series([d], l)
plt.savefig("rltennis/data/rl/output/results/Figures/REINFORCE_mem_pi.png")
g.show()

# REINFORCE Transformer ############################################################################################################
g = Graph(
    "Hits Per Match vs Number of Hidden Layers for REINFORCE with Transformer"
)
series = glob(
    "rltennis/data/rl/output/results/REINFORCE Memory/easy_seq_len2_N*_d16_lr0.01_pi0.3_l20.0001",
    recursive=True,
)
series = sorted(series, key=lambda x: int(x.split("_N")[-1].split("_")[0]))
labels = ["N = " + str(int(s.split("_N")[-1].split("_")[0])) for s in series]
path_list = [glob(f"{s}/*.csv")[0] for s in series]
data = [pd.read_csv(p) for p in path_list]
for d, l in tqdm(zip(data, labels), colour="green"):
    g.add_series([d], l)
plt.savefig("rltennis/data/rl/output/results/Figures/REINFORCE_transformer_N.png")
g.show()


g = Graph("Hits Per Match vs Hidden Layer Size for REINFORCE with Transformer")
series = glob(
    "rltennis/data/rl/output/results/REINFORCE Memory/easy_seq_len2_N1_d*_lr0.01_pi0.3_l20.0001",
    recursive=True,
)
series = sorted(series, key=lambda x: int(x.split("_d")[-1].split("_")[0]))
labels = ["d = " + str(int(s.split("_d")[-1].split("_")[0])) for s in series]
path_list = [glob(f"{s}/*.csv")[0] for s in series]
data = [pd.read_csv(p) for p in path_list]
for d, l in tqdm(zip(data, labels), colour="green"):
    g.add_series([d], l)
plt.savefig("rltennis/data/rl/output/results/Figures/REINFORCE_transformer_d.png")
g.show()


g = Graph("Hits Per Match vs Learning Rate for REINFORCE with Transformer")
series = glob(
    "rltennis/data/rl/output/results/REINFORCE Memory/easy_seq_len2_N1_d16_lr*_pi0.3_l20.0001",
    recursive=True,
)
series = sorted(series, key=lambda x: float(x.split("_lr")[-1].split("_")[0]))
labels = ["lr = " + str(float(s.split("_lr")[-1].split("_")[0])) for s in series]
path_list = [glob(f"{s}/*.csv")[0] for s in series]
data = [pd.read_csv(p) for p in path_list]
for d, l in tqdm(zip(data, labels), colour="green"):
    g.add_series([d], l)
plt.savefig("rltennis/data/rl/output/results/Figures/REINFORCE_transformer_lr.png")
g.show()


g = Graph("Hits Per Match vs Policy Loss Scale for REINFORCE with Transformer")
series = glob(
    "rltennis/data/rl/output/results/REINFORCE Memory/easy_seq_len2_N1_d16_lr0.01_pi0*_l20.0001",
    recursive=True,
)
series = sorted(series, key=lambda x: float(x.split("_pi")[-1].split("_")[0]))
labels = ["pi = " + str(float(s.split("_pi")[-1].split("_")[0])) for s in series]
path_list = [glob(f"{s}/*.csv")[0] for s in series]
data = [pd.read_csv(p) for p in path_list]
for d, l in tqdm(zip(data, labels), colour="green"):
    g.add_series([d], l)
plt.savefig("rltennis/data/rl/output/results/Figures/REINFORCE_transformer_pi.png")
g.show()

g = Graph("Hits Per Match vs Attention Normalization for REINFORCE with Transformer")
series = glob(
    "rltennis/data/rl/output/results/REINFORCE Memory/easy_seq_len2_N1_d16_lr0.01_pi0.3_l2*",
    recursive=True,
)
series = sorted(series, key=lambda x: float(x.split("_l2")[-1].split("_")[0]))
labels = ["l2 = " + str(float(s.split("_l2")[-1].split("_")[0])) for s in series]
path_list = [glob(f"{s}/*.csv")[0] for s in series]
data = [pd.read_csv(p) for p in path_list]
for d, l in tqdm(zip(data, labels), colour="green"):
    g.add_series([d], l)
plt.savefig("rltennis/data/rl/output/results/Figures/REINFORCE_transformer_l2.png")
g.show()

# Comparisons

g = Graph("Hits Per Match vs Algorithm for Easy Difficulty")
series = [
    (
        "rltennis/data/rl/output/results/SARSA Tabular/easy_tabular_sarsa",
        "Tabular Sarsa",
    ),
    ("rltennis/data/rl/output/results/SARSA NN/easy", "MLP Sarsa"),
    (
        "rltennis/data/rl/output/results/REINFORCE/easy_seq_len2_N2_d32_lr0.01_pi0.5",
        "MLP Reinforce With Memory",
    ),
    (
        "rltennis/data/rl/output/results/REINFORCE/easy_seq_len1_N2_d32_lr0.02_pi0.7",
        "MLP Reinforce Without Memory",
    ),
    (
        "rltennis/data/rl/output/results/REINFORCE Memory/easy_seq_len2_N1_d16_lr0.01_pi0.3_l20.001",
        "Transformer Reinforce With Memory",
    ),
]
series_paths = [s for s, _ in series]
labels = [l for _, l in series]
path_lists = [glob(f"{s}/*.csv") for s in series_paths]
data = [[pd.read_csv(p) for p in path_list] for path_list in path_lists]
for d, l in tqdm(zip(data, labels), colour="green"):
    g.add_series(d, l, 5000)
plt.savefig("rltennis/data/rl/output/results/Figures/Easy Comparison.png")
g.show()


g = Graph("Hits Per Match vs Algorithm for Hard Difficulty")
series = [
    (
        "rltennis/data/rl/output/results/SARSA Tabular/hard_tabular_sarsa",
        "Tabular Sarsa",
    ),
    ("rltennis/data/rl/output/results/SARSA NN/hard", "MLP Sarsa"),
    (
        "rltennis/data/rl/output/results/REINFORCE/hard_seq_len1_N2_d32_lr0.02_pi0.7",
        "MLP Reinforce Without Memory",
    ),
    (
        "rltennis/data/rl/output/results/REINFORCE/hard_seq_len2_N2_d32_lr0.01_pi0.5",
        "MLP Reinforce With Memory",
    ),
    (
        "rltennis/data/rl/output/results/REINFORCE Memory/hard_seq_len2_N1_d16_lr0.01_pi0.3_l20.001",
        "Transformer Reinforce With Memory",
    ),
]
series_paths = [s for s, _ in series]
labels = [l for _, l in series]
path_lists = [glob(f"{s}/*.csv") for s in series_paths]
data = [[pd.read_csv(p) for p in path_list] for path_list in path_lists]
for d, l in tqdm(zip(data, labels), colour="green"):
    g.add_series(d, l, 5000)
plt.savefig("rltennis/data/rl/output/results/Figures/Hard Comparison.png")
g.show()
