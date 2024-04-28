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
        data = [df.copy() for df in data]
        # data = [df.copy() for df in data if len(df) == 10000]
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
        
        
# Temp
        
g = Graph(
    "Tennis Return Curves vs Number of Hidden Layers for REINFORCE with Transformer"
)
series = glob(
    "rltennis/data/rl/output/results/*/hard*",
    recursive=True,
)
series = sorted(series, key=lambda x: int(x.split("_N")[-1].split("_")[0]))
labels = [l.split("/")[-1] for l in series]
path_lists = [glob(f"{s}/*.csv") for s in series]
data = [[pd.read_csv(p) for p in path_list] for path_list in path_lists]
for d, l in tqdm(zip(data, labels), colour="green"):
    g.add_series(d, l)
# plt.savefig("rltennis/data/rl/output/results/Figures/REINFORCE_transformer_N.png")
g.show()
        

# REINFORCE without Memory ############################################################################################################
g = Graph(
    "Tennis Return Curves vs Number of Hidden Layers for REINFORCE without Memory"
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


g = Graph("Tennis Return Curves vs Hidden Layer Size for REINFORCE without Memory")
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


g = Graph("Tennis Return Curves vs Learning Rate for REINFORCE without Memory")
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


g = Graph("Tennis Return Curves vs Policy Loss Scale for REINFORCE without Memory")
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
    "Tennis Return Curves vs Number of Hidden Layers for REINFORCE with Memory"
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


g = Graph("Tennis Return Curves vs Hidden Layer Size for REINFORCE with Memory")
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


g = Graph("Tennis Return Curves vs Learning Rate for REINFORCE with Memory")
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


g = Graph("Tennis Return Curves vs Policy Loss Scale for REINFORCE with Memory")
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
    "Tennis Return Curves vs Number of Hidden Layers for REINFORCE with Transformer"
)
series = glob(
    "rltennis/data/rl/output/results/Dropout/REINFORCE Memory/easy_seq_len2_N*_d16_lr0.0005_pi0.3",
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


g = Graph("Tennis Return Curves vs Hidden Layer Size for REINFORCE with Transformer")
series = glob(
    "rltennis/data/rl/output/results/Dropout/REINFORCE Memory/easy_seq_len2_N1_d*_lr0.0005_pi0.3",
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


g = Graph("Tennis Return Curves vs Learning Rate for REINFORCE with Transformer")
series = glob(
    "rltennis/data/rl/output/results/Dropout/REINFORCE Memory/easy_seq_len2_N1_d16_lr*_pi0.3",
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


g = Graph("Tennis Return Curves vs Policy Loss Scale for REINFORCE with Transformer")
series = glob(
    "rltennis/data/rl/output/results/Dropout/REINFORCE Memory/easy_seq_len2_N1_d16_lr0.0005_pi0*",
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


