import os
import random
import signal
from collections import Counter
from dataclasses import dataclass
from os.path import join
from typing import Optional as Op

import gym
import numpy as np
import seaborn as sns  # type: ignore
import torch
from matplotlib import pyplot as plt
from rl.models import SingleVectorWrapper, Transformer
from rl.tennis.discreteTennisWrappers import *
from torch import nn
from tqdm import tqdm  # type: ignore

STATE_FILTER = [1, 0, 6, 5, 4, 17]
# STATE_FILTER = range(18)
RESULTS_PATH = "rltennis/data/rl/output/results"


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def already_ran(model_name: str, run_name: str, seed: int) -> bool:
    path = join(RESULTS_PATH, model_name, run_name, f"seed{seed}.csv")
    return os.path.exists(path)


def save_returns(returns: list[float], model_name: str, run_name: str, seed: int):
    path = join(RESULTS_PATH, model_name, run_name, f"seed{seed}.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("Returns\n")
        f.write("\n".join(map(str, returns)))


class MemoryMLP(nn.Module):
    def __init__(self, input_size, seq_len, output_size, N, d_model):
        super().__init__()
        self.seq_len = seq_len
        self.mods = nn.ModuleList()
        self.mods.add_module("fc0", nn.Linear(input_size * seq_len, d_model))
        self.mods.add_module("relu0", nn.LeakyReLU())
        for i in range(1, N):
            self.mods.add_module(f"fc{i}", nn.Linear(d_model, d_model))
            self.mods.add_module(f"relu{i}", nn.LeakyReLU())
        self.mods.add_module("fc_out", nn.Linear(d_model, output_size * seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] != self.seq_len:
            missing = self.seq_len - x.shape[-2]
            x = torch.cat([torch.zeros_like(x[..., :missing, :]), x], dim=-2)
        x = x.view(*x.shape[:-2], -1)
        for mod in self.mods:
            x = mod(x)
        x = x.view(*x.shape[:-1], self.seq_len, -1)
        return x


@dataclass
class Agent:
    state_size: int
    action_size: int
    seq_len: int
    epilison: float
    discount_rate: float
    lr: float
    MLP: bool
    has_mem: bool
    pi_loss_scale: float
    d_model: Op[int] = None
    N: int = 1
    h: Op[int] = None
    norm: float = 1e-4

    def __post_init__(self):
        input_size = (self.state_size + 3) if self.has_mem else self.state_size
        if self.MLP:
            self.model = MemoryMLP(
                input_size, self.seq_len, self.action_size + 1, self.N, self.d_model
            )
        else:
            transformer = Transformer(
                N=self.N,
                d_model=self.d_model,
                d_ff=self.d_model * 2,
                h=self.h,
                max_len=self.seq_len,
                dropout=0.0,
            )
            self.model = SingleVectorWrapper(
                transformer=transformer,
                input_size=input_size,  # action, reward, done, state
                output_size=self.action_size + 1,  # value, action1, action2, ...
                dropout=0.0,
            )
        # optimistic initialization
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = None
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 2, 0.5)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 10)

    def __call__(self, hist: list[list[float]]) -> int:
        # if random.random() < self.epilison:
        #     return random.randint(0, self.action_size - 1)
        hist_tensor = torch.tensor(hist, dtype=torch.float32)
        hist_tensor = hist_tensor[..., -self.seq_len :, :]
        if not self.has_mem:
            hist_tensor = hist_tensor[..., 3:]
        output: torch.Tensor = self.model(hist_tensor)[-1]  # only use last timestep
        pis: torch.Tensor = output[..., 1:]
        pis = nn.functional.softmax(pis, dim=-1)
        selected_actions = int(torch.multinomial(pis, num_samples=1).squeeze(-1))
        return selected_actions

    def process_hist(
        self,
        hist: list[list[float]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Finds returns and aligns tensors for training."""
        hist_tensor = torch.tensor(hist, dtype=torch.float32)
        returns = hist_tensor[..., 1].clone()
        actions = hist_tensor[..., 0].to(torch.int64)

        for i in range(len(hist) - 2, -1, -1):
            if hist_tensor[i, 2] == 0:
                returns[i] += self.discount_rate * returns[i + 1]

        # shift tensors
        # (a, r, done', s'),  a' -> R'
        returns = returns[..., 1:]
        hist_tensor = hist_tensor[..., :-1, :]
        actions = actions[..., 1:]
        return hist_tensor, actions, returns

    def split_seq(
        self, hist_tensor: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Splits the history tensor into sequences of length seq_len."""
        split_hist = []
        split_actions = []
        split_returns = []
        for i in range(hist_tensor.shape[0] - self.seq_len + 1):
            split_hist.append(hist_tensor[i : i + self.seq_len])
            split_actions.append(actions[i : i + self.seq_len])
            split_returns.append(returns[i : i + self.seq_len])
        if len(split_hist) == 0:
            split_hist.append(hist_tensor)
            split_actions.append(actions)
            split_returns.append(returns)
        return split_hist, split_actions, split_returns

    def update(self, hists: list[list[list[float]]]):
        hists_list = []
        actions_list = []
        returns_list = []
        for hist in hists:
            hist_tensor, actions, returns = self.process_hist(hist)
            split_hist, split_actions, split_returns = self.split_seq(
                hist_tensor, actions, returns
            )
            hists_list += split_hist
            actions_list += split_actions
            returns_list += split_returns

        hists_tensor = torch.stack(hists_list)
        actions = torch.stack(actions_list)[..., -1]  # only use last timestep
        returns = torch.stack(returns_list)[..., -1]  # only use last timestep

        if not self.has_mem:
            hists_tensor = hists_tensor[..., 3:]

        outputs: torch.Tensor = self.model(hists_tensor)
        outputs = outputs[..., -1, :]  # only use last timestep
        values = outputs[..., 0]
        pis = outputs[..., 1:]
        pis = nn.functional.softmax(pis, dim=-1)

        value_loss = nn.functional.mse_loss(values, returns)
        pi_taken = torch.gather(pis, -1, actions.unsqueeze(-1)).squeeze(-1)
        deltas = returns - values
        deltas = deltas.detach()
        pi_loss = -torch.mean(deltas * torch.log(pi_taken))
        loss = (1 - self.pi_loss_scale) * value_loss + self.pi_loss_scale * pi_loss

        if not self.MLP:
            for n in range(self.N):
                attn = self.model.transformer.get_attn(n)
                attn_loss = torch.norm(attn, p=2) * self.norm
                loss += attn_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        # for modules in self.model.modules():
        #     for param in modules.parameters():
        #         if param.grad is not None and param.grad.data is not None:
        #             param.grad.data[param.grad.data == float("-inf")] = -1
        #             param.grad.data[param.grad.data == float("inf")] = 1
        #             param.grad.data[param.grad.data == float("nan")] = 0

        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

        for modules in self.model.modules():
            if any(torch.isnan(param).any() for param in modules.parameters()):
                raise ValueError("Model weights have NaN values")

        return value_loss.item(), pi_loss.item()


@dataclass
class Experiment:
    env: gym.Env
    seq_len: int = 2
    discount_rate: float = 0.99
    epilison: float = 0.0
    batch_size: int = 4
    horizon: Op[int] = None
    has_mem: bool = True
    lr: float = 1e-3
    MLP: bool = False
    pi_loss_scale: float = 0.5
    N: int = 1
    d_model: int = 16
    h: int = 2
    norm: float = 1e-4

    def __post_init__(self):
        self.pi = Agent(
            state_size=len(STATE_FILTER),
            action_size=self.env.action_space.n,  # type: ignore
            seq_len=self.seq_len,
            epilison=self.epilison,
            discount_rate=self.discount_rate,
            lr=self.lr,
            MLP=self.MLP,
            has_mem=self.has_mem,
            pi_loss_scale=self.pi_loss_scale,
            N=self.N,
            d_model=self.d_model,
            h=self.h,
            norm=self.norm,
        )

    def run(
        self,
        n_episodes: int,
        update: bool,
        starting_epsilon: float,
        verbose: bool = False,
        callback=None,
    ) -> tuple[list[float], list[tuple[float, float]]]:

        # Collect data, train after collecting batch_size episodes
        hists = []
        returns = []
        losses = []
        self.pi.epilison = starting_epsilon
        c: Counter = Counter()
        for ep in tqdm(range(n_episodes), colour="green"):
            # for ep in range(n_episodes):
            if (ep + 1) % (n_episodes // 4) == 0:
                self.pi.epilison /= 2
                pass
            s, _ = self.env.reset()
            self.env.render()
            s = s[STATE_FILTER]
            hist = [[0, 0.0, False, *s]]  # a, r, done, s
            ep_return = 0.0
            t = 0
            while True:
                a = self.pi(hist)
                c[a] += 1
                s, r, done, term, *_ = self.env.step(a)
                s = s[STATE_FILTER]
                self.env.render()
                ep_return += r
                done = done or term
                game_done = s[-1]  # as opposed to episode done
                hist.append([a, r, game_done, *s])
                t += 1
                if done or (self.horizon and t >= self.horizon):
                    hists.append(hist)
                    returns.append(ep_return)
                    if callback:
                        callback(returns)
                    break
            if len(hists) >= self.batch_size:
                if update:
                    if verbose:
                        print(c)
                        c: Counter = Counter()  # type: ignore
                    losses.append(self.pi.update(hists))
                hists = []
        return returns, losses

    def train(
        self, n_episodes: int, verbose: bool = False, callback=None
    ) -> tuple[list[float], list[tuple[float, float]]]:
        self.env._render_view = False
        return self.run(n_episodes, True, self.epilison, verbose, callback)

    def eval(
        self, n_episodes: int, verbose: bool = False
    ) -> tuple[list[float], list[tuple[float, float]]]:
        self.env._render_view = True
        return self.run(n_episodes, False, 0.0, verbose)

    def save(self, path: str):
        torch.save(self.pi.model.state_dict(), path)

    def load(self, path: str):
        self.pi.model.load_state_dict(torch.load(path))

    @staticmethod
    def visualize(returns: list[float]):
        sns.lineplot(x=range(len(returns)), y=returns)
        plt.xlabel("Episode")
        plt.show()


if __name__ == "__main__":
    # find best LR
    # model name, env hard?, seq_len, has_mem, MLP, N, d_model, lr, pi_loss_scale
    things_to_try1: list[tuple[str, bool, int, bool, bool, int, int, float, float]] = [
        ("REINFORCE", False, 1, False, True, 1, 16, 1e-3, 0.3),
        ("REINFORCE", False, 1, False, True, 1, 16, 5e-3, 0.3),
        ("REINFORCE", False, 1, False, True, 1, 16, 1e-2, 0.3),
        ("REINFORCE", False, 1, False, True, 1, 16, 5e-2, 0.3),
        #
        ("REINFORCE Memory", False, 2, True, False, 1, 16, 1e-4, 0.3),
        ("REINFORCE Memory", False, 2, True, False, 1, 16, 5e-4, 0.3),
        ("REINFORCE Memory", False, 2, True, False, 1, 16, 1e-3, 0.3),
    ]

    # model name, env hard?, seq_len, has_mem, MLP, N, d_model, lr, pi_loss_scale, norm
    things_to_try2: list[
        tuple[str, bool, int, bool, bool, int, int, float, float, float]
    ] = [
        ("REINFORCE", False, 1, False, True, 1, 16, 1e-2, 0.5, 1e-4),
        ("REINFORCE", False, 1, False, True, 1, 16, 2e-2, 0.5, 1e-4),
        ("REINFORCE", False, 1, False, True, 1, 16, 5e-3, 0.5, 1e-4),
        ("REINFORCE", False, 1, False, True, 1, 8, 1e-2, 0.5, 1e-4),
        ("REINFORCE", False, 1, False, True, 1, 32, 1e-2, 0.5, 1e-4),
        ("REINFORCE", False, 1, False, True, 2, 16, 1e-2, 0.5, 1e-4),
        ("REINFORCE", False, 1, False, True, 1, 16, 1e-2, 0.7, 1e-4),
        ("REINFORCE", False, 1, False, True, 1, 16, 1e-2, 0.3, 1e-4),
        #
        ("REINFORCE", False, 2, True, True, 1, 16, 1e-2, 0.5, 1e-4),
        ("REINFORCE", False, 2, True, True, 1, 16, 2e-2, 0.5, 1e-4),
        ("REINFORCE", False, 2, True, True, 1, 16, 5e-3, 0.5, 1e-4),
        ("REINFORCE", False, 2, True, True, 1, 8, 1e-2, 0.5, 1e-4),
        ("REINFORCE", False, 2, True, True, 1, 32, 1e-2, 0.5, 1e-4),
        ("REINFORCE", False, 2, True, True, 2, 16, 1e-2, 0.5, 1e-4),
        ("REINFORCE", False, 2, True, True, 1, 16, 1e-2, 0.7, 1e-4),
        ("REINFORCE", False, 2, True, True, 1, 16, 1e-2, 0.3, 1e-4),
        #
        ("REINFORCE Memory", False, 2, True, False, 1, 16, 5e-4, 0.3, 1e-4),
        ("REINFORCE Memory", False, 2, True, False, 1, 16, 1e-3, 0.3, 1e-4),
        ("REINFORCE Memory", False, 2, True, False, 1, 16, 1e-4, 0.3, 1e-4),
        ("REINFORCE Memory", False, 2, True, False, 1, 16, 2e-4, 0.3, 1e-4),
        ("REINFORCE Memory", False, 2, True, False, 1, 8, 5e-4, 0.3, 1e-4),
        ("REINFORCE Memory", False, 2, True, False, 1, 32, 5e-4, 0.3, 1e-4),
        ("REINFORCE Memory", False, 2, True, False, 2, 16, 5e-4, 0.3, 1e-4),
        ("REINFORCE Memory", False, 2, True, False, 1, 16, 5e-4, 0.5, 1e-4),
        ("REINFORCE Memory", False, 2, True, False, 1, 16, 5e-4, 0.2, 1e-4),

    ]

    # best
    # model name, env hard?, seq_len, has_mem, MLP, N, d_model, lr, pi_loss_scale, norm
    things_to_try3: list[tuple[str, bool, int, bool, bool, int, int, float, float, float]] = [
        ("REINFORCE", False, 1, False, True, 2, 32, 0.02, 0.7, 1e-4),
        ("REINFORCE", False, 2, True, True, 2, 32, 0.01, 0.5, 1e-4),

        ("REINFORCE", True, 1, False, True, 2, 32, 0.02, 0.7, 1e-4),
        ("REINFORCE", True, 2, True, True, 2, 32, 0.01, 0.5, 1e-4),
    ]

    # Transformer
    # best lr
    things_to_try4: list[tuple[str, bool, int, bool, bool, int, int, float, float, float]] = [
        ("REINFORCE Memory", True, 2, True, False, 1, 16, 0.1, 0.3, 1e-4),
        ("REINFORCE Memory", True, 2, True, False, 1, 16, 0.2, 0.3, 1e-4),
        ("REINFORCE Memory", True, 2, True, False, 1, 16, 0.01, 0.3, 1e-4),
        ("REINFORCE Memory", True, 2, True, False, 1, 16, 0.02, 0.3, 1e-3),
    ]

    # model name, env hard?, seq_len, has_mem, MLP, N, d_model, lr, pi_loss_scale, norm
    things_to_try5: list[
        tuple[str, bool, int, bool, bool, int, int, float, float, float]
    ] = [
        ("REINFORCE Memory", False, 2, True, False, 1, 16, 1e-2, 0.3, 1e-4),
        ("REINFORCE Memory", False, 2, True, False, 2, 16, 1e-2, 0.3, 1e-4),
        ("REINFORCE Memory", False, 2, True, False, 1, 32, 1e-2, 0.3, 1e-4),
        ("REINFORCE Memory", False, 2, True, False, 1, 8, 1e-2, 0.3, 1e-4),
        ("REINFORCE Memory", False, 2, True, False, 1, 16, 1e-2, 0.5, 1e-4),
        ("REINFORCE Memory", False, 2, True, False, 1, 16, 1e-2, 0.7, 1e-4),
        ("REINFORCE Memory", False, 2, True, False, 1, 16, 1e-2, 0.3, 1e-3),
        ("REINFORCE Memory", False, 2, True, False, 1, 16, 1e-2, 0.3, 1e-2),
    ]
    
    things_to_try6: list[
        tuple[str, bool, int, bool, bool, int, int, float, float, float]
    ] = [
        ("REINFORCE Memory", False, 2, True, False, 1, 16, 1e-2, 0.3, 1e-3),
        ("REINFORCE Memory", True, 2, True, False, 1, 16, 1e-2, 0.3, 1e-3),
    ]

    for things_to_try, n_seeds in zip(
        [
            things_to_try1,
            things_to_try2,
            things_to_try3,
            things_to_try4,
            things_to_try5,
            things_to_try6,
        ],
        [1, 1, 5, 1, 1, 5]
    ):
        for seed in range(n_seeds):
            for (
                model,
                hard,
                seq_len,
                has_mem,
                MLP,
                N,
                d_model,
                lr,
                pi_loss_scale,
                norm,
            ) in things_to_try:
                run_name = f"{'hard' if hard else 'easy'}_seq_len{seq_len}_N{N}_d{d_model}_lr{lr}_pi{pi_loss_scale}"
                if not MLP:
                    run_name += f"_l2{norm}"
                if already_ran(model, run_name, seed):
                    print(f"Already ran {model} {run_name} {seed}")
                    continue
                print(f"Running {model} {run_name} {seed}")
                set_seed(seed)
                env = (
                    DiscreteTennisHard(seed=seed)
                    if hard
                    else DiscreteTennisEasy(seed=seed)
                )
                exp = Experiment(
                    env=env,
                    seq_len=seq_len,
                    has_mem=has_mem,
                    MLP=MLP,
                    pi_loss_scale=pi_loss_scale,
                    N=N,
                    d_model=d_model,
                    lr=lr,
                    norm=norm,
                )

                def handler(signum, frame):
                    raise TimeoutError()

                signal.signal(signal.SIGALRM, handler)

                try:
                    signal.alarm(60 * 30)  # 30 minutes
                    returns, losses = exp.train(
                        5000,
                        verbose=False,
                        callback=lambda returns: save_returns(
                            returns, model, run_name, seed
                        ),
                    )
                    # exp.visualize(returns)
                    # exp.visualize([l[0] for l in losses])
                    # exp.visualize([l[1] for l in losses])
                    # exp.eval(5)
                except Exception as e:
                    print(e)
                    continue
