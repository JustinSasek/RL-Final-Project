import random
from collections import Counter
from dataclasses import dataclass

import gym
import numpy as np
import seaborn as sns  # type: ignore
import torch
from matplotlib import pyplot as plt
from rl.models import SingleVectorWrapper, Transformer
from rl.tennis.discreteTennisWrappers import *
from tqdm import tqdm  # type: ignore

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
STATE_FILTER = [1, 0, 6, 5, 4]

class Agent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        seq_len: int,
        epilison: float,
        discount_rate: float,
    ):
        self.action_size = action_size
        self.seq_len = seq_len
        self.epilison = epilison
        self.discount_rate = discount_rate
        transformer = Transformer(
            N=1, d_model=16, d_ff=32, h=2, max_len=seq_len, dropout=0.1
        )
        self.model = SingleVectorWrapper(
            transformer=transformer,
            input_size=state_size + 3,  # action, reward, done, state
            output_size=action_size + 1,  # value, action1, action2, ...
        )
        # optimistic initialization
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = None
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 2, 0.5)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 10)

    def __call__(self, hist: list[list[float]]) -> int:
        if np.random.rand() < self.epilison:
            return np.random.randint(self.action_size)
        hist_tensor = torch.tensor(hist, dtype=torch.float32)
        hist_tensor = hist_tensor[..., -self.seq_len :, :]
        output: torch.Tensor = self.model(hist_tensor)[-1]  # only use last timestep
        pis: torch.Tensor = output[..., 1:]
        pis = torch.nn.functional.softmax(pis, dim=-1)
        selected_actions = int(torch.multinomial(pis, num_samples=1).squeeze(-1))
        return selected_actions

    def process_hist(
        self,
        hist: list[list[float]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Finds returns and aligns tensors for training."""
        hist_tensor = torch.tensor(
            hist, dtype=torch.float32
        )  # boost precision for discounting
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

        outputs: torch.Tensor = self.model(hists_tensor)
        outputs = outputs[..., -1, :]  # only use last timestep
        values = outputs[..., 0]
        pis = outputs[..., 1:]
        pis = torch.nn.functional.softmax(pis, dim=-1)

        value_loss = torch.nn.functional.mse_loss(values, returns)
        pi_taken = torch.gather(pis, -1, actions.unsqueeze(-1)).squeeze(-1)
        deltas = returns - values
        deltas = deltas.detach()
        pi_loss = -torch.mean(deltas * torch.log(pi_taken))
        loss = value_loss + 0.5 * pi_loss

        if torch.isnan(loss):
            raise ValueError("NAN Loss")

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

        if torch.isnan(self.model._modules["fc_out"].weight).any():
            raise ValueError("NAN Weights")

        return value_loss.item(), pi_loss.item()


@dataclass
class Experiment:
    seq_len: int = 2
    discount_rate: float = 0.99
    epilison: float = 0.1
    batch_size: int = 4
    horizon: int = 64
    max_game_length: int = 4
    has_mem: bool = True

    def __post_init__(self):
        self.env = DiscreteTennisHard(max_game_length = self.max_game_length)
        self.pi = Agent(
            len(STATE_FILTER),
            self.env.action_space.n,  # type: ignore
            self.seq_len,
            self.epilison,
            self.discount_rate,
        )

    def run(
        self,
        n_episodes: int,
        update: bool,
        starting_epsilon: float,
        verbose: bool = False,
    ) -> tuple[list[float], list[tuple[float, float]]]:

        # Collect data, train after collecting batch_size episodes
        hists = []
        returns = []
        losses = []
        self.pi.epilison = starting_epsilon
        c: Counter = Counter()
        for ep in tqdm(range(n_episodes), colour="green"):
            if (ep + 1) % (n_episodes // 2) == 0:
                self.pi.epilison /= 2
                pass
            s, _ = self.env.reset()
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
                if self.has_mem:
                    hist.append([a, r, game_done, *s])
                else:
                    hist.append([0, 0.0, False, *s])
                t += 1
                if done or t >= self.horizon:
                    hists.append(hist)
                    returns.append(ep_return)
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
        self, n_episodes: int, verbose: bool = False
    ) -> tuple[list[float], list[tuple[float, float]]]:
        self.env._render_view = False
        return self.run(n_episodes, True, self.epilison, verbose)

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
    set_seed(42)
    exp = Experiment(seq_len=2, has_mem=True)
    returns, losses = exp.train(1000, verbose=False)
