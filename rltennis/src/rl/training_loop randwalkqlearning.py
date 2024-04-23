import random
from collections import Counter
from dataclasses import dataclass

import gym
import numpy as np
import seaborn as sns  # type: ignore
import torch
from matplotlib import pyplot as plt
from rl.models import SingleVectorWrapper, Transformer
from rl.other_envs.random_walk import RandomWalkEnv
from rl.tennis.behaviorLearnable import TennisBehaviorShotRewardOnly
from rl.tennis.discreteTennis import DiscreteTennis
from tqdm import tqdm  # type: ignore

# torch.manual_seed(42)
# np.random.seed(42)
# random.seed(42)


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
        # transformer = Transformer(
        #     N=1, d_model=64, d_ff=128, h=4, max_len=seq_len, dropout=0.1
        # )
        # self.model = SingleVectorWrapper(
        #     transformer=transformer,
        #     input_size=state_size + 3,  # action, reward, done, state
        #     output_size=action_size + 1,  # value, action1, action2, ...
        # )
        # # optimistic initialization
        # self.model.fc_out.weight.data.zero_()
        # self.model.fc_out.bias.data.fill_(5)
        self.q = torch.nn.Sequential(
            torch.nn.Linear(state_size + 3, 32),
            torch.nn.Sigmoid(),
            torch.nn.Linear(32, action_size),
        )
        self.q_optimizer = torch.optim.Adam(self.q.parameters(), lr=1e-3)
        self.scheduler = None
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 8, 0.5)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 10)

    def __call__(self, hist) -> int:
        # if hist[-1][1] == 0:
        #     return hist[-1][0]
        # return hist[-1][0] * -1 + 1
        if np.random.rand() < self.epilison:
            return np.random.randint(self.action_size)
        hist_tensor = torch.tensor(hist, dtype=torch.float32)
        hist_tensor = hist_tensor[..., -self.seq_len :, :]
        qs: torch.Tensor = self.q(hist_tensor)[-1]  # only use last timestep
        return int(torch.argmax(qs).item())

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

        qs: torch.Tensor = self.q(hists_tensor)
        qs = qs[..., -1, :]  # only use last timestep
        qs_taken = torch.gather(qs, -1, actions.unsqueeze(-1)).squeeze(-1)

        q_loss = torch.nn.functional.mse_loss(qs_taken, returns)

        if torch.isnan(q_loss):
            raise ValueError("NAN Loss")

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        return q_loss.item()


@dataclass
class Experiment:
    seq_len: int = 1
    discount_rate: float = 0.9
    epilison: float = 0.9
    batch_size: int = 32
    horizon: int = 20000
    max_game_length: int = 20000

    def __post_init__(self):
        self.env = RandomWalkEnv()
        self.pi = Agent(
            1,
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
        self.env.reset()

        # Collect data, train after collecting batch_size episodes
        hists = []
        returns = []
        losses = []
        self.pi.epilison = starting_epsilon
        c: Counter = Counter()
        for ep in tqdm(range(n_episodes), colour="green"):
            if (ep + 1) % (n_episodes // 8) == 0:
                self.pi.epilison /= 2
                pass
            s, _ = self.env.reset()
            hist = [[0, 0.0, False, *s]]  # a, r, done, s
            ep_return = 0.0
            t = 0
            while True:
                a = self.pi(hist)
                c[a] += 1
                s, r, done, term, *_ = self.env.step(a)
                self.env.render()
                ep_return += r
                done = done or term
                hist.append([a, r, done, *s])
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
        model_dict = {"q": self.pi.q.state_dict()}
        torch.save(model_dict, path)

    def load(self, path: str):
        model_dict = torch.load(path)
        self.pi.q.load_state_dict(model_dict["q"])

    @staticmethod
    def visualize(returns: list[float]):
        sns.lineplot(x=range(len(returns)), y=returns)
        plt.xlabel("Episode")
        plt.show()


if __name__ == "__main__":
    exp = Experiment()
    # exp.load("modelrw.pt")

    try:
        returns, losses = exp.train(10000, verbose=False)
        exp.visualize(returns)
        exp.visualize(losses)
    except KeyboardInterrupt:
        pass
    exp.save("modelrw.pt")

    returns, _ = exp.eval(5)
    exp.visualize(returns)
