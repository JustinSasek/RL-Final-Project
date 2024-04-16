from dataclasses import dataclass

import gym
import numpy as np
import seaborn as sns  # type: ignore
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm  # type: ignore

from rltennis.src.rl.models import SingleVectorWrapper, Transformer
from rltennis.src.rl.other_envs.random_walk import RandomWalkEnv


class Agent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        horizon: int,
        epilison: float,
        discount_rate: float,
    ):
        self.action_size = action_size
        self.epilison = epilison
        self.discount_rate = discount_rate
        transformer = Transformer(
            N=1, d_model=8, d_ff=16, h=2, max_len=horizon, dropout=0.1
        )
        self.model = SingleVectorWrapper(
            transformer=transformer,
            input_size=state_size + 3,  # action, reward, done, state
            output_size=action_size + 1,  # value, action1, action2, .
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def __call__(self, hist: list[list[float]]) -> int:
        if np.random.rand() < self.epilison:
            return np.random.randint(self.action_size)
        hist_tensor = torch.tensor(hist, dtype=torch.float32)
        output: torch.Tensor = self.model(hist_tensor)[-1]  # only use last timestep
        Qs: torch.Tensor = output[..., 1:] + output[..., 0].unsqueeze(-1)
        return int(torch.argmax(Qs).item())

    def update(self, hists: list[list[list[float]]]):
        hist_tensor = torch.tensor(hists, dtype=torch.float32)

        rewards = hist_tensor[..., 1]
        actions = hist_tensor[..., 0].to(torch.int64)
        discounts = self.discount_rate ** torch.arange(
            rewards.shape[-1], dtype=torch.float32
        )
        discounts = discounts.unsqueeze(0)
        rewards *= discounts
        returns = torch.cumsum(rewards.flip(-1), dim=-1).flip(-1)
        returns /= discounts

        # shift tensors
        returns = returns[..., 1:]
        hist_tensor = hist_tensor[..., :-1, :]
        actions = actions[..., 1:]

        outputs: torch.Tensor = self.model(hist_tensor)
        values = outputs[..., 0]
        Qs = outputs[..., 1:]

        value_loss = torch.nn.functional.mse_loss(values, returns)
        Qs_taken = torch.gather(Qs, -1, actions.unsqueeze(-1)).squeeze(-1)
        Q_deltas = returns - Qs_taken
        Q_loss = -torch.mean(Q_deltas * torch.log(Qs_taken))
        loss = value_loss + Q_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


@dataclass
class Experiment:
    seq_len: int = 10
    n_episodes: int = 100
    discount_rate: float = 0.8
    epilison: float = 0.1
    batch_size: int = 1

    def run(self) -> list[float]:
        # env = gym.make("MountainCar-v0")
        env = RandomWalkEnv()

        pi = Agent(
            1,
            env.action_space.n,  # type: ignore
            self.seq_len,
            self.epilison,
            self.discount_rate,
        )

        hists = []
        returns = []
        for _ in tqdm(range(self.n_episodes)):
            s, _ = env.reset()
            hist = [[0, 0.0, False, *s]]  # a, r, done, s
            ep_return = 0.0
            while True:
                a = pi(hist)
                s, r, done, term, *_ = env.step(a)
                ep_return += r
                done = done or term
                hist.append([a, r, done, *s])
                if len(hist) >= self.seq_len:  # TODO: keep history together, split up in the update function so that returns are in tact
                    hists.append(hist)
                    hist = hist[-1:]
                    if len(hists) >= self.batch_size:
                        pi.update(hists)
                        hists = []
                if done:
                    hists.append(hist)
                    if len(hists) >= self.batch_size:
                        pi.update(hists)
                        hists = []
                    returns.append(ep_return)
                    break
            s = env.reset()
        return returns


if __name__ == "__main__":
    exp = Experiment(n_episodes=1000)
    returns = exp.run()
    sns.lineplot(x=range(len(returns)), y=returns)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Returns over Episodes")
    plt.show()
