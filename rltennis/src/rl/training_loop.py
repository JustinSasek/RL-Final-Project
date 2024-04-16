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
        seq_len: int,
        epilison: float,
        discount_rate: float,
    ):
        self.action_size = action_size
        self.seq_len = seq_len
        self.epilison = epilison
        self.discount_rate = discount_rate
        transformer = Transformer(
            N=1, d_model=16, d_ff=32, h=4, max_len=seq_len, dropout=0.1
        )
        self.model = SingleVectorWrapper(
            transformer=transformer,
            input_size=state_size + 3,  # action, reward, done, state
            output_size=action_size + 1,  # value, action1, action2, .
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)
        self.scheduler = None
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 20, 0.5)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 50)

    def __call__(self, hist: list[list[float]]) -> int:
        # if hist[-1][1] == 1:
        #     return hist[-1][0]
        # return 1 - hist[-1][0]
        if np.random.rand() < self.epilison:
            return np.random.randint(self.action_size)
        hist_tensor = torch.tensor(hist, dtype=torch.float32)
        hist_tensor = hist_tensor[..., -self.seq_len :, :]
        output: torch.Tensor = self.model(hist_tensor)[-1]  # only use last timestep
        pis: torch.Tensor = output[..., 1:]
        return int(torch.argmax(pis).item())

    def process_hist(
        self,
        hist: list[list[float]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hist_tensor = torch.tensor(hist, dtype=torch.float32)
        rewards = hist_tensor[..., 1]
        actions = hist_tensor[..., 0].to(torch.int64)
        discounts = self.discount_rate ** torch.arange(
            rewards.shape[-1], dtype=torch.float32
        )
        discounts = discounts
        rewards *= discounts
        returns = torch.cumsum(rewards.flip(-1), dim=-1).flip(-1)
        returns /= discounts

        # shift tensors
        returns = returns[..., 1:]
        hist_tensor = hist_tensor[..., :-1, :]
        actions = actions[..., 1:]
        return hist_tensor, actions, returns

    def split_seq(
        self, hist_tensor: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
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
        actions = torch.stack(actions_list)
        returns = torch.stack(returns_list)

        outputs: torch.Tensor = self.model(hists_tensor)
        values = outputs[..., 0]
        pis = outputs[..., 1:]

        value_loss = torch.nn.functional.mse_loss(values, returns)
        pi_taken = torch.gather(pis, -1, actions.unsqueeze(-1)).squeeze(-1)
        deltas = returns - values
        deltas = deltas.detach()
        pi_loss = -torch.mean(deltas * torch.log(pi_taken))
        loss = value_loss + 0.5 * pi_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

        return value_loss.item(), pi_loss.item()


@dataclass
class Experiment:
    seq_len: int = 1
    discount_rate: float = 0.5
    epilison: float = 0.1
    batch_size: int = 16

    def __post_init__(self):
        env = RandomWalkEnv()
        self.pi = Agent(
            1,
            env.action_space.n,  # type: ignore
            self.seq_len,
            self.epilison,
            self.discount_rate,
        )

    def run(self, n_episodes: int, update: bool, starting_epsilon: float):
        # env = gym.make("MountainCar-v0")
        env = RandomWalkEnv()

        hists = []
        returns = []
        losses = []
        self.pi.epilison = starting_epsilon
        for ep in tqdm(range(n_episodes), colour="green"):
            if (ep + 1) % (n_episodes // 4) == 0:
                self.pi.epilison /= 2
                pass
            s, _ = env.reset()
            hist = [[0, 0.0, False, *s]]  # a, r, done, s
            ep_return = 0.0
            while True:
                a = self.pi(hist)
                s, r, done, term, *_ = env.step(a)
                ep_return += r
                done = done or term
                hist.append([a, r, done, *s])
                if done:
                    hists.append(hist)
                    returns.append(ep_return)
                    break
            s = env.reset()
            if len(hists) >= self.batch_size:
                if update:
                    losses.append(self.pi.update(hists))
                hists = []
        return returns, losses

    def train(self, n_episodes: int) -> list[float]:
        return self.run(n_episodes, True, self.epilison)

    def eval(self, n_episodes: int) -> list[float]:
        return self.run(n_episodes, False, 0.0)

    @staticmethod
    def visualize(returns: list[float]):
        sns.lineplot(x=range(len(returns)), y=returns)
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title("Returns over Episodes")
        plt.show()


if __name__ == "__main__":
    exp = Experiment()
    returns, losses = exp.train(1000)
    exp.visualize(returns)
    exp.visualize([val for val, _ in losses])
    exp.visualize([pi for _, pi in losses])
    # returns = exp.eval(100)
    # exp.visualize(returns)
