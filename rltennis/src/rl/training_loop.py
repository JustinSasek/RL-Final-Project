from dataclasses import dataclass

import gym
import numpy as np
import torch
from tqdm import tqdm # type: ignore

from rl.models.transformer import Transformer # type: ignore
from rl.models.transformerwrapper import SingleVectorWrapper # type: ignore


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
        output: torch.Tensor = self.model(hist_tensor)
        Qs: torch.Tensor = output[..., 1:] + output[..., 0]
        return int(torch.argmax(Qs).item())

    def update(self, hist: list[list[float]]):
        rewards = np.array([r for _, r, _, _ in hist])
        discounts = self.discount_rate ** np.arange(len(rewards))
        rewards *= discounts
        returns = rewards[::-1].cumsum()[::-1]
        returns /= discounts

        # convert to tensor
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        hist_tensor = torch.tensor(hist, dtype=torch.float32)
        action_tensor = torch.tensor([a for a, _, _, _ in hist], dtype=torch.int64)

        # shift tensors
        returns_tensor = returns_tensor[1:]
        hist_tensor = hist_tensor[:-1]
        action_tensor = action_tensor[1:]

        outputs: torch.Tensor = self.model(hist_tensor)
        values = outputs[..., 0]
        Qs = outputs[..., 1:]

        value_loss = torch.nn.functional.mse_loss(values, returns_tensor)
        Qs_taken = Qs[range(len(action_tensor)), action_tensor]
        Q_deltas = returns_tensor - Qs_taken
        Q_loss = -torch.mean(Q_deltas * torch.log(Qs))
        loss = value_loss + Q_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


@dataclass
class Experiment:
    horizon: int = 1000
    n_episodes: int = 100
    discount_rate: float = 1.0
    epilison: float = 0.1

    def run(self):
        env = gym.make("CartPole-v0")

        pi = Agent(
            env.observation_space.shape[0],
            env.action_space.n,
            self.horizon,
            self.epilison,
            self.discount_rate,
        )

        s, _ = env.reset()
        hist = [[0.0, 0.0, 0.0, *s]]  # a, r, done, s

        for _ in tqdm(range(self.n_episodes)):
            for _ in range(self.horizon):
                a = pi(hist)
                s, r, done, _ = env.step(a)
                hist.append([a, r, done, *s])
                if done:
                    break
            s = env.reset()
            pi.update(hist)

        s = env.reset()
        while True:
            a = pi(s)
            s, r, done, _ = env.step(a)
            env.render()
            if done:
                break


if __name__ == "__main__":
    exp = Experiment()
    exp.run()
