import gym
import numpy as np
from gym import spaces


class RandomWalkEnv(gym.Env):
    def __init__(self, num_states=10):
        super(RandomWalkEnv, self).__init__()
        self.num_states = num_states
        self.state = 0
        self.action_space = spaces.Discrete(2)  # Two actions: move left or move right
        self.observation_space = spaces.Discrete(num_states)
        self.reset()

    def reset(self):
        self.count = 0
        self.goal = np.random.randint(0, self.num_states - 1)
        self.state = np.random.randint(0, self.num_states - 1)
        while self.state == self.goal:
            self.state = np.random.randint(0, self.num_states - 1)
        return np.array([self.state]), None

    def step(self, action):
        self.count += 1
        assert self.action_space.contains(action)

        if self.state < self.goal:
            if action == 0:  # Move left
                self.state = max(0, self.state - 1)
                reward = -1
            else:  # Move right
                self.state += 1
                reward = 1
        else:
            if action == 0:
                self.state -= 1
                reward = 1
            else:
                self.state = min(self.num_states - 1, self.state + 1)
                reward = -1

        done = False
        if self.state == self.goal:
            reward = 10
            done = True

        return np.array([self.state]), reward, done, self.count >= 50, {}

    def render(self, mode="human"):
        # Not implemented for this simple environment
        pass


if __name__ == "__main__":
    for ep in range(100):
        env = RandomWalkEnv()
        s, _ = env.reset()
        R = 0
        s, r, done, term, _ = env.step(0)
        R += r
        if done:
            pass
        elif r == 1:
            while not done and not term:
                s, r, done, term, _ = env.step(0)
                R += r

        else:
            while not done and not term:
                s, r, done, term, _ = env.step(1)
                R += r
        print(f"Episode {ep} done with return {R}")
