import random
import os
import time
import numpy as np
import ray

class Discrete:
    def __init__(self, num_actions: int):
        self.n = num_actions

    def sample(self):
        return random.randint(0, self.n - 1)

class Environment:
    def __init__(self, n, random_init=False):
        self.seeker = (0,0)
        if random_init:
            self.goal = (random.randint(0, n-1), random.randint(0, n-1))
        else:
            self.goal = (n-1, n-1)
        self.info = {'seeker': self.seeker, 'goal': self.goal}

        self.action_space = Discrete(4)
        self.observation_space = Discrete(n*n)

    def reset(self):
        self.seeker = (0,0)
        return self.get_observation()

    def get_observation(self):
        return 5 * self.seeker[0]+ self.seeker[1]

    def get_reward(self):
        return 1 if self.seeker == self.goal else 0

    def is_done(self):
        return self.seeker == self.goal

    def step(self, action):
        if action == 0:
            self.seeker = (min(self.seeker[0] + 1, 4), self.seeker[1])
        elif action == 1:
            self.seeker = (self.seeker[0], max(self.seeker[1] - 1, 0))
        elif action == 2:
            self.seeker = (max(self.seeker[0] - 1, 0), self.seeker[1])
        elif action == 3:
            self.seeker = (self.seeker[0], min(self.seeker[1] + 1, 4))
        else:
            raise ValueError("Invalid action")

        obs = self.get_observation()
        rew = self.get_reward()
        done = self.is_done()
        return obs, rew, done, self.info

    def render(self):
        os.system('cls' if os.name == 'nt' else 'clear')

        grid = [['|' for _ in range(5)] + ['|\n'] for _ in range(5)]
        grid[self.seeker[0]][self.seeker[1]] = '|S'
        grid[self.goal[0]][self.goal[1]] = '|G'
        print(''.join([''.join(grid_row) for grid_row in grid]))


class Policy:
    def __init__(self, env):
        self.state_action_table = [
            [0 for _ in range(env.action_space.n)]
            for _ in range(env.observation_space.n)
        ]
        self.action_space = env.action_space

    def get_action(self, state, explore=True, epsilon=0.1):
        if explore and random.uniform(0, 1) < epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self.state_action_table[state])

class Simulation(object):
    def __init__(self, env):
        self.env = env

    def rollout(self, policy, render=False, explore=True, epsilon=0.1):
        experience = []
        state = self.env.reset()
        done = False
        while not done:
            action = policy.get_action(state, explore, epsilon)
            next_state, reward, done, info = self.env.step(action)
            experience.append([state, action, reward, next_state])
            state = next_state
            if render:
                self.env.render()
        return experience





# environment  = Environment(5, random_init=True)
environment = Environment(5)

# untrained_policy = Policy(environment)
# sim = Simulation(environment)
# exp = sim.rollout(untrained_policy, render=True, epsilon=0.1)
# for row in untrained_policy.state_action_table:
#     print(row)


def update_policy(policy, experiences, weight=0.1, discount_factor=0.9):
    for state, action, reward, next_state in experiences:
        next_max = np.max(policy.state_action_table[next_state])
        value = policy.state_action_table[state][action]
        new_value = (1 - weight) * value + weight * (reward + discount_factor * next_max)
        policy.state_action_table[state][action] = new_value


def train_policy(env, num_episodes=10000, weight=0.1, discount_factor=0.9):
    policy = Policy(env)
    sim = Simulation(env)
    for i in range(num_episodes):
        # print("Episode:", i)
        experiences = sim.rollout(policy)
        update_policy(policy, experiences, weight, discount_factor)

    return policy

# trained_policy = train_policy(environment)

def evaluate_policy(env, policy, num_episodes=10):
    simulation = Simulation(env)
    steps = 0
    for _ in range(num_episodes):
        experiences = simulation.rollout(policy, render=True, explore=False)
        steps += len(experiences)

    print(f"{steps / num_episodes} steps on average "
          f"for a total of {num_episodes} episodes")
    return steps / num_episodes

# evaluate_policy(environment, trained_policy)

ray.init()

@ray.remote
class SimulationActor(Simulation):
    def __init__(self):
        env = Environment(5)
        super().__init__(env)


def train_policy_parallel(env, num_episodes=1000, num_simulations=4):
    policy = Policy(env)
    simulations = [SimulationActor.remote() for _ in range(num_simulations)]

    policy_ref = ray.put(policy)
    for _ in range(num_episodes):
        experiences = [sim.rollout.remote(policy_ref) for sim in simulations]
        while len(experiences) > 0:
            finished, experiences = ray.wait(experiences)
            for xp in ray.get(finished):
                update_policy(policy, xp)

    return policy

parallel_policy = train_policy_parallel(environment)

evaluate_policy(environment, parallel_policy)

input()