#!/usr/bin/env python3

#dd7e3410-38c0-11e8-9b58-00505601122b
#6e14ef6b-3281-11e8-9de3-00505601122b

import argparse
import sys

import numpy as np

class MultiArmedBandits():
    def __init__(self, bandits, episode_length, seed=42):
        self._generator = np.random.RandomState(seed)

        self._bandits = []
        for _ in range(bandits):
            self._bandits.append(self._generator.normal(0., 1.))
        self._done = True
        self._episode_length = episode_length

    def reset(self):
        self._done = False
        self._trials = 0
        return None

    def step(self, action):
        if self._done:
            raise ValueError("Cannot step in MultiArmedBandits when there is no running episode")
        self._trials += 1
        self._done = self._trials == self._episode_length
        reward = self._generator.normal(self._bandits[action], 1.)
        return None, reward, self._done, {}

parser = argparse.ArgumentParser()
parser.add_argument("--bandits", default=10, type=int, help="Number of bandits.")
parser.add_argument("--episodes", default=100, type=int, help="Training episodes.")
parser.add_argument("--episode_length", default=1000, type=int, help="Number of trials per episode.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")

parser.add_argument("--mode", default="gradient", type=str, help="Mode to use -- greedy, ucb and gradient.")
parser.add_argument("--alpha", default=0, type=float, help="Learning rate to use (if applicable).")
parser.add_argument("--c", default=1, type=float, help="Confidence level in ucb (if applicable).")
parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration factor (if applicable).")
parser.add_argument("--initial", default=0, type=float, help="Initial value function levels (if applicable).")


def greedy(q, args):
    if np.random.uniform(size=1)[0] > args.epsilon:
        return np.argmax(q)
    else:
        return np.random.randint(low=0, high=args.bandits, size=1)[0]

def ucb(q, n, args):
    for i in range(args.bandits):
        if n[i] == 0:
            return i

    t = np.sum(n)
    return np.argmax(q + args.c * np.sqrt(np.log(t) / n))

def softmax(h):
    dist = [np.math.exp(val) for val in h]
    total = np.sum(dist)
    return [val / total for val in dist]

def gradient(h, bandits):
    distribution = softmax(h)
    return np.random.choice(bandits, p = distribution)

def main(args):
    # Fix random seed
    np.random.seed(args.seed)

    # Create environment
    env = MultiArmedBandits(args.bandits, args.episode_length)

    rewards = np.zeros(args.episodes)
    for episode in range(args.episodes):
        env.reset()

        # TODO: Initialize parameters (depending on mode).
        q = np.full(shape = args.bandits, fill_value=args.initial)
        n = np.zeros(args.bandits)
        h = np.full(args.bandits, fill_value=args.initial)

        done = False
        steps = 0
        reward_sum = 0
        while not done:
            # TODO: Action selection according to mode
            if args.mode == "greedy":
                action = greedy(q, args)
            elif args.mode == "ucb":
                action = ucb(q, n, args)
            elif args.mode == "gradient":
                action = gradient(h, args.bandits) 

            _, reward, done, _ = env.step(action)
            # TODO: Update parameters
            n[action] += 1
            steps += 1
            reward_sum += reward
            if args.mode == "gradient":
                dist = softmax(h)
                for a, p in enumerate(dist):
                    h[a] += args.alpha * reward * ((a == action) - p)
            elif args.alpha != 0:
                q[action] += args.alpha * (reward - q[action])
            else:
                q[action] += (reward - q[action]) / n[action]

        rewards[episode] = reward_sum / steps

    # TODO: For every episode, compute its average reward (a single number),
    # obtaining `args.episodes` values. Then return the final score as
    # mean and standard deviation of these `args.episodes` values.
    return np.mean(rewards), np.std(rewards)

if __name__ == "__main__":
    mean, std = main(parser.parse_args())
    # Print the mean and std for ReCodEx to validate
    print("{:.2f} {:.2f}".format(mean, std))
