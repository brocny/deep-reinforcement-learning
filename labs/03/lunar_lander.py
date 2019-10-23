#!/usr/bin/env python3
import numpy as np

import lunar_lander_evaluator

import math

class ExpDecay:
    def __init__(self, initial, final, epochs):
        self.initial = initial
        self.final = final
        self.current = initial
        self.decay = math.pow((initial / final), (1 / epochs))
        self.episode = 0

    def get(self, episode=None):
        if episode != None and episode != self.episode:
            return self.initial / (self.decay ** episode)

        self.episode += 1    
        current = self.current
        self.current /= self.decay
        return current


def progress_log(Q, progress):
    print("{0} %".format(progress))
    for i in range(100):
        state, done = env.reset(), False
        while not done:
            action = np.argmax(Q[state])
            state, reward, done, _ = env.step(action)       


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=500, type=int, help="Training episodes.")
    parser.add_argument("--from_expert_episodes", default=500, type=int, help="Learn from expert episodes.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")

    parser.add_argument("--alpha", default=0.05, type=float, help="Learning rate.")
    parser.add_argument("--alpha_final", default=0.005, type=float, help="Final learning rate.")
    parser.add_argument("--epsilon", default=0.4, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.05, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=0.98, type=float, help="Discounting factor.")
    args = parser.parse_args()

    # Create the environment
    env = lunar_lander_evaluator.environment()

    # The environment has env.states states and env.actions actions.

    # TODO: Implement a suitable RL algorithm.
    #
    # The overall structure of the code follows.
    Q = np.zeros([env.states, env.actions])

    alpha_scheduler = ExpDecay(args.alpha, args.alpha_final, args.episodes)
    epsilon_scheduler = ExpDecay(args.epsilon, args.epsilon_final, args.episodes)

    print('Learning from expert')
    # learn from expert
    for episode in range(args.from_expert_episodes):
        if episode % (args.from_expert_episodes / 10) == 0:
            progress_log(Q, episode / args.from_expert_episodes * 100)
        # trajectory = list((action, reward, state)), state = initial state
        state, trajectory = env.expert_trajectory()
        alpha = alpha_scheduler.get()
        for action, reward, new_state in trajectory: 
            Q[state, action] += alpha * (reward + args.gamma * Q[new_state].max() - Q[state, action])
            state = new_state





    # Perform last 100 evaluation episodes
    while True:
        state, done = env.reset(True), False
        while not done:
            action = np.argmax(Q[state])
            state, reward, done, _ = env.step(action)