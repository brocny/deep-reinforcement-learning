#!/usr/bin/env python3

#dd7e3410-38c0-11e8-9b58-00505601122b
#6e14ef6b-3281-11e8-9de3-00505601122b


import numpy as np
import lunar_lander_evaluator
import math
import pickle

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


def action_distribution(state):
    minn = state.min()
    normalized_state = [s_a - minn for s_a in state]
    summ = sum(normalized_state)
    if summ == 0:
        return np.full(env.actions, 1 / env.actions)
    return np.array([s_a / summ for s_a in normalized_state])


def q_learning(state):
    return state.max()


def expected_sarsa(state):
    return np.dot(action_distribution(state), state)


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exploring_episodes", default=0, type=int, help="Training episodes.")
    parser.add_argument("--from_expert_episodes", default=0, type=int, help="Learn from expert episodes.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")

    parser.add_argument("--alpha", default=0.05, type=float, help="Learning rate.")
    parser.add_argument("--alpha_final", default=0.005, type=float, help="Final learning rate.")
    parser.add_argument("--epsilon", default=0.4, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.05, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=0.98, type=float, help="Discounting factor.")

    parser.add_argument("--model", default="model.py", type=str, help="Load pretrained model.")
    args = parser.parse_args()

    # Create the environment
    env = lunar_lander_evaluator.environment()

    # The environment has env.states states and env.actions actions.

    # TODO: Implement a suitable RL algorithm.
    #
    # The overall structure of the code follows.

    if args.model != '':
        Q = np.loadtxt(args.model)
    else:
        Q = np.zeros([env.states, env.actions])
        alpha_scheduler = ExpDecay(args.alpha, args.alpha_final, args.exploring_episodes + args.from_expert_episodes)
        epsilon_scheduler = ExpDecay(args.epsilon, args.epsilon_final, args.exploring_episodes)

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

        print('Exploring')
        for episode in range(args.exploring_episodes):
            if episode % (args.exploring_episodes / 10) == 0:
                progress_log(Q, episode / args.exploring_episodes * 100)
            state, done = env.reset(), False
            epsilon = epsilon_scheduler.get()
            alpha = alpha_scheduler.get()
            while not done:
                action = Q[state].argmax() if np.random.uniform() > epsilon else np.random.randint(env.actions)
                #action = np.random.choice(env.actions, p = actions_distrituction(Q[state])) if np.random.uniform() > epsilon else np.random.randint(env.actions)
                next_state, reward, done, _ = env.step(action)
                #target_value = Q_learning(Q[next_state])
                #target_value = Expected_sarsa(Q[next_state])
                Q[state, action] += alpha * (reward + args.gamma * Q[next_state].max() - Q[state, action])
                state = next_state 
        #np.savetxt('model.py', Q)

    # Perform last 100 evaluation episodes
    while True:
        state, done = env.reset(True), False
        while not done:
            action = np.argmax(Q[state])
            state, reward, done, _ = env.step(action)


