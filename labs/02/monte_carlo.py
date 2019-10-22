#!/usr/bin/env python3

#dd7e3410-38c0-11e8-9b58-00505601122b
#6e14ef6b-3281-11e8-9de3-00505601122b

import numpy as np
import cart_pole_evaluator

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")

    parser.add_argument("--epsilon", default=0.05, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=1, type=float, help="Discounting factor.")
    args = parser.parse_args()

    # Create the environment
    env = cart_pole_evaluator.environment()

    # TODO: Implement Monte-Carlo RL algorithm.
    #
    # The overall structure of the code follows.
    Q = np.zeros([env.states, env.actions])
    C = np.zeros([env.states, env.actions])

    for _ in range(args.episodes):
        # Perform a training episode
        state, done = env.reset(), False
        rewards, states, actions = [], [], []
        while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()

            action = Q[state].argmax() if np.random.uniform() > args.epsilon else np.random.randint(env.actions)
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            states.append(state)
            actions.append(action)

            state = next_state
        
        rewards.reverse()
        states.reverse()
        actions.reverse()
        G = 0
        episode_len = len(rewards)

        for t in range(episode_len - 1):
            r = rewards[t]
            s = states[t]
            a = actions[t]
            G = args.gamma * G + r
            C[s, a] = C[s, a] + 1
            Q[s, a] += 1 / C[s, a] * (G - Q[s, a])


    # Perform last 100 evaluation episodes
    while True:
        state, done = env.reset(True), False
        while not done:
            action = np.argmax(Q[state])
            state, reward, done, _ = env.step(action)

