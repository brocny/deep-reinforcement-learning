#!/usr/bin/env python3

#dd7e3410-38c0-11e8-9b58-00505601122b
#6e14ef6b-3281-11e8-9de3-00505601122b
import numpy as np

import mountain_car_evaluator



def full_state(state, size):
    s = np.zeros(size)
    for i in state:
        s[i] = 1    
    return s

def greedy(W, state):
    state = full_state(state, W.shape[0])
    return np.array([np.dot(w, state) for w in W.T]).argmax()


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=4300, type=int, help="Training episodes.")
    parser.add_argument("--render_each", default=None, type=int, help="Render some episodes.")

    parser.add_argument("--alpha", default=0.8, type=float, help="Learning rate.")
    parser.add_argument("--alpha_final", default=0.01, type=float, help="Final learning rate.")
    parser.add_argument("--epsilon", default=0.6, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.05, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=0.98, type=float, help="Discounting factor.")
    parser.add_argument("--tiles", default=8, type=int, help="Number of tiles.")
    args = parser.parse_args()

    # Create the environment
    env = mountain_car_evaluator.environment(tiles=args.tiles)

    # Implement Q-learning RL algorithm, using linear approximation.
    W = np.zeros([env.weights, env.actions])
    epsilon = args.epsilon
    alpha = args.alpha / args.tiles

    evaluating = False
    i = 0
    while not evaluating:
        # Perform a training episode
        previous_state = None
        previous_action = None
        previous_reward = None
        state, done = env.reset(evaluating), False
        while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()
            # TODO: Choose `action` according to epsilon-greedy strategy
            action = greedy(W, state) if np.random.uniform() > epsilon else np.random.randint(env.actions)
            next_state, reward, done, _ = env.step(action)

            # TODO: Update W values
            state_f = full_state(state, W.shape[0])
            next_state_f = full_state(next_state, W.shape[0])
            #estimated_value = reward + args.gamma * max([np.dot(w, next_state_f) for w in W.T])
            if not previous_state is None:
                e = previous_reward + (args.gamma * reward) + (args.gamma * args.gamma * max([np.dot(w, next_state_f) for w in W.T]))
                W.T[previous_action] += alpha * (e - np.dot(previous_state,  W.T[previous_action])) * previous_state
            previous_state = state_f
            previous_action = action
            previous_reward = reward
            state = next_state
            if done:
                break
                

        # TODO: Decide if we want to start evaluating
        if i == args.episodes:
            evaluating = True
        i += 1
        
        if not evaluating:
            if args.epsilon_final:
                epsilon = np.exp(np.interp(env.episode + 1, [0, args.episodes], [np.log(args.epsilon), np.log(args.epsilon_final)]))
            if args.alpha_final:
                alpha = np.exp(np.interp(env.episode + 1, [0, args.episodes], [np.log(args.alpha), np.log(args.alpha_final)])) / args.tiles


    #print(time.time() - start)
    # Perform the final evaluation episodes
    while True:
        state, done = env.reset(evaluating), False
        while not done:
            # TODO: choose action as a greedy action
            action = greedy(W, state)
            state, reward, done, _ = env.step(action)
