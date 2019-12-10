# dd7e3410-38c0-11e8-9b58-00505601122b
# 6e14ef6b-3281-11e8-9de3-00505601122b

#!/usr/bin/env python3
import numpy as np
import tensorflow as tf


import gym_evaluator



class Network:
    def __init__(self, env, args):
        # TODO: Similarly to reinforce, define two models:
        # - _policy, which predicts distribution over the actions
        # - _value, which predicts the value function
        # Use independent networks for both of them, each with
        # `args.hidden_layer` neurons in one hidden layer,
        # and train them using Adam with given `args.learning_rate`.
        self._policy, self._value = tf.keras.Sequential(), tf.keras.Sequential()
        self._policy.add(tf.keras.layers.Dense(args.hidden_layer, activation = tf.nn.tanh, input_shape = env.state_shape))
        self._policy.add(tf.keras.layers.Dense(env.actions, activation = tf.nn.softmax))
        self._policy.compile(
            optimizer = tf.optimizers.Adam(args.learning_rate), 
            loss      = 'sparse_categorical_crossentropy',
            experimental_use_tf_function=False
        )

        self._value.add(tf.keras.layers.Dense(args.hidden_layer, activation = tf.nn.tanh, input_shape = env.state_shape))
        self._value.add(tf.keras.layers.Dense(1))
        self._value.compile(
            optimizer = tf.optimizers.Adam(args.learning_rate), 
            loss      = 'mean_squared_error',
            experimental_use_tf_function=False
        )



    def train(self, states, actions, returns):
        states, actions, returns = np.array(states, np.float32), np.array(actions, np.int32), np.array(returns, np.float32)
        #print(returns)
        # TODO: Train the policy network using policy gradient theorem
        # and the value network using MSE.
        self._policy.train_on_batch(states, actions, sample_weight = returns - self.predict_values(states))
        self._value.train_on_batch(states, returns)


    def predict_actions(self, states):
        states = np.array(states, np.float32)
        return self._policy.predict_on_batch(states)

    def predict_values(self, states):
        states = np.array(states, np.float32)
        return self._value.predict_on_batch(states)[:, 0]

if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1", type=str, help="Environment.")
    parser.add_argument("--evaluate_each", default=100, type=int, help="Evaluate each number of batches.")
    parser.add_argument("--evaluate_for", default=100, type=int, help="Evaluate for number of batches.")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layer", default=10, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=None, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--workers", default=2, type=int, help="Number of parallel workers.")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create the environment
    env = gym_evaluator.GymEnvironment(args.env)

    # Construct the network
    network = Network(env, args)
    # Initialize parallel workers by env.parallel_init
    states = env.parallel_init(args.workers)
    while True:
        # Training
        for i in range(args.evaluate_each):
            # TODO: Choose actions using network.predict_actions
            actions_dist = network.predict_actions(states)
            actions = [np.random.choice(env.actions, 1, p = dist.numpy())[0] for dist in actions_dist]

            # TODO: Perform steps by env.parallel_step
            steps = env.parallel_step(actions)

            # TODO: Compute return estimates by
            # - extracting next_states from steps
            # - computing value function approximation in next_states
            # - estimating returns by reward + (0 if done else args.gamma * next_state_value)
            next_states = [step[0] for step in steps]
            next_states_value = network.predict_values(next_states)
            returns = [step[1] + (not step[2]) * (args.gamma * next_state_value) \
                for step, next_state_value in zip(steps, next_states_value)] 

            # TODO: Train network using current states, chosen actions and estimated returns
            network.train(states, actions, returns)
            states = next_states


        # Periodic evaluation
        returns = []
        for _ in range(args.evaluate_for):
            returns.append(0)
            state, done = env.reset(), False
            while not done:
                if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                    env.render()

                probabilities = network.predict_actions([state])[0]
                action = np.argmax(probabilities)
                state, reward, done, _ = env.step(action)
                returns[-1] += reward
        print("Evaluation of {} episodes: {}".format(args.evaluate_for, np.mean(returns)))

        if np.mean(returns) > 450:
            break

    # On the end perform final evaluations with `env.reset(True)`
    # Periodic evaluation
    returns = []
    for _ in range(args.evaluate_for):
        returns.append(0)
        state, done = env.reset(True), False
        while not done:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()

            probabilities = network.predict_actions([state])[0]
            action = np.argmax(probabilities)
            state, reward, done, _ = env.step(action)
            returns[-1] += reward
    print("Evaluation of {} episodes: {}".format(args.evaluate_for, np.mean(returns)))
