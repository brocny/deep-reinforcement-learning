#!/usr/bin/env python3

#dd7e3410-38c0-11e8-9b58-00505601122b
#6e14ef6b-3281-11e8-9de3-00505601122b

import collections

import numpy as np
import tensorflow as tf

from os import path

import cart_pole_evaluator
import embedded_data

class Network:
    def __init__(self, env, args):
        # TODO: Create a suitable network
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(args.hidden_layer_size, activation=tf.nn.tanh, input_shape=env.state_shape),
            tf.keras.layers.Dense(env.actions, activation=None)
        ])

        self.model.compile(
            loss = tf.losses.MeanSquaredError(),
            optimizer = tf.optimizers.Adam(args.learning_rate),
            experimental_run_tf_function=False
        )

        # Warning: If you plan to use Keras `.train_on_batch` and/or `.predict_on_batch`
        # methods, pass `experimental_run_tf_function=False` to compile. There is
        # a bug in TF 2.0 which causes the `*_on_batch` methods not to use `tf.function`.

        # Otherwise, if you are training manually, using `tf.function` is a good idea
        # to get good performance.

    # Define a training method. Generally you have two possibilities
    # - pass new q_values of all actions for a given state; all but one are the same as before
    # - pass only one new q_value for a given state, including the index of the action to which
    #   the new q_value belongs
    def train(self, states, targets, actions, clip=8):
        target_qs = self.model.predict_on_batch(states)
        for i in range(len(targets)):
            diff = np.clip(targets[i] - target_qs[i, actions[i]], -clip, clip)
            target_qs[i, actions[i]] += diff
        self.model.train_on_batch(states, target_qs)

    def predict(self, states):
        return self.model.predict_on_batch(states)

if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--episodes", default=1200, type=int, help="Episodes for epsilon decay.")
    parser.add_argument("--epsilon", default=0.3, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.03, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=1, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layers", default=1, type=int, help="Number of hidden layers.")
    parser.add_argument("--hidden_layer_size", default=32, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=0.002, type=float, help="Learning rate.")
    parser.add_argument("--learning_rate_final", default=0.0004, type=float, help="Final learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args([])

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create the environment
    env = cart_pole_evaluator.environment(discrete=False)

    # Construct the network
    network = Network(env, args)
    if path.exists('05/model.h5'):
        network.model = tf.keras.models.load_model('05/model.h5')
        training = False
    else:
        target_network = Network(env, args)
        target_network.model.set_weights(network.model.get_weights())
        training = True
    # Replay memory; maxlen parameter can be passed to deque for a size limit,
    # which we however do not need in this simple task.
    replay_buffer = collections.deque(maxlen=75_000)
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    epsilon = args.epsilon
    steps = 0
    while training:
        # Perform episode
        state, done = env.reset(), False
        while not done:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()
            
            # TODO: compute action using epsilon-greedy policy. You can compute
            # the q_values of a given state using
            #   q_values = network.predict(np.array([state], np.float32))[0]
            q_values = network.predict(np.array([state], np.float32))[0]
            action = q_values.argmax() if np.random.uniform() > epsilon else np.random.choice(env.actions)
            next_state, reward, done, _ = env.step(action)

            # Append state, action, reward, done and next_state to replay_buffer
            replay_buffer.append(Transition(state, action, reward, done, next_state))

            # TODO: If the replay_buffer is large enough, preform a training batch
            # of `args.batch_size` uniformly randomly chosen transitions.
            #
            # After you choose `states` and suitable targets, you can train the network as
            #   network.train(states, ...)
            if steps > 4000 and steps % 1000 == 0:
                print(steps)
                target_network.model.set_weights(network.model.get_weights())

            if len(replay_buffer) >= 2000:
                batch_indices = np.random.choice(len(replay_buffer), args.batch_size)
                targets, states, actions = [], [], []
                for i in batch_indices:
                    t = replay_buffer[i]
                    s = t.state
                    a = t.action
                    target = t.reward
                    if not t.done:
                        target += args.gamma * np.max(target_network.predict(np.array([t.next_state], np.float32))[0])
                    targets.append(target)
                    actions.append(a)
                    states.append(s)

                network.train(np.array(states), targets, actions)
            state = next_state
            steps += 1

        if args.epsilon_final:
            epsilon = np.exp(np.interp(env.episode + 1, [0, args.episodes], [np.log(args.epsilon), np.log(args.epsilon_final)]))
        if args.learning_rate_final:
            network.model.optimizer.lr = np.exp(np.interp(env.episode + 1, [0, args.episodes], [np.log(args.learning_rate), np.log(args.learning_rate_final)]))
        if env.episode > 1500:
            epsilon = 0

    # Final evaluation
    while True:
        state, done = env.reset(True), False
        while not done:
            action = np.argmax(network.predict(np.array([state], np.float32))[0])
            state, reward, done, _ = env.step(action)
