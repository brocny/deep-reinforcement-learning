#!/usr/bin/env python3
import collections

import numpy as np
import tensorflow as tf
import time

import gym_evaluator

class Network:
    def __init__(self, env, args):
        assert len(env.action_shape) == 1
        action_components = env.action_shape[0]
        action_lows, action_highs = map(np.array, env.action_ranges)

        self.tau = args.target_tau
        self.gamma = args.gamma

        # TODO: Create `actor` network, starting with `inputs` and returning
        # `action_components` values for each batch example. Usually, one
        # or two hidden layers are employed. Each `action_component[i]` should
        # be mapped to range `[actions_lows[i]..action_highs[i]]`, for example
        # using `tf.nn.sigmoid` and suitable rescaling.
        #
        # Then, create a target actor as a copy of the model using
        # `tf.keras.models.clone_model`.
        inp = tf.keras.layers.Input(env.state_shape)

        hidden = tf.keras.layers.Dense(80, activation=tf.nn.relu)(inp)
        hidden = tf.keras.layers.Dense(80, activation=tf.nn.relu)(hidden)
        hidden = tf.keras.layers.Dense(action_components, activation=tf.nn.sigmoid)(hidden)
        out_actor = tf.add(action_lows, tf.multiply(hidden, action_highs - action_lows))

        self.actor_model = tf.keras.Model(inputs=inp, outputs=out_actor)
        self.actor_target = tf.keras.models.clone_model(self.actor_model)

        # TODO: Create `critic` network, starting with `inputs` and `actions`
        # and producing a vector of predicted returns. Usually, `inputs` are fed
        # through a hidden layer first, and then concatenated with `actions` and fed
        # through two more hidden layers, before computing the returns.
        #
        # Then, create a target critic as a copy of the model using `tf.keras.models.clone_model`.
        
        action_inp = tf.keras.layers.Input(env.action_shape)

        hidden = tf.keras.layers.Dense(100, activation=tf.nn.relu)(inp)        
        hidden = tf.keras.layers.Concatenate()([action_inp, hidden])
        hidden = tf.keras.layers.Dense(200, activation=tf.nn.relu)(hidden)
        hidden = tf.keras.layers.Dense(200, activation=tf.nn.relu)(hidden)
        critic_out = tf.keras.layers.Dense(1, activation=None)(hidden)

        self.critic_model = tf.keras.Model(inputs=[inp, action_inp], outputs=critic_out)
        self.critic_target = tf.keras.models.clone_model(self.critic_model)

        self.actor_optimizer = tf.optimizers.Adam(args.learning_rate)
        self.critic_optimizer = tf.optimizers.Adam(5 * args.learning_rate)

    @tf.function
    def _train(self, states, actions, returns, next_states, dones):
        # TODO: Train separately the actor and critic.
        #
        # Furthermore, update the weights of the target actor and critic networks
        # by using args.target_tau option.
        
        # Actor training
        with tf.GradientTape() as tape:
            a = self._predict_actions(states)
            actor_loss = -tf.reduce_mean(self.critic_model([states, a]))
        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)

        # Critic training 
        with tf.GradientTape() as tape:
            values = self._predict_values(next_states)
            target_Q = returns + values * (1 - dones) * self.gamma
            target_Q = tf.stop_gradient(target_Q)
            current_Q = self.critic_model([states, actions])
            td_errors = target_Q - current_Q
        
        critic_loss = tf.reduce_mean(td_errors)
        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        
    def train(self, states, actions, returns, next_states, dones):
        states, actions, returns = np.array(states, np.float32), np.array(actions, np.float32), np.array(returns, np.float32)
        next_states, dones = np.array(next_states, np.float32), np.array(dones, np.float32)
        self._train(states, actions, returns, next_states, dones)

        self.critic_target.set_weights(np.array(self.critic_model.weights) * self.tau + np.array(self.critic_target.weights) * (1 - self.tau))
        self.actor_target.set_weights(np.array(self.actor_model.weights) * self.tau + np.array(self.actor_target.weights) * (1 - self.tau))

    @tf.function
    def _predict_actions(self, states):
        # TODO: Compute actions by the actor
        return self.actor_model(states)

    def predict_actions(self, states):
        states = np.array(states, np.float32)
        return self._predict_actions(states).numpy()

    @tf.function
    def _predict_values(self, states):
        # TODO: Predict actions by the target actor and evaluate them using
        # target_critic.
        actions = self.actor_target(states)
        return self.critic_target([states, actions])

    def predict_values(self, states):
        states = np.array(states, np.float32)
        return self._predict_values(states).numpy()[:, 0]


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, shape, mu, theta, sigma):
        self.mu = mu * np.ones(shape)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        self.state += self.theta * (self.mu - self.state) + np.random.normal(scale=self.sigma, size=self.state.shape)
        return self.state


if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--env", default="Pendulum-v0", type=str, help="Environment.")
    parser.add_argument("--evaluate_each", default=50, type=int, help="Evaluate each number of episodes.")
    parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate for number of batches.")
    parser.add_argument("--noise_sigma", default=0.2, type=float, help="UB noise sigma.")
    parser.add_argument("--noise_theta", default=0.15, type=float, help="UB noise theta.")
    parser.add_argument("--gamma", default=0.98, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layer", default=None, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--target_tau", default=0.001, type=float, help="Target network update weight.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create the environment
    env = gym_evaluator.GymEnvironment(args.env)
    action_lows, action_highs = map(np.array, env.action_ranges)

    # Construct the network
    network = Network(env, args)

    # Replay memory; maxlen parameter can be passed to deque for a size limit,
    # which we however do not need in this simple task.
    replay_buffer = collections.deque()
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    noise = OrnsteinUhlenbeckNoise(env.action_shape[0], 0., args.noise_theta, args.noise_sigma)
    while True:
        # Training
        for _ in range(args.evaluate_each):
            state, done = env.reset(), False
            noise.reset()
            while not done:
                # TODO: Perform an action and store the transition in the replay buffer
                action = tf.clip_by_value(network.predict_actions([state])[0] + noise.sample(), action_lows, action_highs)
                next_state, reward, done, _ = env.step(action)
                replay_buffer.append(Transition(state, action, reward, done, next_state))
                state = next_state

                if len(replay_buffer) % 10000 == 0:
                    print(f'** Performed {len(replay_buffer)} steps. **')

                # If the replay_buffer is large enough, perform training
                if len(replay_buffer) >= args.batch_size:
                    batch = np.random.choice(len(replay_buffer), size=args.batch_size, replace=False)
                    states, actions, rewards, dones, next_states = zip(*[replay_buffer[i] for i in batch])
                    network.train(states, actions, rewards, next_states, dones)

        # Periodic evaluation
        returns = []
        for _ in range(args.evaluate_for):
            returns.append(0)
            state, done = env.reset(), False
            while not done:
                if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                    env.render()

                action = network.predict_actions([state])[0]
                state, reward, done, _ = env.step(action)
                returns[-1] += reward
        print("Evaluation of {} episodes: {}".format(args.evaluate_for, np.mean(returns)))
        if np.mean(returns) > -185:
            network.actor_model.save(f'actor-{time.strftime("%d-%m-%H:%M")}.h5')
            network.critic_model.save(f'critic-{time.strftime("%d-%m-%H:%M")}.h5')
            print('Saved trained model')
            break

    # On the end perform final evaluations with `env.reset(True)`
    while True:
        state, done = env.reset(True), False
        while not done:
            action = network.predict_actions([state])[0]
            state, reward, done, _ = env.step(action)
