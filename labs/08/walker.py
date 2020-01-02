#!/usr/bin/env optirun python3.7

#dd7e3410-38c0-11e8-9b58-00505601122b
#6e14ef6b-3281-11e8-9de3-00505601122b

import collections

import numpy as np
import tensorflow as tf
import time

import gym_evaluator

class Network:
    def __init__(self, env, args):
        assert len(env.action_shape) == 1
        action_components = env.action_shape[0]
        self.action_lows, self.action_highs = map(np.array, env.action_ranges)

        self.tau = args.target_tau
        self.gamma = args.gamma

        self.i = -1

        # TODO: Create `actor` network, starting with `inputs` and returning
        # `action_components` values for each batch example. Usually, one
        # or two hidden layers are employed. Each `action_component[i]` should
        # be mapped to range `[actions_lows[i]..action_highs[i]]`, for example
        # using `tf.nn.sigmoid` and suitable rescaling.
        #
        # Then, create a target actor as a copy of the model using
        # `tf.keras.models.clone_model`.
        if not args.actor_path:
            inp = tf.keras.layers.Input(env.state_shape)

            hidden = tf.keras.layers.Dense(400, activation=tf.nn.relu)(inp)
            hidden = tf.keras.layers.Dense(300, activation=tf.nn.relu)(hidden)

            hidden = tf.keras.layers.Dense(action_components, activation=tf.nn.tanh)(hidden)
            out_actor = tf.multiply(hidden, self.action_highs)

            self.actor_model = tf.keras.Model(inputs=inp, outputs=out_actor)
            self.actor_target = tf.keras.models.clone_model(self.actor_model)
        else:
            import embedded_data
            self.actor_model = tf.keras.models.load_model(args.actor_path)
            if args.train:
                self.actor_target = tf.keras.models.clone_model(self.actor_model)

        # TODO: Create `critic` network, starting with `inputs` and `actions`
        # and producing a vector of predicted returns. Usually, `inputs` are fed
        # through a hidden layer first, and then concatenated with `actions` and fed
        # through two more hidden layers, before computing the returns.
        #
        # Then, create a target critic as a copy of the model using `tf.keras.models.clone_model`.
        if args.train:
            if args.critic_path1 and args.critic_path2:
                self.critic_model1 = tf.keras.models.load_model(args.critic_path1)
                self.critic_model2 = tf.keras.models.load_model(args.critic_path2)
                self.critic_target1 = tf.keras.models.clone_model(self.critic_model1)
                self.critic_target2 = tf.keras.models.clone_model(self.critic_model2)
            else: 
                self.critic_model1, self.critic_target1 = self.make_critic(env, args)
                self.critic_model2, self.critic_target2 = self.make_critic(env, args)
            
            self.actor_optimizer = tf.optimizers.Adam(args.learning_rate)
            self.critic_optimizer1 = tf.optimizers.Adam(args.learning_rate)
            self.critic_optimizer2 = tf.optimizers.Adam(args.learning_rate)
            

    def make_critic(self, env, args):
        inp = tf.keras.layers.Input(env.state_shape)
        action_inp = tf.keras.layers.Input(env.action_shape)

        hidden = tf.keras.layers.Concatenate()([inp, action_inp])
        
        hidden = tf.keras.layers.Dense(400, activation=tf.nn.relu)(hidden)
        hidden = tf.keras.layers.Dense(300, activation=tf.nn.relu)(hidden)

        critic_out = tf.keras.layers.Dense(1, activation=None)(hidden)
        critic_model = tf.keras.Model(inputs=[inp, action_inp], outputs=critic_out)
        critic_target = tf.keras.models.clone_model(critic_model)

        return critic_model, critic_target

    @tf.function
    def _train_critic(self, states, actions, returns, next_states, dones):
        # TODO: Train separately the actor and critic.
        #
        # Furthermore, update the weights of the target actor and critic networks
        # by using args.target_tau option.
        # Critic training 
        values = self._predict_values(next_states)[:, 0]
        target_Q = returns + values * (1 - dones) * self.gamma
        
        with tf.GradientTape() as tape1:
            current_Q1 = self.critic_model1([states, actions], training=True)[:, 0]
            critic_loss1 = tf.losses.mse(target_Q, current_Q1)
        critic_grad1 = tape1.gradient(critic_loss1, self.critic_model1.trainable_variables)
        self.critic_optimizer1.apply_gradients(zip(critic_grad1, self.critic_model1.trainable_variables))

        with tf.GradientTape() as tape2:
            current_Q2 = self.critic_model2([states, actions], training=True)[:, 0]
            critic_loss2 = tf.losses.mse(target_Q, current_Q2)
        critic_grad2 = tape2.gradient(critic_loss2, self.critic_model2.trainable_variables)
        self.critic_optimizer2.apply_gradients(zip(critic_grad2, self.critic_model2.trainable_variables))
        
    @tf.function
    def _train_actor(self, states):
        with tf.GradientTape() as tape:
            a = self.actor_model(states, training=True)
            actor_loss = -tf.reduce_mean(self.critic_model1([states, a], training=False))
        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))
        
    def train(self, states, actions, returns, next_states, dones):
        states, actions, returns = np.array(states, np.float32), np.array(actions, np.float32), np.array(returns, np.float32)
        next_states, dones = np.array(next_states, np.float32), np.array(dones, np.float32)
        self._train_critic(states, actions, returns, next_states, dones)
        
        self.i += 1
        if self.i % 2 == 0:
            self._train_actor(states)
            self.critic_target1.set_weights(np.array(self.critic_model1.weights) * self.tau + np.array(self.critic_target1.weights) * (1 - self.tau))
            self.critic_target2.set_weights(np.array(self.critic_model2.weights) * self.tau + np.array(self.critic_target2.weights) * (1 - self.tau))

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
        actions += tf.clip_by_value(np.random.normal(scale=0.15, size=actions.shape).astype(np.float32), -0.4, 0.4)
        actions = tf.clip_by_value(actions, self.action_lows, self.action_highs)
        return tf.minimum(self.critic_target1([states, actions]), self.critic_target2([states, actions]))

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
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
    parser.add_argument("--env", default="BipedalWalker-v2", type=str, help="Environment.")
    parser.add_argument("--evaluate_each", default=50, type=int, help="Evaluate each number of episodes.")
    parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate for number of batches.")
    parser.add_argument("--noise_sigma", default=0.2, type=float, help="UB noise sigma.")
    parser.add_argument("--noise_theta", default=0.15, type=float, help="UB noise theta.")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layer", default=100, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--target_tau", default=0.005, type=float, help="Target network update weight.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--train", action='store_true', help="Whether to train or just evaluate.")
    parser.add_argument("--critic_path1", type=str, default=None)
    parser.add_argument("--critic_path2", type=str, default=None)
    parser.add_argument("--actor_path", type=str, default='actor.h5')
    args = parser.parse_args([])

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
    ret_max = 0
    noise = OrnsteinUhlenbeckNoise(env.action_shape[0], 0., args.noise_theta, args.noise_sigma)
    while args.train:
        # Training
        for _ in range(args.evaluate_each):
            state, done = env.reset(), False
            noise.reset()
            while not done:
                # TODO: Perform an action and store the transition in the replay buffer
                #noise_sample = noise.sample()
                noise_sample = np.random.normal(scale=0.1, size=env.action_shape[0])
                action = np.clip(network.predict_actions([state])[0] + noise_sample, action_lows, action_highs)
                next_state, reward, done, _ = env.step(action)
                replay_buffer.append(Transition(state, action, reward, done, next_state))
                state = next_state

                if len(replay_buffer) % 10000 == 0:
                    print(f'** Performed {len(replay_buffer)} steps. **')
                if len(replay_buffer) > 3e6:
                    args.train = False
                    
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
        ret_mean = np.mean(returns)
        ret_std = np.std(returns)
        print(f"** Evaluation of {args.evaluate_for} episodes: {ret_mean} +-{ret_std} **")
        if ret_mean - ret_std > 250 and ret_mean - ret_std > ret_max:
            ret_max = ret_mean - ret_std
            network.actor_model.save(f'models/actor-{time.strftime("%d-%m-%H:%M")}-r{round(ret_mean)}_{round(ret_std)}.h5')
            network.critic_model1.save(f'models/critic1-{time.strftime("%d-%m-%H:%M")}-r{round(ret_mean)}_{round(ret_std)}.h5')
            network.critic_model2.save(f'models/critic2-{time.strftime("%d-%m-%H:%M")}-r{round(ret_mean)}_{round(ret_std)}.h5')
            print('Saved trained model')
            if ret_mean - 2 * ret_std > 360:
                break

    # On the end perform final evaluations with `env.reset(True)`
    while True:
        state, done = env.reset(True), False
        while not done:
            action = network.predict_actions([state])[0]
            state, reward, done, _ = env.step(action)
