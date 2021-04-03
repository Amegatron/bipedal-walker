import random
import tensorflow as tf
import numpy as np
import gym
from collections import deque


class ActorNetwork(tf.Module):
    def __init__(self, learning_rate=0.001):
        super().__init__()

        self.layer1 = tf.keras.layers.Dense(400, activation="relu", dtype=tf.float32)
        self.layer2 = tf.keras.layers.Dense(300, activation="relu", dtype=tf.float32)
        self.output_layer = tf.keras.layers.Dense(4, activation="tanh", dtype=tf.float32)
        self.optimizer = tf.optimizers.Adam(learning_rate)

    def __call__(self, states):
        return self.output_layer(self.layer2(self.layer1(states)))


class CriticNetwork(tf.Module):
    def __init__(self, learning_rate=0.002):
        super().__init__()

        self.learning_rate = learning_rate
        self.layer1 = tf.keras.layers.Dense(400, activation="relu", dtype=tf.float32)
        self.layer2 = tf.keras.layers.Dense(300, activation="relu", dtype=tf.float32)
        self.output_layer = tf.keras.layers.Dense(1, activation="linear", dtype=tf.float32)
        self.optimizer = tf.optimizers.Adam(learning_rate)

    def __call__(self, states, actions):
        return self.output_layer(
            self.layer2(
                self.layer1(
                    tf.concat(
                        [
                            states,
                            actions,
                        ],
                        1
                    ),
                )
            )
        )


@tf.function
def train(states, actions, rewards, next_states, dones):
    with tf.GradientTape() as tape:
        target_actions = actor_target(next_states)
        q_ = tf.squeeze(critic_target(next_states, target_actions), 1)
        q = tf.squeeze(critic(states, actions), 1)
        target = rewards + gamma * q_ * (1 - dones)
        loss = tf.losses.mean_squared_error(target, q)

    gradients = tape.gradient(loss, critic.trainable_variables)
    critic.optimizer.apply_gradients(zip(gradients, critic.trainable_variables))

    with tf.GradientTape() as tape:
        next_actions = actor(states)
        loss = tf.math.reduce_mean(-critic(states, next_actions))

    gradients = tape.gradient(loss, actor.trainable_variables)
    actor.optimizer.apply_gradients(zip(gradients, actor.trainable_variables))


def soft_copy_weights(source_layer, target_layer, tau=0.003):
    source_weights = source_layer.get_weights()
    target_weights = target_layer.get_weights()
    weights = []
    for i, weight in enumerate(source_weights):
        weights.append(weight * tau + target_weights[i] * (1 - tau))
    target_layer.set_weights(weights)


def transfer_networks(tau=0.003):
    soft_copy_weights(critic.layer1, critic_target.layer1, tau)
    soft_copy_weights(critic.layer2, critic_target.layer2, tau)
    soft_copy_weights(critic.output_layer, critic_target.output_layer, tau)

    soft_copy_weights(actor.layer1, actor_target.layer1, tau)
    soft_copy_weights(actor.layer2, actor_target.layer2, tau)
    soft_copy_weights(actor.output_layer, actor_target.output_layer, tau)


env = gym.make("BipedalWalker-v3")
actor = ActorNetwork()
actor_target = ActorNetwork()
critic = CriticNetwork()
critic_target = CriticNetwork()

transfer_networks(1)

gamma = 0.99  # Discount coefficient
max_timesteps = 2000
iterations = 300
batch_size = 64
log_episodes_every = 5

replay_buffer = deque(maxlen=500000)
epsilon_max = 0.2
epsilon = epsilon_max
epsilon_decay = 0.993
epsilon_min = 0.1

episode = 0
rewards_history = []
maximum_average_reward = -1000
average_result = -1000

while True:
    episode += 1
    state = env.reset()
    episode_reward = 0

    for step in range(max_timesteps):
        stateTensor = tf.reshape(state, [1, -1])

        actions = actor(stateTensor)
        actions += tf.random.normal(shape=[4], mean=0.0, stddev=epsilon)
        actions = tf.clip_by_value(actions, -1, 1)
        action = actions[0]

        next_state, reward, done, _ = env.step(action)

        replay_buffer.append(
            (state.astype("float32"), action, float(reward), next_state.astype("float32"), float(done))
        )
        state = next_state

        episode_reward += reward

        if episode % 5 == 0:
            env.render()

        if len(replay_buffer) >= 1000:
            #
            #   LEARN
            #

            samples = random.sample(replay_buffer, batch_size)
            samplesReshaped = list(zip(*samples))
            states = tf.convert_to_tensor(samplesReshaped[0])
            actions = tf.convert_to_tensor(samplesReshaped[1])
            rewards = tf.convert_to_tensor(samplesReshaped[2])
            next_states = tf.convert_to_tensor(samplesReshaped[3])
            dones = tf.convert_to_tensor(samplesReshaped[4])

            train(
                states,
                actions,
                rewards,
                next_states,
                dones
            )

            transfer_networks()

            #
            #   END LEARN
            #

        if done:
            epsilon *= epsilon_decay
            if epsilon < epsilon_min:
                epsilon = epsilon_max

            rewards_history.append(episode_reward)
            if len(rewards_history) >= 20:
                average_result = np.mean(rewards_history[-20:])

            if average_result > maximum_average_reward:
                print("New record!")
                maximum_average_reward = average_result

            if episode % 5 == 0:
                print("Average reward: ", average_result, ". ", episode, " episodes have passed so far");
                print("Epsilon: ", epsilon)

            if episode_reward >= 300:
                print("Solved!!!!!")

            break
