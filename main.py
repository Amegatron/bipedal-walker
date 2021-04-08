import random
import tensorflow as tf
import numpy as np
import gym
from collections import deque
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import threading as th
import urllib.parse


class ServerThread(th.Thread):
    def run(self) -> None:
        print("Starting server ...")
        httpd = ThreadingHTTPServer(('', 8123), MyHandler)
        httpd.serve_forever()
        print("Server started.")


class MyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global epsilon, epsilon_mode, epsilon_max, epsilon_min, \
            do_render, learning, epsilon_decay

        parts = urllib.parse.parse_qs(self.path[2:])
        if parts['action'][0] == 'epsilon':
            epsilon = float(parts['value'][0])
            print("Epsilon set to", epsilon)
        elif parts['action'][0] == 'epsilon_mode':
            epsilon_mode = int(parts['value'][0])
            print("Epsilon mode set to", epsilon_mode)
        elif parts['action'][0] == 'epsilon_max':
            epsilon_max = float(parts['value'][0])
            print("Epsilon max set to", epsilon_max)
        elif parts['action'][0] == 'epsilon_min':
            epsilon_min = float(parts['value'][0])
            print('Epsilon min set to', epsilon_min)
        elif parts['action'][0] == 'epsilon_decay':
            epsilon_decay = float(parts['value'][0])
            print('Epsilon decay set to', epsilon_decay)
        elif parts['action'][0] == 'render':
            val = int(parts['value'][0])
            if val == 1:
                do_render = True
            else:
                do_render = False

            print("Rendering set to", do_render)
        elif parts['action'][0] == 'learn':
            val = int(parts['value'][0])
            if val == 1:
                learning = True
            else:
                learning = False

            print("Learning set to", learning)


        return


class StateLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.surface_input = tf.keras.layers.Conv1D(3, 3, 1, input_shape=(None, 10, 1), activation="relu")
        self.surface_combined = tf.keras.layers.MaxPool1D(2)
        self.robot_input = tf.keras.layers.Dense(20, activation="relu")

    def call(self, inputs, **kwargs):
        robot_inputs, surface_inputs = tf.split(inputs, [14, 10], 1)
        surface_inputs = tf.reshape(surface_inputs, (-1, 10, 1))

        robot_input = self.robot_input(robot_inputs)
        surface_input = tf.reshape(self.surface_combined(self.surface_input(surface_inputs)), (-1, 12))

        return tf.concat([robot_input, surface_input], 1)


class ActorNetwork(tf.keras.Model):
    def __init__(self, learning_rate=0.001):
        super().__init__()

        self.layer0 = StateLayer()
        self.layer1 = tf.keras.layers.Dense(300, activation="relu", dtype=tf.float32)
        self.layer2 = tf.keras.layers.Dense(200, activation="relu", dtype=tf.float32)
        self.output_layer = tf.keras.layers.Dense(4, activation="tanh", dtype=tf.float32)
        self.optimizer = tf.optimizers.Adam(learning_rate)

    def call(self, inputs, training=None, mask=None):
        return self.output_layer(self.layer2(self.layer1(self.layer0(inputs))))


class CriticNetwork(tf.keras.Model):
    def __init__(self, learning_rate=0.002):
        super().__init__()

        self.learning_rate = learning_rate
        self.state_layer = StateLayer()
        self.layer1 = tf.keras.layers.Dense(300, activation="relu", dtype=tf.float32)
        self.layer2 = tf.keras.layers.Dense(200, activation="relu", dtype=tf.float32)
        self.output_layer = tf.keras.layers.Dense(1, activation="linear", dtype=tf.float32)
        self.optimizer = tf.optimizers.Adam(learning_rate)

    def call(self, inputs, training=None, mask=None):
        states, actions = tf.split(inputs, [24, 4], 1)

        return self.output_layer(
            self.layer2(
                self.layer1(
                    tf.concat(
                        [
                            self.state_layer(states),
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
        q_ = tf.squeeze(critic_target(tf.concat([next_states, target_actions], 1)), 1)
        q = tf.squeeze(critic(tf.concat([states, actions], 1)), 1)
        target = rewards + gamma * q_ * (1 - dones)
        loss = tf.losses.mean_squared_error(target, q)

    gradients = tape.gradient(loss, critic.trainable_variables)
    critic.optimizer.apply_gradients(zip(gradients, critic.trainable_variables))

    with tf.GradientTape() as tape:
        next_actions = actor(states)
        loss = tf.math.reduce_mean(-critic(tf.concat([states, next_actions], 1)))

    gradients = tape.gradient(loss, actor.trainable_variables)
    actor.optimizer.apply_gradients(zip(gradients, actor.trainable_variables))


def soft_copy_weights(source_model, target_model, tau=0.003):
    source_weights = source_model.get_weights()
    target_weights = target_model.get_weights()
    weights = []
    for i, weight in enumerate(source_weights):
        weights.append(weight * tau + target_weights[i] * (1 - tau))
    target_model.set_weights(weights)


def transfer_networks(tau=0.003):
    soft_copy_weights(actor, actor_target, tau)
    soft_copy_weights(critic, critic_target, tau)


def save_networks(key):
    actor.save_weights('./saves/actor_' + key)
    actor_target.save_weights('./saves/actor_target_' + key)
    critic.save_weights('./saves/critic_' + key)
    critic_target.save_weights('./saves/critic_target_' + key)


def load_networks(key):
    actor.load_weights('./saves/actor_' + key)
    actor_target.load_weights('./saves/actor_target_' + key)
    critic.load_weights('./saves/critic_' + key)
    critic_target.load_weights('./saves/critic_target_' + key)


server_thread = ServerThread()
server_thread.start()

# 0 - dynamic, 1 - static, 2 - 0
epsilon_mode = 0
do_render = True
learning = False

env = gym.make("BipedalWalker-v3")
actor = ActorNetwork()
actor_target = ActorNetwork()
critic = CriticNetwork()
critic_target = CriticNetwork()

#
# Either load or transfer
#
# transfer_networks(1)
load_networks('2088_263.56')


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
maximum_reward = -1000

while True:
    episode += 1
    state = env.reset()
    episode_reward = 0

    for step in range(max_timesteps):
        stateTensor = tf.reshape(state, [1, -1])

        actions = actor(stateTensor)

        if learning and epsilon_mode != 2:
            actions += tf.random.normal(shape=[4], mean=0.0, stddev=epsilon)
            actions = tf.clip_by_value(actions, -1, 1)

        action = actions[0]

        next_state, reward, done, _ = env.step(action)

        replay_buffer.append(
            (state.astype("float32"), action, float(reward), next_state.astype("float32"), float(done))
        )
        state = next_state

        episode_reward += reward

        if do_render:
            env.render()

        if learning and len(replay_buffer) >= 1000:
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
            if epsilon_mode == 0:
                epsilon *= epsilon_decay
                if epsilon < epsilon_min:
                    epsilon = epsilon_max

            rewards_history.append(episode_reward)
            if len(rewards_history) >= 10:
                average_result = np.mean(rewards_history[-10:])

            if episode_reward > maximum_reward:
                maximum_reward = episode_reward
                print("New record!", episode_reward)

            if average_result > maximum_average_reward:
                print("New average record!", average_result)
                if average_result > 0 and learning:
                    save_networks(str(episode) + '_' + str(round(average_result, 2)))

                maximum_average_reward = average_result

            if episode % 5 == 0:
                print("Average reward: ", average_result, ". ", episode, " episodes have passed so far")
                print("Epsilon: ", epsilon)

            if episode % 100 == 0 and episode > 300 and learning:
                save_networks(str(episode))

            if episode_reward >= 300:
                print("Solved!!!!!")

            break
