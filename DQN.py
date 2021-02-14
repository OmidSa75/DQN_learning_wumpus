import time
import sys
import random
import numpy as np
import tensorflow as tf
from Environment import Env
from collections import deque
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.models import Sequential
import logging

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# file_handler = logging.FileHandler('info.log')
# formatter = logging.Formatter('%(asctime)s :: %(message)s')
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)


logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(message)s')

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

EPISODES = 5000
STATE_SIZE = 225
COPY_STEPS = 10


def create_model(input_size, output_size):
    inputs = tf.keras.Input(shape=input_size)
    x = layers.Conv2D(16, kernel_size=(3, 3), activation=tf.nn.relu, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, kernel_size=(3, 3), activation=tf.nn.relu, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(16, kernel_size=(3, 3), activation=tf.nn.relu, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(16, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same')(x)
    x = layers.BatchNormalization()(x)
    # x = layers.GlobalAvgPool2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation=tf.nn.relu)(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(output_size, activation='linear')(x)

    model = tf.keras.Model(inputs, outputs)
    model.build(input_shape=input_size)

    return model


class DQNAgent:
    def __init__(self):

        self.render = False
        self.load = False
        self.save_loc = './DQN'
        self.action_size = 4
        self.discount_factor = 0.99
        self.learning_rate = 0.1
        self.lr_decay = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.998
        self.epsilon_min = 0.01
        self.batch_size = 16
        self.train_start = 5000
        self.state_size = STATE_SIZE
        self.model = self.build_model()
        self.optimizer = Adam(learning_rate=self.learning_rate)
        self.target_model = self.build_model()
        self.update_target_model()
        self.memory = deque(maxlen=10000)

        if self.load:
            self.load_model()

    def build_model(self):
        # Neural Network for Deep Q-learning
        model = create_model((15, 15, 1), self.action_size)
        logging.info(model.summary())
        time.sleep(2)
        return model

    def predict(self, inputs):
        """
        Predicts the inputs by the NN model
        :param inputs: float32 datatype array
        :return: model predictions
        """
        output = self.model(inputs).numpy()
        return output

    def select_action(self, state):
        # select action using epsilon-greedy
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        else:
            state = np.expand_dims(state, axis=(0, 3))
            action = np.argmax(self.predict(state))
            return action

    def MEMORY(self, state, action, reward, next_state, goal):
        self.memory.append([state, action, reward, next_state, goal])

    def update_target_model(self):
        variables1 = self.target_model.trainable_variables
        variables2 = self.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())

    def train_replay(self):
        if len(self.memory) <= self.train_start:
            return 0

        ids = np.random.randint(low=0, high=len(self.memory), size=self.batch_size)
        zero_reward = np.zeros(self.batch_size)
        states = np.asarray([self.memory[i][0] for i in ids])
        states = np.expand_dims(states, axis=3)
        actions = np.asarray([self.memory[i][1] for i in ids])
        rewards = np.asarray([self.memory[i][2] for i in ids])
        next_states = np.asarray([self.memory[i][3] for i in ids])
        next_states = np.expand_dims(next_states, axis=3)
        ends = np.asarray([self.memory[i][4] for i in ids])
        next_values = np.max(self.target_model.predict(next_states), axis=1)
        actual_values = np.where(ends, zero_reward, rewards + self.discount_factor * next_values)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(self.model(states) * tf.one_hot(actions, 4),
                                                        axis=1)  # [2, 3, 1]  [[0 , 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0]]
            loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))

        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    # save the model which is under training
    def save_model(self):
        self.model.save_weights('model_1.h5')
        print("The Model Saved")

    # load the saved model
    def load_model(self):
        self.model.load_weights('model.h5')
        print("The Model loaded")

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

    def learning_rate_decay(self):
        lr = self.optimizer.lr.numpy()
        lr = max(self.lr_decay * lr, 0.0001)
        self.optimizer.lr.assign(lr)


def test(env, train_agent: DQNAgent):
    score = 0
    steps = 0
    state = env.reset()  # done
    check_list = env.check_if_reward(state)
    goal = check_list['if_goal']  #
    wumpus = check_list['if_wumpus']
    train_agent.epsilon = 0
    while not goal or not wumpus:
        action = train_agent.select_action(state)
        state, reward, goal, wumpus = env.step(action)
        steps += 1
        score += reward

    print("Testing steps: {} score: {} ".format(steps, score))


if __name__ == "__main__":

    # create environment
    env = Env()
    train_agent = DQNAgent()
    total_scores = np.empty(EPISODES)
    for e in range(EPISODES):
        train_agent.update_epsilon()

        state = env.reset()
        check_list = env.check_if_reward(state)
        goal = check_list['if_goal']  # done
        wumpus = check_list['if_wumpus']  # done

        losses = []
        score = 0  # done

        while (not goal) and (not wumpus):
            if train_agent.render:
                env.render()

            action = train_agent.select_action(state)  # done  8
            next_state, reward, goal, wumpus = env.step(action)  # done 10
            score += reward  # done 11
            state = ((state - np.mean(state)) / np.std(state))
            next_state = ((next_state - np.mean(next_state)) / np.std(next_state))

            train_agent.MEMORY(state, action, reward, next_state, goal or wumpus)  # done 16
            state = next_state.copy()

            loss = train_agent.train_replay()
            losses.append(loss)
            if goal or wumpus:
                break

        if e % COPY_STEPS == 0:
            train_agent.update_target_model()
            train_agent.learning_rate_decay()

        mean_loss = np.mean(losses)
        total_scores[e] = score
        avg_scores = total_scores[max(0, e - 100): (e + 1)].mean()
        logging.info(
            "episode: {:3}   \nepisode score: {:8.6}    \nepsilon {:.3}  \navg score (last 100): {:8.6}\nlosses: {:8.6} \n{}"
                .format(e, float(score), float(train_agent.epsilon), avg_scores, mean_loss, 80 * '*'), )

        # save the model every 100 episodes
        if e % 100 == 0:
            train_agent.save_model()
