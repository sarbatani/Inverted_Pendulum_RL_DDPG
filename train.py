
import numpy as np
import gym
import gym_singlePendulum

from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import GaussianWhiteNoiseProcess, OrnsteinUhlenbeckProcess
from rl.callbacks import FileLogger
from keras.callbacks import ModelCheckpoint
from network import create_networks

import os

# Disable cuda if your GPU device is not fast
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def train(session_name):
    kwargs = {'viewer': False}
    # Get the environment and extract the number of actions.
    ENV_NAME = 'singlePendulum-v0'

    env = gym.make(ENV_NAME, **kwargs)
    np.random.seed(7)
    env.seed(7)

    actor, critic, action_input = create_networks(env)

    assert len(env.action_space.shape) == 1

    nb_actions = env.action_space.shape[0]

    # Logger callback
    cb = FileLogger(filepath='logs/ddpg_{}_{}.log'.format(ENV_NAME, session_name), interval=1)

    # model checkpoint callback
    checkpoint_filepath = 'checkpoint/ddpg_{}_{}_weights.h5f'.format(ENV_NAME, session_name)
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='episode_reward',
        mode='max',
        verbose=1,
        save_best_only=True)

    # Experience memory
    memory = SequentialMemory(limit=200000, window_length=1)

    # Random process
    random_process = GaussianWhiteNoiseProcess()
    # random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)

    # Create DDPG agent
    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                      batch_size=100,
                      train_interval=1, memory_interval=1, memory=memory, nb_steps_warmup_critic=10000,
                      nb_steps_warmup_actor=10000,
                      random_process=random_process, gamma=.99, target_model_update=1e-3)
    agent.compile(Adam(learning_rate=.0005, clipnorm=1., epsilon=1.e-7, beta_1=0.9, beta_2=0.999), metrics=['mae'])

    # Train the agent
    agent.fit(env, nb_steps=200000, visualize=False, verbose=2, nb_max_episode_steps=400, callbacks=[cb])

    # After training is done, we save the final weights.
    agent.save_weights(filepath='ddpg_{}_{}_weights.h5f'.format(ENV_NAME, session_name), overwrite=True)

    # Visualize the training metrics
    os.system('python visualize_log.py logs/ddpg_{}_{}.log'.format(ENV_NAME, session_name))

    env.close()


if __name__== "__main__":
    session_name = input("Enter session name: ")
    train(session_name)