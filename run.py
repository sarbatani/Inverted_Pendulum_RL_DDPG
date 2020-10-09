
import numpy as np
import gym
import gym_singlePendulum

from keras.optimizers import Adam
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory

from network import create_networks


def visualize(session_name):
    kwargs = {'viewer': True}

    ENV_NAME = 'singlePendulum-v0'
    env = gym.make(ENV_NAME, **kwargs)
    np.random.seed(7)
    env.seed(7)
    assert len(env.action_space.shape) == 1
    nb_actions = env.action_space.shape[0]

    actor, critic, action_input = create_networks(env)

    memory = SequentialMemory(limit=400, window_length=1)
    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory)
    agent.compile(Adam(lr=.0005, clipnorm=1., epsilon=1.e-7, beta_1=0.9, beta_2=0.999), metrics=['mae'])

    checkpoint_filepath = 'checkpoint/ddpg_{}_{}_weights.h5f'.format(ENV_NAME, session_name)
    filepath = 'ddpg_{}_{}_weights.h5f'.format(ENV_NAME, session_name)
    agent.load_weights(filepath=filepath)

    env.viewer = True
    agent.test(env, nb_episodes=1, visualize=False, nb_max_episode_steps=400)
    env.close()


if __name__== "__main__":
    session_name = input("Enter session name: ")
    visualize(session_name)