
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
import gym


def create_actor(env):
    nb_actions = env.action_space.shape[0]
    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    actor.add(Dense(32))
    actor.add(Activation('relu'))
    actor.add(Dense(32))
    actor.add(Activation('relu'))
    actor.add(Dense(32))
    actor.add(Activation('relu'))
    actor.add(Dense(32))
    actor.add(Activation('relu'))
    actor.add(Dense(32))
    actor.add(Activation('relu'))
    actor.add(Dense(nb_actions))
    actor.add(Activation('tanh'))
    return actor


def create_critic(env):
    nb_actions = env.action_space.shape[0]
    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    return critic, action_input


def create_networks(env):
    actor = create_actor(env)
    critic, action_input = create_critic(env)
    print(actor.summary())
    print(critic.summary())
    return actor, critic, action_input
