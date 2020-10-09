from gym.envs.registration import register

register(
    id='singlePendulum-v0',
    entry_point='gym_singlePendulum.envs:singlePendulumEnv',
)