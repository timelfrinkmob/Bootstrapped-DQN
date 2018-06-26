from gym_chain.envs.chain_env import ChainEnv
from gym.envs.registration import register

register(
    id='Chainbla-v0',
    entry_point='chain_env.chain_env:Chainbla',
)
