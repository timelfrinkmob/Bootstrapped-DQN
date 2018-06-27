import gym

from model_gym import model, bootstrap_model
from baselines import deepq


def main():
    env = gym.make("MountainCar-v0")
    # Enabling layer_norm here is import for parameter space noise!
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.1,
        print_freq=1,
        bootstrap = False,
        noisy = False,
        greedy = False
    )


if __name__ == '__main__':
    main()
