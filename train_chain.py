import argparse
import gym
import numpy as np
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tempfile
import time

import baselines.common.tf_util as U

from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.common.misc_util import (
    boolean_flag,
    pickle_load,
    pretty_eta,
    relatively_safe_pickle_dump,
    set_global_seeds,
    RunningAvg,
    SimpleMonitor
)
from baselines.common.schedules import LinearSchedule, PiecewiseSchedule
# when updating this to non-deperecated ones, it is important to
# copy over LazyFrames
#from baselines.common.azure_utils import Container
from model_gym import model, bootstrap_model
import gym_chain


def parse_args():
    parser = argparse.ArgumentParser("DQN experiments for Atari games")
    # Environment
    parser.add_argument("--env", type=str, default="Chainbla-v0", help="name of the game")
    parser.add_argument("--seed", type=int, default=42, help="which seed to use")
    # Core DQN parameters
    parser.add_argument("--replay-buffer-size", type=int, default=int(0), help="replay buffer size")
    parser.add_argument("--n", type=int, default=int(10), help="length of chain")
    parser.add_argument("--episodes", type=int, default=int(2000), help="number of episodes")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for Adam optimizer")
    parser.add_argument("--num-steps", type=int, default=int(0), help="total number of steps to run the environment for")
    parser.add_argument("--batch-size", type=int, default=32, help="number of transitions to optimize at the same time")
    parser.add_argument("--learning-freq", type=int, default=1, help="number of iterations between every optimization step")
    parser.add_argument("--target-update-freq", type=int, default=1, help="number of iterations between every target network update")
    # Bells and whistles
    boolean_flag(parser, "noisy", default=False, help="whether or not to NoisyNetwork")
    boolean_flag(parser, "double-q", default=True, help="whether or not to use double q learning")
    boolean_flag(parser, "dueling", default=False, help="whether or not to use dueling model")
    boolean_flag(parser, "bootstrap", default=True, help="whether or not to use bootstrap model")
    boolean_flag(parser, "prioritized", default=False, help="whether or not to use prioritized replay buffer")
    boolean_flag(parser, "greedy", default=False, help="whether or not to use e-greedy")
    parser.add_argument("--eps", type=float, default=0.1, help="epsilon when e-greedy")
    parser.add_argument("--prioritized-alpha", type=float, default=0.6, help="alpha parameter for prioritized replay buffer")
    parser.add_argument("--prioritized-beta0", type=float, default=0.4, help="initial value of beta parameters for prioritized replay")
    parser.add_argument("--prioritized-eps", type=float, default=1e-6, help="eps parameter for prioritized replay buffer")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./logs", help="directory in which training state and model should be saved.")
    parser.add_argument("--save-azure-container", type=str, default=None,
                        help="It present data will saved/loaded from Azure. Should be in format ACCOUNT_NAME:ACCOUNT_KEY:CONTAINER")
    parser.add_argument("--save-freq", type=int, default=1e6, help="save model once every time this many iterations are completed")
    boolean_flag(parser, "load-on-start", default=True, help="if true and model was previously saved then training will be resumed")
    return parser.parse_args()


def make_env(game_name):
    env = gym.make(game_name)  # Already performs a frame-skip of 4 @ baselines.common.atari_wrappers_deprecated
    env.__init__(n=args.n)
    monitored_env = SimpleMonitor(env)  # puts rewards and number of steps in info, before environment is wrapped
    #env = wrap_dqn(monitored_env)  # applies a bunch of modification to simplify the observation space (downsample, make b/w)
    return env, monitored_env


def maybe_save_model(savedir, container, state, rewards, steps):
    """This function checkpoints the model and state of the training algorithm."""
    if savedir is None:
        return
    start_time = time.time()
    model_dir = "model-{}".format(state["num_iters"])
    U.save_state(os.path.join(savedir, model_dir, "saved"))
    if container is not None:
        container.put(os.path.join(savedir, model_dir), model_dir)
    relatively_safe_pickle_dump(state, os.path.join(savedir, 'training_state.pkl.zip'), compression=True)
    if container is not None:
        container.put(os.path.join(savedir, 'training_state.pkl.zip'), 'training_state.pkl.zip')
    relatively_safe_pickle_dump(state["monitor_state"], os.path.join(savedir, 'monitor_state.pkl'))
    if container is not None:
        container.put(os.path.join(savedir, 'monitor_state.pkl'), 'monitor_state.pkl')
    relatively_safe_pickle_dump(rewards, os.path.join(savedir, 'rewards.pkl'))
    if container is not None:
        container.put(os.path.join(savedir, 'rewards.pkl'), 'rewards.pkl')
    relatively_safe_pickle_dump(steps, os.path.join(savedir, 'steps.pkl'))
    if container is not None:
        container.put(os.path.join(savedir, 'steps.pkl'), 'steps.pkl')
    logger.log("Saved model in {} seconds\n".format(time.time() - start_time))


def maybe_load_model(savedir, container):
    """Load model if present at the specified path."""
    if savedir is None:
        return

    state_path = os.path.join(os.path.join(savedir, 'training_state.pkl.zip'))
    if container is not None:
        logger.log("Attempting to download model from Azure")
        found_model = container.get(savedir, 'training_state.pkl.zip')
    else:
        found_model = os.path.exists(state_path)
    if found_model:
        state = pickle_load(state_path, compression=True)
        model_dir = "model-{}".format(state["num_iters"])
        if container is not None:
            container.get(savedir, model_dir)
        U.load_state(os.path.join(savedir, model_dir, "saved"))
        logger.log("Loaded models checkpoint at {} iterations".format(state["num_iters"]))
        return state


if __name__ == '__main__':
    args = parse_args()
    args.num_steps = (args.n + 9) * (args.episodes + 2)
    args.replay_buffer_size = args.num_steps 
    
    
    # Parse savedir and azure container.
    savedir = "models/" + args.save_dir + "_" + args.env + "_" + str(args.seed) + "_" + str(args.bootstrap) + "_" + str(args.noisy) + "_" + str(args.greedy) + "_" + str(args.n)
    logger.configure(savedir,['json','stdout'])
    if args.save_azure_container is not None:
        account_name, account_key, container_name = args.save_azure_container.split(":")
        container = Container(account_name=account_name,
                              account_key=account_key,
                              container_name=container_name,
                              maybe_create=True)
        if savedir is None:
            # Careful! This will not get cleaned up. Docker spoils the developers.
            savedir = tempfile.TemporaryDirectory().name
    else:
        container = None
    # Create and seed the env.
    env, monitored_env = make_env(args.env)
    if args.seed > 0:
        set_global_seeds(args.seed)
        env.unwrapped.seed(args.seed)
        env.seed(args.seed)

    with U.make_session(120) as sess:
        # Create training graph and replay buffer
        if args.bootstrap :
            act, train, update_target, debug = deepq.build_train(
                make_obs_ph=lambda name: U.Uint8Input(env.observation_space.shape, name=name),
                q_func=bootstrap_model,
                bootstrap=args.bootstrap,
                num_actions=env.action_space.n,
                optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
                gamma=0.99,
                double_q=args.double_q,
                noisy=args.noisy
            )
        else:
            act, train, update_target, debug = deepq.build_train(
                make_obs_ph=lambda name: U.Uint8Input(env.observation_space.shape, name=name),
                q_func=model,
                num_actions=env.action_space.n,
                optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
                gamma=0.99,
                double_q=args.double_q,
                noisy=args.noisy
            )

        approximate_num_iters = args.num_steps / 4
        exploration = PiecewiseSchedule([
            (0, 1.0),
            (int(args.num_steps *0.1), 0.1) # (approximate_num_iters / 5, 0.01)
        ], outside_value=0.1)

        learning_rate = PiecewiseSchedule([
            (0, 1e-3),
            (1, 1e-3)
        ], outside_value=1e-3)
        
        if args.prioritized:
            replay_buffer = PrioritizedReplayBuffer(args.replay_buffer_size, args.prioritized_alpha)
            beta_schedule = LinearSchedule(approximate_num_iters, initial_p=args.prioritized_beta0, final_p=1.0)
        else:
            replay_buffer = ReplayBuffer(args.replay_buffer_size)

        U.initialize()
        update_target()
        num_iters = 0

        # Load the model
        state = maybe_load_model(savedir, container)
        if state is not None:
            num_iters, replay_buffer = state["num_iters"], state["replay_buffer"],
            monitored_env.set_state(state["monitor_state"])

        start_time, start_steps = None, None
        steps_per_iter = RunningAvg(0.999)
        iteration_time_est = RunningAvg(0.999)
        obs = env.reset()

        rewards_list = []
        episodes = 0
        rew = 0

        # Main training loop
        head = np.random.randint(10)        #Initial head initialisation
        while True:
            num_iters += 1
            # Take action and store transition in the replay buffer.
            if args.bootstrap:
                action = act(np.array(obs)[None], head=head, update_eps=exploration.value(num_iters))[0]
            elif args.noisy:
                action = act(np.array(obs)[None], stochastic=False)[0]
            else:
                action = act(np.array(obs)[None], update_eps=exploration.value(num_iters))[0]
            new_obs, n_rew, done, info = env.step(action)
            rew = rew + n_rew
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs
            if done:
                obs = env.reset()
                head = np.random.randint(10)

            if (num_iters > max(0.5 * args.batch_size, args.replay_buffer_size // 20) and
                    num_iters % args.learning_freq == 0):
                # Sample a bunch of transitions from replay buffer
                if args.prioritized:
                    experience = replay_buffer.sample(args.batch_size, beta=beta_schedule.value(num_iters))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(args.batch_size)
                    weights = np.ones_like(rewards)
                # Minimize the error in Bellman's equation and compute TD-error
                # Minimize the error in Bellman's equation and compute TD-error
                if args.bootstrap:
                    td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights,
                                      learning_rate.value(num_iters))
                else:
                    td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)                # Update the priorities in the replay buffer
                if args.prioritized:
                    new_priorities = np.abs(td_errors) + args.prioritized_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)
            # Update target network.
            if num_iters % args.target_update_freq == 0:
                update_target()

            if start_time is not None:
                steps_per_iter.update(num_iters - start_steps)
                iteration_time_est.update(time.time() - start_time)
            start_time, start_steps = time.time(), num_iters

            # Save the model and training state.
            if num_iters > 0 and (num_iters % args.save_freq == 0 or num_iters> args.num_steps):
                maybe_save_model(savedir, container, {
                    'replay_buffer': replay_buffer,
                    'num_iters': num_iters,
                    'monitor_state': monitored_env.get_state()
                }, rewards, num_iters)

            if num_iters> args.num_steps:
                break

            if done:
                episodes +=1
                steps_left = args.num_steps - num_iters
                completion = np.round(num_iters/ args.num_steps, 1)
                rewards_list.append(rew)
                logger.record_tabular("% completion", completion)
                logger.record_tabular("steps", num_iters)
                logger.record_tabular("episodes", episodes)
                logger.record_tabular("reward (100 epi mean)", np.mean(rewards_list[-100:]))
                logger.record_tabular("reward", rew)
                if args.bootstrap:
                    logger.record_tabular("head for episode", (head+1))
                if not args.noisy:
                    logger.record_tabular("exploration", exploration.value(num_iters))
                if args.prioritized:
                    logger.record_tabular("max priority", replay_buffer._max_priority)
                fps_estimate = (float(steps_per_iter) / (float(iteration_time_est) + 1e-6)
                                if steps_per_iter._value is not None else "calculating...")
                logger.dump_tabular()
                logger.log()
                logger.log("ETA: " + pretty_eta(int(steps_left / fps_estimate)))
                logger.log()
                rew = 0
