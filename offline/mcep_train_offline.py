from jax.config import config
import os
from typing import Tuple

import datetime
import gym
import numpy as np
from tqdm import tqdm
import time
import absl
import sys
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter
from dataclasses import dataclass

import wrappers
from dataset_utils import D4RLDataset, split_into_trajectories
from evaluation import evaluate
from mcep_learner import MCEPLearner
import wandb
import warnings

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'halfcheetah-expert-v2', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_float('temp', 1.0, 'Loss temperature')
flags.DEFINE_boolean('double', True, 'Use double q-learning')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_integer('sample_random_times', 0, 'Number of random actions to add to smooth dataset')
flags.DEFINE_boolean('grad_pen', False, 'Add a gradient penalty to critic network')
flags.DEFINE_float('lambda_gp', 1, 'Gradient penalty coefficient')
flags.DEFINE_float('max_clip', 7., 'Loss clip value')
flags.DEFINE_integer('num_v_updates', 1, 'Number of value updates per iter')
flags.DEFINE_boolean('log_loss', False, 'Use log gumbel loss')

flags.DEFINE_float('beta', 2.0, 'beta for evaluation policy')
flags.DEFINE_float('beta_target', 2.0, 'beta for target policy')

flags.DEFINE_boolean('noise', False, 'Add noise to actions')
flags.DEFINE_float('noise_std', 0.1, 'Noise std for actions')

config_flags.DEFINE_config_file(
    'config',
    'default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)

# config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", True)


@dataclass(frozen=True)
class ConfigArgs:
    sample_random_times: int
    grad_pen: bool
    noise: bool
    noise_std: float
    lambda_gp: int
    max_clip: float
    num_v_updates: int
    log_loss: bool


def normalize(dataset):

    trajs = split_into_trajectories(dataset.observations, dataset.actions,
                                    dataset.rewards, dataset.masks,
                                    dataset.dones_float,
                                    dataset.next_observations)

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)

    dataset.rewards /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset.rewards *= 1000.0


def make_env_and_dataset(env_name: str,
                         seed: int) -> Tuple[gym.Env, D4RLDataset]:
    env = gym.make(env_name)

    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    dataset = D4RLDataset(env)

    if 'antmaze' in FLAGS.env_name:
        dataset.rewards -= 1.0
        # dataset.rewards = (dataset.rewards - 0.5) * 4
        # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
        # but I found no difference between (x - 0.5) * 4 and x - 1.0
    elif ('halfcheetah' in FLAGS.env_name or 'walker2d' in FLAGS.env_name
          or 'hopper' in FLAGS.env_name):
        normalize(dataset)

    return env, dataset


def main(_):
    # tags = FLAGS.save_dir.split('/')[-1]
    wandb.init(project="extQL", group=FLAGS.env_name+'_mcep_'+str(FLAGS.beta)+"_"+str(FLAGS.beta_target), name=str(FLAGS.seed), sync_tensorboard=True,
               reinit=True, settings=wandb.Settings(_disable_stats=True))
    wandb.config.update(flags.FLAGS)

    ts_str = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(FLAGS.save_dir, ts_str)

    hparam_str_dict = dict(seed=FLAGS.seed, env=FLAGS.env_name)
    hparam_str = ','.join([
        '%s=%s' % (k, str(hparam_str_dict[k]))
        for k in sorted(hparam_str_dict.keys())
    ])

    summary_writer = SummaryWriter(os.path.join(save_dir, 'tb',
                                                hparam_str),
                                   write_to_disk=True)
    os.makedirs(save_dir, exist_ok=True)

    env, dataset = make_env_and_dataset(FLAGS.env_name, FLAGS.seed)

    kwargs = dict(FLAGS.config)
    wandb.config.update(kwargs)
    wandb.config.update({'base_dir': save_dir})

    args = ConfigArgs(sample_random_times=FLAGS.sample_random_times,
                      grad_pen=FLAGS.grad_pen,
                      lambda_gp=FLAGS.lambda_gp,
                      noise=FLAGS.noise,
                      max_clip=FLAGS.max_clip,
                      num_v_updates=FLAGS.num_v_updates,
                      log_loss=FLAGS.log_loss,
                      noise_std=FLAGS.noise_std)
    agent = MCEPLearner(FLAGS.seed,
                    env.observation_space.sample()[np.newaxis],
                    env.action_space.sample()[np.newaxis],
                    max_steps=FLAGS.max_steps,
                    loss_temp=FLAGS.temp,
                    double_q=FLAGS.double,
                    vanilla=False,
                    temperature=FLAGS.beta,
                    temperature_target=FLAGS.beta_target,
                    args=args,
                    **kwargs)

    best_eval_returns = -np.inf
    eval_returns = []
    for i in tqdm(range(1, FLAGS.max_steps + 1)): # Remove TQDM
        batch = dataset.sample(FLAGS.batch_size)

        update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    summary_writer.add_scalar(f'training/{k}', v, i)
                else:
                    summary_writer.add_histogram(f'training/{k}', v, i, max_bins=512)
            summary_writer.flush()

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent, env, FLAGS.eval_episodes)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/average_{k}s', v, i)

            summary_writer.flush()

            if eval_stats['return'] >= best_eval_returns:
                # Store best eval returns
                best_eval_returns = eval_stats['return']
                agent.save(os.path.join(save_dir, f'{FLAGS.seed}', f'best'))

            summary_writer.add_scalar(f'evaluation/best_returns', best_eval_returns, i)
            wandb.run.summary["best_returns"] = best_eval_returns

            eval_returns.append((i, eval_stats['return']))
            np.savetxt(os.path.join(save_dir, f'{FLAGS.seed}.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])

            # agent.save(os.path.join(save_dir, f'{FLAGS.seed}', f'iter_{i}'))

    wandb.finish()
    sys.exit(0)
    os._exit(0)
    raise SystemExit


if __name__ == '__main__':
    app.run(main)
