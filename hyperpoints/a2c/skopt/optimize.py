import time
import torch
import torch.optim as optim

from vel.rl.metrics import EpisodeRewardMetric
from vel.storage.streaming.stdout import StdoutStreaming
from vel.util.random import set_seed

from vel.rl.env.classic_atari import ClassicAtariEnv
from vel.rl.vecenv.subproc import SubprocVecEnvWrapper

from vel.rl.models.policy_gradient_model import PolicyGradientModelFactory
from vel.rl.models.backbone.nature_cnn import NatureCnnFactory

from vel.rl.reinforcers.on_policy_iteration_reinforcer import (
    OnPolicyIterationReinforcer, OnPolicyIterationReinforcerSettings
)

from vel.rl.algo.policy_gradient.a2c import A2CPolicyGradient
from vel.rl.env_roller.vec.step_env_roller import StepEnvRoller

from vel.api.info import TrainingInfo, EpochInfo

import numpy as np
from skopt import dump
from skopt import gp_minimize
from skopt.callbacks import CheckpointSaver


def objective(hparams):
    kernel1, kernel2, kernel3, discount_factor = hparams
    print(f'kernels: {kernel1}, {kernel2}, {kernel3}')
    discount_factor = float(discount_factor)

    device = torch.device('cuda')
    seed = 1001

    # Set random seed in python std lib, numpy and pytorch
    set_seed(seed)

    # Create 16 environments evaluated in parallel in sub processess with all usual DeepMind     wrappers
    # These are just helper functions for that
    vec_env = SubprocVecEnvWrapper(
        ClassicAtariEnv('BreakoutNoFrameskip-v4'), frame_history=4
    ).instantiate(parallel_envs=16, seed=seed)

    model = PolicyGradientModelFactory(
        backbone=NatureCnnFactory(input_width=84, input_height=84, input_channels=4,
                                  kernel1=kernel1, kernel2=kernel2, kernel3=kernel3)
    ).instantiate(action_space=vec_env.action_space)

    # Reinforcer - an object managing the learning process
    reinforcer = OnPolicyIterationReinforcer(
        device=device,
        settings=OnPolicyIterationReinforcerSettings(
            discount_factor=discount_factor,
            batch_size=256
        ),
        model=model,
        algo=A2CPolicyGradient(
            entropy_coefficient=0.01,
            value_coefficient=0.5,
            max_grad_norm=0.5
        ),
        env_roller=StepEnvRoller(
            environment=vec_env,
            device=device,
            number_of_steps=5,
            discount_factor=discount_factor,
        )
    )

    # Model optimizer
    optimizer = optim.RMSprop(reinforcer.model.parameters(), lr=7.0e-4, eps=1e-3)

    # Overall information store for training information
    training_info = TrainingInfo(
        metrics=[
            EpisodeRewardMetric('episode_rewards'),  # Calculate average reward from episode
        ],
        callbacks=[StdoutStreaming()]  # Print live metrics every epoch to standard output
    )

    # A bit of training initialization bookkeeping...
    training_info.initialize()
    reinforcer.initialize_training(training_info)
    training_info.on_train_begin()

    # Let's make 100 batches per epoch to average metrics nicely
    #num_epochs = int(1.1e7 / (5 * 16) / 100)
    num_epochs = 200

    # Normal handrolled training loop
    for i in range(1, num_epochs+1):
        epoch_info = EpochInfo(
            training_info=training_info,
            global_epoch_idx=i,
            batches_per_epoch=100,
            optimizer=optimizer
        )

        reinforcer.train_epoch(epoch_info)

    training_info.on_train_end()
    # Use the average of the last 50 rewards to determine score.
    history = training_info.history.frame()
    rewards = np.array(history['episode_rewards'])
    endgame_average = sum(rewards[-50:])/50
    return -endgame_average


def main():
    start = time.time()
    checkpoint_saver = CheckpointSaver("./checkpoint.pkl")
    space = [(2,10), (2,8), (3,8), (0.00, .99)]
    res_gp = gp_minimize(objective, space, n_calls=20, random_state=0, callback=[checkpoint_saver], verbose=True)
    dump(res_gp, 'result200.pkl')
    print(f'Runtime: {time.time() - start}')


if __name__ == '__main__':
    main()
