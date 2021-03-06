import torch
import numpy as np
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

from vel.schedules.linear import LinearSchedule
from vel.rl.algo.policy_gradient.ppo import PpoPolicyGradient
from vel.rl.env_roller.vec.step_env_roller import StepEnvRoller

from vel.api.info import TrainingInfo, EpochInfo
from vel.rl.commands.rl_train_command import FrameTracker


def hyperband_objective(hparams, iterations):
    device = torch.device('cuda:2')
    seed = 1001

    # Set random seed in python std lib, numpy and pytorch
    set_seed(seed)

    kernel1 = int(hparams[0])
    kernel2 = int(hparams[1])
    kernel3 = int(hparams[2])
    discount_factor = float(hparams[3])
    entropy = float(hparams[4])
    gae_lambda = float(hparams[5])
    lr = float(hparams[6])
    value_coefficient = float(hparams[7])

    # Create 16 environments evaluated in parallel in sub processess with all usual DeepMind wrappers
    # These are just helper functions for that
    vec_env = SubprocVecEnvWrapper(
        ClassicAtariEnv('QbertNoFrameskip-v4'), frame_history=4
    ).instantiate(parallel_envs=8, seed=seed)

    # Again, use a helper to create a model
    # But because model is owned by the reinforcer, model should not be accessed using this variable
    # but from reinforcer.model property
    model = PolicyGradientModelFactory(
        backbone=NatureCnnFactory(input_width=84, input_height=84, input_channels=4,
                                  kernel1=kernel1, kernel2=kernel2, kernel3=kernel3)
    ).instantiate(action_space=vec_env.action_space)

    # Set schedule for gradient clipping.
    cliprange = LinearSchedule(
        initial_value=0.1,
        final_value=0.0
    )

    # Reinforcer - an object managing the learning process
    reinforcer = OnPolicyIterationReinforcer(
        device=device,
        settings=OnPolicyIterationReinforcerSettings(
            discount_factor=discount_factor,
            batch_size=256,
            experience_replay=4
        ),
        model=model,
        algo=PpoPolicyGradient(
            entropy_coefficient=entropy,
            value_coefficient=value_coefficient,
            max_grad_norm=0.5,
            cliprange=cliprange
        ),
        env_roller=StepEnvRoller(
            environment=vec_env,
            device=device,
            gae_lambda=gae_lambda,
            number_of_steps=128,
            discount_factor=discount_factor,
        )
    )

    # Model optimizer
    optimizer = optim.Adam(reinforcer.model.parameters(), lr=lr, eps=1.0e-5)

    # Overall information store for training information
    training_info = TrainingInfo(
        metrics=[
            EpisodeRewardMetric('episode_rewards'),  # Calculate average reward from episode
        ],
        callbacks=[
            StdoutStreaming(), # Print live metrics every epoch to standard output
            FrameTracker()]    # We need frame tracker to track the progress of learning
    )

    training_info['total_frames'] = 1.1e7  # How many frames in the whole training process

    # A bit of training initialization bookkeeping...
    training_info.initialize()
    reinforcer.initialize_training(training_info)
    training_info.on_train_begin()

    # Let's make 100 batches per epoch to average metrics nicely
    #num_epochs = int(1.1e7 / (5 * 16) / 100)
    num_epochs = iterations

    # Normal handrolled training loop
    for i in range(1, num_epochs+1):
        epoch_info = EpochInfo(
            training_info=training_info,
            global_epoch_idx=i,
            batches_per_epoch=10,
            optimizer=optimizer
        )

        reinforcer.train_epoch(epoch_info)

    training_info.on_train_end()

    # Use the average of the last 25% of rewards to determine score.
    endgame = int(num_epochs * .25)
    history = training_info.history.frame()
    rewards = np.array(history['episode_rewards'])
    endgame_average = sum(rewards[-endgame:])/endgame
    return -endgame_average
