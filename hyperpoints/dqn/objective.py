import torch
import torch.optim as optim

from vel.rl.metrics import EpisodeRewardMetric
from vel.storage.streaming.stdout import StdoutStreaming
from vel.util.random import set_seed

from vel.rl.env.classic_atari import ClassicAtariEnv

from vel.rl.models.q_model import QModelFactory
from vel.rl.models.backbone.nature_cnn import NatureCnnFactory

from vel.rl.reinforcers.buffered_single_off_policy_iteration_reinforcer import (
    BufferedSingleOffPolicyIterationReinforcer, 
    BufferedSingleOffPolicyIterationReinforcerSettings 
)

from vel.schedules.linear_and_constant import LinearAndConstantSchedule
from vel.rl.algo.dqn import DeepQLearning 
from vel.rl.env_roller.single.deque_replay_roller_epsgreedy import DequeReplayRollerEpsGreedy 

from vel.api.info import TrainingInfo, EpochInfo
from vel.rl.commands.rl_train_command import FrameTracker


def enduro_dqn(hparams):
    device = torch.device('cuda')
    seed = 1001

    # Set random seed in python std lib, numpy and pytorch
    set_seed(seed)

    kernel1 = int(hparams[0])
    kernel2 = int(hparams[1])
    kernel3 = int(hparams[2])
    discount_factor = float(hparams[3])
    buffer_capacity = int(hparams[4])
    batch_size = int(hparams[5])

    # Create 16 environments evaluated in parallel in sub processess with all usual DeepMind wrappers
    # These are just helper functions for that
    env = ClassicAtariEnv('EnduroNoFrameskip-v4').instantiate(seed=seed)

    # Again, use a helper to create a model
    # But because model is owned by the reinforcer, model should not be accessed using this variable
    # but from reinforcer.model property
    model = QModelFactory(
        backbone=NatureCnnFactory(input_width=84, input_height=84, input_channels=4,
                                  kernel1=kernel1, kernel2=kernel2, kernel3=kernel3)
    )

    # Reinforcer - an object managing the learning process
    reinforcer = BufferedSingleOffPolicyIterationReinforcer(
        device=device,
        settings=BufferedSingleOffPolicyIterationReinforcerSettings(
            batch_training_rounds=1,
            batch_rollout_rounds=4,
            batch_size=batch_size,
            discount_factor=discount_factor
        ),
        model=model.instantiate(action_space=env.action_space),
        environment=env,
        algo=DeepQLearning(
            model_factory=model,
            target_update_frequency=10000,
            double_dqn=False,
            max_grad_norm=0.5,
        ),
        env_roller=DequeReplayRollerEpsGreedy(
            environment=env,
            device=device,
            buffer_capacity=buffer_capacity,
            buffer_initial_size=30000,
            frame_stack=4,
            batch_size=batch_size,
            epsilon_schedule=LinearAndConstantSchedule(
                end_of_interpolation=0.1,
                initial_value=1.0,
                final_value=0.1,
            )
        )
    )

    # Model optimizer
    optimizer = optim.RMSprop(reinforcer.model.parameters(), lr=2.5e-4, alpha=0.95, momentum=0.95, eps=1.0e-1)

    # Overall information store for training information
    training_info = TrainingInfo(
        metrics=[
            EpisodeRewardMetric('episode_rewards'),  # Calculate average reward from episode
        ],
        callbacks=[
            StdoutStreaming(),   # Print live metrics every epoch to standard output
            FrameTracker(1.1e7)  # We need frame tracker to track the progress of learning
        ]
    )

    # A bit of training initialization bookkeeping...
    training_info.initialize()
    reinforcer.initialize_training(training_info)
    training_info.on_train_begin()

    # Let's make 10 batches per epoch to average metrics nicely
    # Rollout size is 8 environments times 128 steps
    num_epochs = int(1.1e7 / (128 * 8) / 10)

    # Normal handrolled training loop
    for i in range(1, num_epochs+1):
        epoch_info = EpochInfo(
            training_info=training_info,
            global_epoch_idx=i,
            batches_per_epoch=2500,
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
