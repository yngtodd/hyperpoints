import torch
import torch.optim as optim

from vel.rl.metrics import EpisodeRewardMetric
from vel.storage.streaming.stdout import StdoutStreaming
from vel.util.random import set_seed

from vel.rl.env.classic_atari import ClassicAtariEnv
from vel.rl.vecenv.subproc import SubprocVecEnvWrapper

from vel.rl.models.q_policy_gradient_model import QPolicyGradientModelFactory
from vel.rl.models.backbone.nature_cnn import NatureCnnFactory

from vel.rl.reinforcers.buffered_mixed_policy_iteration_reinforcer import (
         BufferedMixedPolicyIterationReinforcer,
         BufferedMixedPolicyIterationReinforcerSettings
)

from vel.rl.algo.policy_gradient.acer import AcerPolicyGradient
from vel.rl.env_roller.vec.replay_q_env_roller import ReplayQEnvRoller

from vel.api.info import TrainingInfo, EpochInfo

from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def enduro_acer(hparams):
    device = torch.device('cuda:1')
    seed = 1001

    kernel1 = int(hparams[0])
    kernel2 = int(hparams[1])
    kernel3 = int(hparams[2])
    entropy = float(hparams[3])
    q_coefficient = float(hparams[4])
    rho_cap = float(hparams[5])
    retrace_rho_cap = float(hparams[6])

    # Set random seed in python std lib, numpy and pytorch
    set_seed(seed)

    # Create 16 environments evaluated in parallel in sub processess with all usual DeepMind wrappers
    # These are just helper functions for that
    vec_env = SubprocVecEnvWrapper(
        ClassicAtariEnv('EnduroNoFrameskip-v4'), frame_history=4
    ).instantiate(parallel_envs=12, seed=seed)

    # Again, use a helper to create a model
    # But because model is owned by the reinforcer, model should not be accessed using this variable
    # but from reinforcer.model property
    model = QPolicyGradientModelFactory(
        backbone=NatureCnnFactory(input_width=84, input_height=84, input_channels=4
                                  kernel1=kernel1, kernel2=kernel2, kernel3=kernel3)
    )

    # Reinforcer - an object managing the learning process
    reinforcer = BufferedMixedPolicyIterationReinforcer(
        device=device,
        env=vec_env,
        settings=BufferedMixedPolicyIterationReinforcerSettings(
            discount_factor=0.99,
            experience_replay=4,
            stochastic_experience_replay=True
        ),
        model=model.instantiate(action_space=vec_env.action_space),
        algo=AcerPolicyGradient(
            model_factory=model,
            trust_region=False,
            entropy_coefficient=entropy,
            q_coefficient=q_coefficient,
            rho_cap=rho_cap,
            retrace_rho_cap=retrace_rho_cap,
        ),
        env_roller=ReplayQEnvRoller(
            environment=vec_env,
            device=device,
            buffer_capacity=50000,
            buffer_initial_size=10000,
            frame_stack_compensation=1,
            number_of_steps=20,
            discount_factor=0.99
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
    num_epochs = int(1.1e7 / (5 * 16) / 100)

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

    # Use the average of the last 25% of rewards to determine score.
    endgame = int(num_epochs * .25)
    history = training_info.history.frame()
    rewards = np.array(history['episode_rewards'])
    endgame_average = sum(rewards[-endgame:])/endgame
    return -endgame_average
