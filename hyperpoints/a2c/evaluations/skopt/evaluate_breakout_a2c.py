import torch
import argparse

import pandas as pd
import numpy as np

from vel.rl.env.classic_atari import ClassicAtariEnv

from vel.rl.models.policy_gradient_model import PolicyGradientModelFactory
from vel.rl.models.backbone.nature_cnn import NatureCnnFactory

from vel.openai.baselines.common.atari_wrappers import FrameStack


def eval(checkpoint_file_path, algo, gpu=0, takes=10):
    model_checkpoint = torch.load(checkpoint_file_path)
    device = torch.device('cuda:' + str(gpu))

    env = FrameStack(
        ClassicAtariEnv('BreakoutNoFrameskip-v4').instantiate(preset='raw'), k=4
    )

    model = PolicyGradientModelFactory(
        backbone=NatureCnnFactory(input_width=84, input_height=84, input_channels=4,
                                  kernel1=10, kernel2=2, kernel3=8)
    ).instantiate(action_space=env.action_space)

    model.load_state_dict(model_checkpoint)
    model = model.to(device)

    model.eval()

    rewards = []
    lengths = []
    all_rewards = []

    for i in range(takes):
        result, eval_rewards = record_take(model, env, device)
        rewards.append(result['r'])
        lengths.append(result['l'])
        print(f'eval_rewards has shape {np.array(eval_rewards).shape}')
        all_rewards.append(eval_rewards)
        print(f'Num rewards in evaluation: {len(eval_rewards)}')

    all_evals = pd.concat([pd.Series(x) for x in all_rewards], axis=1)

    filename = 'breakout_a2c_' + algo + '.csv'
    all_evals.to_csv(filename, index=False)

    print(pd.DataFrame({'lengths': lengths, 'rewards': rewards}).describe())


@torch.no_grad()
def record_take(model, env_instance, device):
    frames = []

    observation = env_instance.reset()

    frames.append(env_instance.render('rgb_array'))

    print("Evaluating environment...")
    rewards = []
    while True:
        observation_array = np.expand_dims(np.array(observation), axis=0)
        observation_tensor = torch.from_numpy(observation_array).to(device)
        actions = model.step(observation_tensor, argmax_sampling=True)['actions']

        observation, reward, done, epinfo = env_instance.step(actions.item())
        rewards.append(reward)

        frames.append(env_instance.render('rgb_array'))

        if 'episode' in epinfo:
            # End of an episode
            return epinfo['episode'], rewards


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate pretrained Algorithm on an Atari environment!')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint.')
    parser.add_argument('--algo', type=str, help='Your method of choice, madamm.')
    parser.add_argument('--gpu', type=int, help='Pick your GPU!.')
    parser.add_argument('--takes', type=int, default=5, help='Number of takes.')
    args = parser.parse_args()

    eval(args.checkpoint, args.algo, gpu=args.gpu, takes=args.takes)
