import argparse

from ddpg_model import DDPGAgentAdjustedUnity
from utils import get_ddpg_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-path', default='results/DDPGAgentAdjustedUnity-vanilla-800000', type=str
    )
    parser.add_argument('--brain-name', default='ReacherBrain', type=str)
    parser.add_argument('--n-episodes', default=20, type=int)
    args = parser.parse_args()

    config = get_ddpg_config()
    config.eval_episodes = args.n_episodes
    ddpg_model = DDPGAgentAdjustedUnity(config, args.brain_name)
    ddpg_model.load(args.model_path)

    ddpg_model.eval_episodes()
