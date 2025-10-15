import os
from subprocess import call
import pickle
from pathlib import Path

import tap


class Arguments(tap.Tap):
    level1_dir: Path = None
    level2_dir: Path = None
    level3_dir: Path = None
    level4_dir: Path = None

    save_dir: Path


def process_task(root_dir: Path, save_dir: Path):

    # list all tasks in root_dir
    tasks = os.listdir(root_dir)
    for task in tasks:
        # create the save task dir
        save_task_dir = f'{save_dir}/{task}/all_variations/episodes'
        os.makedirs(save_task_dir, exist_ok=True)

        # find all episodes of the task in save_task_dir to determine the start episode index
        all_episodes_in_save_task_dir = os.listdir(f'{save_task_dir}')
        all_episodes_in_save_task_dir = [int(episode.replace('episode', '')) for episode in all_episodes_in_save_task_dir if 'episode' in episode]
        if len(all_episodes_in_save_task_dir) == 0:
            start_episode_index = 0
        else:
            start_episode_index = max(all_episodes_in_save_task_dir) + 1

        # create link to the episodes in the root_dir
        all_episodes = os.listdir(f'{root_dir}/{task}/all_variations/episodes')
        for episode in all_episodes:
            episode_dir = f'{root_dir}/{task}/all_variations/episodes/{episode}'
            if os.path.isdir(episode_dir):
                call(['ln', '-s',
                      episode_dir,
                      f'{save_task_dir}/episode{start_episode_index}'])
                start_episode_index += 1



if __name__ == '__main__':
    args = Arguments().parse_args()
    for root_dir in [args.level1_dir, args.level2_dir, args.level3_dir, args.level4_dir]:
        if root_dir is not None and root_dir.exists():
            process_task(root_dir, args.save_dir)
