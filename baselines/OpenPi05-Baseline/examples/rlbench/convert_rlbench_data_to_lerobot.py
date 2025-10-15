"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

from abc import abstractmethod
from pathlib import Path
import pickle
import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image
import tyro


class ProprioModeBase:
    @abstractmethod
    def convert_low_dim_obs_to_proprio(self, step: int, low_dim_obs: list[dict]):
        pass

    @abstractmethod
    def get_description(self):
        """Returns a brief description of the action."""

    @abstractmethod
    def get_shape(self):
        """Returns the shape the action."""


class JointPositionsProprioMode(ProprioModeBase):
    def __init__(self, joint_count):
        self.joint_count = joint_count

    def convert_low_dim_obs_to_proprio(self, step: int, low_dim_obs: list[dict]):
        return np.concatenate(
            [low_dim_obs[step].joint_positions, [low_dim_obs[step].gripper_open]],
            axis=-1,
            dtype=np.float32,
        )

    def get_description(self):
        return f"Joint positions angles ({self.joint_count}) and the discrete gripper state (1)."

    def get_shape(self):
        return (self.joint_count + 1,)


def main(data_dir: str, repo_name: str, train_mode: str, *, push_to_hub: bool = False):
    assert train_mode in ("vanilla", "half")

    data_dir = Path(data_dir).resolve()
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Specified data_dir is not a valid directory: {data_dir}")

    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="panda",
        fps=10,
        features={
            "front_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    proprio_mode = JointPositionsProprioMode(joint_count=7)

    task_folders = data_dir.glob("*")
    for task_folder in task_folders:
        if not task_folder.is_dir():
            continue

        episodes_paths = task_folder.glob("all_variations/episodes/*")
        for episode_path in episodes_paths:
            if not episode_path.is_dir():
                continue

            descriptions_path = episode_path / "variation_descriptions.pkl"
            if not descriptions_path.exists():
                print(f"Warning: {descriptions_path} does not exist, skipping this episode")
                continue
            with descriptions_path.open("rb") as file:
                all_descriptions = pickle.load(file)
                description_vanilla = all_descriptions["vanilla"][0]
                description_half = all_descriptions["oracle_half"][0].split("\n")
                instr_index = 0

            low_dim_obs_path = episode_path / "low_dim_obs.pkl"
            if not low_dim_obs_path.exists():
                print(f"Warning: {low_dim_obs_path} does not exist, skipping this episode")
                continue
            with low_dim_obs_path.open("rb") as file:
                low_dim_obs = pickle.load(file)

            flag = False
            for step_idx, _ in enumerate(low_dim_obs):
                if step_idx == len(low_dim_obs) - 1:
                    break

                front_image_path = episode_path / "front_rgb" / f"{step_idx}.png"
                wrist_image_path = episode_path / "wrist_rgb" / f"{step_idx}.png"

                if not front_image_path.exists():
                    print(f"Warning: {front_image_path} does not exist, skipping this step")
                    continue
                if not wrist_image_path.exists():
                    print(f"Warning: {wrist_image_path} does not exist, skipping this step")
                    continue

                try:
                    front_image = np.array(Image.open(front_image_path))
                    wrist_image = np.array(Image.open(wrist_image_path))
                except Exception as e:
                    print(f"Warning: Error reading images at step {step_idx}: {e}, skipping this step")
                    continue

                proprio = proprio_mode.convert_low_dim_obs_to_proprio(step_idx, low_dim_obs)
                proprio_next = proprio_mode.convert_low_dim_obs_to_proprio(step_idx + 1, low_dim_obs)

                if train_mode == "vanilla":
                    description = description_vanilla
                else:  # train_mode == "half":
                    description = description_half[instr_index]
                    if step_idx != len(low_dim_obs) - 2 and abs(proprio[-1] - proprio_next[-1]) > 1e-9:
                        instr_index = (instr_index + 1) % len(description_half)
                        flag = True
                # print(description)
                dataset.add_frame(
                    {
                        "front_image": front_image,
                        "wrist_image": wrist_image,
                        "state": proprio,
                        "actions": proprio_next,
                        "task": description,
                    }
                )

                if flag:
                    dataset.save_episode()
                    flag = False
            try:
                dataset.save_episode()
            except Exception as e:
                print(e)

    # dataset.consolidate(run_compute_stats=False)

    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
