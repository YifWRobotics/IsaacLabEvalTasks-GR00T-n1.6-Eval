# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
from io_utils import load_gr1_joints_config
from policies.image_conversion import resize_frames_with_padding
from policies.joints_conversion import remap_policy_joints_to_sim_joints, remap_sim_joints_to_policy_joints
from policies.policy_base import PolicyBase
from robot_joints import JointsAbsPosition

from isaaclab.sensors import Camera

from config.args import Gr00tN1ClosedLoopArguments


class Gr00tN1Policy(PolicyBase):
    def __init__(self, args: Gr00tN1ClosedLoopArguments):
        self.args = args
        self.policy = self._load_policy()
        self._load_policy_joints_config()
        self._load_sim_joints_config()

    def _load_policy_joints_config(self):
        """Load the policy joint config from the data config."""
        self.gr00t_joints_config = load_gr1_joints_config(self.args.gr00t_joints_config_path)

    def _load_sim_joints_config(self):
        """Load the simulation joint config from the data config."""
        self.gr1_state_joints_config = load_gr1_joints_config(self.args.state_joints_config_path)
        self.gr1_action_joints_config = load_gr1_joints_config(self.args.action_joints_config_path)

    def _load_policy(self):
        """Load a GR00T N1.6 policy from the model path."""
        assert os.path.exists(self.args.model_path), f"Model path {self.args.model_path} does not exist"
        embodiment_tag = EmbodimentTag(self.args.embodiment_tag)
        return Gr00tPolicy(
            model_path=self.args.model_path,
            embodiment_tag=embodiment_tag,
            device=self.args.policy_device,
            strict=True,
        )

    def step(self, current_state: JointsAbsPosition, camera: Camera) -> JointsAbsPosition:
        """Call every simulation step to update policy's internal state."""
        pass

    def get_new_goal(
        self, current_state: JointsAbsPosition, ego_camera: Camera, language_instruction: str
    ) -> JointsAbsPosition:
        """
        Run policy prediction on the given observations. Produce a new action goal for the robot.

        Args:
            current_state: robot proprioceptive state observation
            ego_camera: camera sensor observation
            language_instruction: language instruction for the task

        Returns:
            A dictionary containing the inferred action for robot joints.
        """
        rgb = ego_camera.data.output["rgb"]
        # Apply preprocessing to rgb
        rgb = resize_frames_with_padding(
            rgb, target_image_size=self.args.target_image_size, bgr_conversion=False, pad_img=True
        )
        # Retrieve joint positions as proprioceptive states and remap to policy joint orders
        robot_state_policy = remap_sim_joints_to_policy_joints(current_state, self.gr00t_joints_config)
        modality_config = self.policy.get_modality_config()
        state_keys = modality_config["state"].modality_keys
        video_keys = modality_config["video"].modality_keys

        state_obs = {}
        for key in state_keys:
            if key not in robot_state_policy:
                raise KeyError(
                    f"Policy expects state key '{key}', but it is missing from remapped humanoid state."
                )
            state_obs[key] = robot_state_policy[key].reshape(robot_state_policy[key].shape[0], 1, -1).astype("float32")

        video_obs = {}
        for key in video_keys:
            video_obs[key] = rgb.reshape(-1, 1, self.args.target_image_size[0], self.args.target_image_size[1], 3)

        language_key = self.policy.language_key
        language_obs = {language_key: [[language_instruction]] * rgb.shape[0]}

        # Pack inputs to dictionary and run N1.6 inference
        observations = {
            "video": video_obs,
            "state": state_obs,
            "language": language_obs,
        }

        robot_action_policy = self.policy.get_action(observations)
        if isinstance(robot_action_policy, tuple):
            robot_action_policy = robot_action_policy[0]

        # Accept both N1.6 action keys ('left_arm') and legacy keys ('action.left_arm').
        normalized_action = {}
        for key, value in robot_action_policy.items():
            normalized_action[key] = value
            if not key.startswith("action."):
                normalized_action[f"action.{key}"] = value

        robot_action_sim = remap_policy_joints_to_sim_joints(
            normalized_action, self.gr00t_joints_config, self.gr1_action_joints_config, self.args.simulation_device
        )

        return robot_action_sim

    def reset(self):
        """Resets the policy's internal state."""
        # As GN1 is a single-shot policy, we don't need to reset its internal state
        pass
