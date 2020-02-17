# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Berkeley (BAIR) robot pushing dataset.

Self-Supervised Visual Planning with Temporal Skip Connections
Frederik Ebert, Chelsea Finn, Alex X. Lee, and Sergey Levine.
https://arxiv.org/abs/1710.05268

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
import numpy as np

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import video_utils
from tensor2tensor.layers import modalities
from tensor2tensor.utils import registry

import cv2
from collections import defaultdict
import os
import h5py
import time
from tqdm import tqdm
import imageio

import tensorflow as tf

BASE_DIR = '.' 

if PROBLEM == 'SawyerLaundryLayout':
    DATA_FILE = os.path.join(BASE_DIR, 'SawyerLaundryLayout_aligned_output.hdf5')
elif PROBLEM == 'SawyerTowerCreation':
    DATA_FILE = os.path.join(BASE_DIR, 'SawyerTowerCreation_aligned_output.hdf5')
else:
    DATA_FILE = os.path.join(BASE_DIR, 'SawyerObjectSearch_aligned_output.hdf5')

# Lazy load PIL.Image
def PIL_Image():  # pylint: disable=invalid-name
  from PIL import Image  # pylint: disable=g-import-not-at-top
  return Image

# These hparams were from the BAIR Robot Pushing SV2P Experiments
@registry.register_hparams
def custom_next_frame_sv2p():
  """SV2P model hparams."""
  hparams = basic_stochastic.next_frame_basic_stochastic()
  hparams.optimizer = "true_adam"
  # Optimization hparams from SV2P Paper
  hparams.optimizer_adam_beta1 = 0.9
  hparams.optimizer_adam_beta2 = 0.999
  hparams.optimizer_adam_epsilon = 1e-8
  hparams.learning_rate_schedule = "constant"
  hparams.learning_rate_constant = 1e-3

  # Video Input Frames from BAIR
  hparams.video_num_input_frames = 2
  hparams.video_num_target_frames = 10
  hparams.batch_size = 16

  # Inference Network
  hparams.anneal_end = 200000
  hparams.num_iterations_1st_stage = 50000
  hparams.num_iterations_2nd_stage = 50000
  hparams.bottom = {
      "inputs": modalities.video_raw_bottom,
      "targets": modalities.video_raw_targets_bottom,
  }
  hparams.loss = {
      "targets": modalities.video_l2_raw_loss,
  }
  hparams.top = {
      "targets": modalities.video_raw_top,
  }
  hparams.video_modality_loss_cutoff = 0.0
  hparams.scheduled_sampling_mode = "count"
  hparams.scheduled_sampling_k = 900.0
  hparams.add_hparam("reward_prediction", True)
  hparams.add_hparam("reward_prediction_stop_gradient", False)
  hparams.add_hparam("reward_prediction_buffer_size", 0)
  hparams.add_hparam("model_options", "CDNA")
  hparams.add_hparam("num_masks", 10)
  hparams.add_hparam("multi_latent", False)
  hparams.add_hparam("relu_shift", 1e-12)
  hparams.add_hparam("dna_kernel_size", 5)
  hparams.add_hparam("upsample_method", "conv2d_transpose")
  hparams.add_hparam("reward_model", "basic")
  hparams.add_hparam("visualize_logits_histogram", True)
  return hparams


@registry.register_problem
class VideoRoboturkStanfordDataset(video_utils.VideoProblem):
  """Video Prediction for Roboturk Dataset."""

  @property
  def num_channels(self):
    return 3

  @property
  def frame_height(self):
    return 48

  @property
  def frame_width(self):
    return 64

  @property
  def is_generate_per_split(self):
    return True

  # num_train_files * num_videos * num_frames
  @property
  def total_number_of_frames(self):
    return 167 * 256 * 30

  @property
  def random_skip(self):
    return False

  @property
  def only_keep_videos_from_0th_frame(self):
    return False

  @property
  def use_not_breaking_batching(self):
    return True

  def eval_metrics(self):
    return []

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each."""
    return [
        {"split": problem.DatasetSplit.TRAIN, "shards": 10},
        {"split": problem.DatasetSplit.EVAL, "shards": 1},
        {"split": problem.DatasetSplit.TEST, "shards": 1}]

  @property
  def extra_reading_spec(self):
    """Additional data fields to store on disk and their decoders."""
    data_fields = {
        "frame_number": tf.FixedLenFeature([1], tf.int64),
        "state": tf.FixedLenFeature([4], tf.float32),
        "action": tf.FixedLenFeature([4], tf.float32),
    }
    decoders = {
        "frame_number": tf.contrib.slim.tfexample_decoder.Tensor(
            tensor_key="frame_number"),
        "state": tf.contrib.slim.tfexample_decoder.Tensor(tensor_key="state"),
        "action": tf.contrib.slim.tfexample_decoder.Tensor(tensor_key="action"),
    }
    return data_fields, decoders

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.modality = {"inputs": modalities.ModalityType.VIDEO,
                  "targets": modalities.ModalityType.VIDEO}
    p.vocab_size = {"inputs": 256,
                    "targets": 256}

  def parse_frames(self, demo, video_path):
    frame_num = 0
    joint_grippers = demo['robot_observation']['joint_states_gripper']
    eef_poses = demo['robot_observation']['eef_poses']

    vid = imageio.get_reader(video_path,  'ffmpeg')

    previous_state = None
    for index, frame in enumerate(vid):
        eef_pose = eef_poses[index][0:3]
        gripper_state = joint_grippers[index][0]
        if index % 2 == 0:
            previous_state = np.append(eef_pose, gripper_state) 
            continue

        image = cv2.resize(frame, (64, 48))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        diff_position = previous_state[0:3] - eef_pose
        diff_joint = np.absolute(previous_state[3] - gripper_state)
        state = np.append(eef_pose, gripper_state)
        action = np.append(diff_position, diff_joint)

        previous_state = state

        yield frame_num, image, state, action
        
        frame_num += 1


  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    filtered_filename = None

    if dataset_split == problem.DatasetSplit.TEST:
        filtered_filename = '../scripts/{}_test.txt'.format(PROBLEM)
    elif dataset_split == problem.DatasetSplit.EVAL:
        filtered_filename = '../scripts/{}_eval.txt'.format(PROBLEM)
    else:
        filtered_filename = '../scripts/{}_train.txt'.format(PROBLEM)

    filtered_users_demos = defaultdict(list)
    demos_to_videos = {}
    with open(filtered_filename, 'r') as filtered:
        for line in filtered:
            split_line = line.rstrip().split(',')
            user_demo, video_path = split_line
            demos_to_videos[user_demo] = video_path
            split_demo = user_demo.split('_')
            user, demo_id = split_demo
            filtered_users_demos[user].append(demo_id)

    counter = 0
    done = False
    f = h5py.File(DATA_FILE)
    data = f['data']
    data_keys = data.keys()
    for key in tqdm(data_keys):
        
        if key not in filtered_users_demos.keys():
            continue

        user = data[key]
        demo_ids = user.keys()
        for demo_id in demo_ids:
            if demo_id not in filtered_users_demos[key]:
                continue

            video_path_key = '{}_{}'.format(key, demo_id)
            video_path = demos_to_videos[video_path_key]
            for frame_num, image, state, action in self.parse_frames(user[demo_id], video_path):
                yield {
                    "frame": image, 
                    "frame_number": [frame_num],
                    "state": state.tolist(),
                    "action": action.tolist(),
                }
                counter += 1

