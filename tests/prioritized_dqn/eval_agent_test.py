# Copyright 2022 The Deep RL Zoo Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for Prioritized DQN."""
from absl import flags
from absl.testing import flagsaver
from absl.testing import absltest
from absl.testing import parameterized

from deep_rl_zoo.prioritized_dqn import eval_agent

FLAGS = flags.FLAGS
FLAGS.tensorboard = False
FLAGS.num_iterations = 1
FLAGS.load_checkpoint_file = '/tmp/not_found_checkpoint.ckpt'
FLAGS.recording_video_dir = ''


class RunEvaluationAgentTest(parameterized.TestCase):
    @flagsaver.flagsaver
    def test_can_not_find_checkpoint(self):
        FLAGS.environment_name = 'Pong'
        FLAGS.num_eval_frames = 200

        with self.assertRaisesRegex(ValueError, 'is not a valid checkpoint file'):
            eval_agent.main(None)


if __name__ == '__main__':
    absltest.main()
