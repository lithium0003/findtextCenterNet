# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for time_two ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.framework import test_util
try:
  from custom_fill_op.python.ops import custom_fill_ops
except ImportError:
  import custom_fill_ops


class CustomFillOpTest(test.TestCase):

  @test_util.run_gpu_only
  def testCustomFill(self):
    with self.test_session():
      with ops.device("/gpu:0"):
        self.assertAllClose(
            custom_fill_ops.id_fill_ops(buffer=tf.zeros([3,3,2], tf.int32),position=[[0.,0.,0.,0.]],code_list=[[1,2]],sortidx=[0]), [1,2] * np.ones([3,3,1], np.int32))


if __name__ == '__main__':
  test.main()
