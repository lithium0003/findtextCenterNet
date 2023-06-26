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

from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.framework import test_util
try:
  from custom_rotate_op.python.ops import custom_rotate_ops
except ImportError:
  import custom_rotate_ops


class CustomRotateOpTest(test.TestCase):

  @test_util.run_gpu_only
  def testCustomRotate(self):
    with self.test_session():
      with ops.device("/gpu:0"):
        self.assertAllClose(
            custom_rotate_ops.biliner_rotate_ops([[1, 2], [3, 4]], width=2, height=2, rot_cx=0.5, rot_cy=0.5, sx=1., sy=1., angle=0.), np.array([[1., 2.], [3., 4.]]))
        self.assertAllClose(
            custom_rotate_ops.nearest_rotate_ops([[1, 2], [3, 4]], width=2, height=2, rot_cx=0.5, rot_cy=0.5, sx=1., sy=1., angle=0.), np.array([[1, 2], [3, 4]]))


if __name__ == '__main__':
  test.main()
