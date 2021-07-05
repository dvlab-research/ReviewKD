# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import paddle
import paddle.nn as nn

from .resnet import ResNet50
from .reviewkd import ReviewKD

__all__ = [
    'DistillationModel',
]

class DistillationModel(nn.Layer):
    def __init__(self, classes_num=1000, freeze_teacher=False, **args):
        super(DistillationModel, self).__init__()

        teacher_config = args["teacher"]
        student_config = args["student"]

        t_name = teacher_config.pop("name")
        s_name = student_config.pop("name")

        self.teacher = eval(t_name)(classes_num=classes_num, **teacher_config)
        self.student = eval(s_name)(classes_num=classes_num, **student_config)

        if freeze_teacher:
            for param in self.teacher.parameters():
                param.trainable = False

    def forward(self, input):
        teacher_out = self.teacher(input)
        student_out = self.student(input)
        output = {
            "teacher": teacher_out,
            "student": student_out,
        }
        return output