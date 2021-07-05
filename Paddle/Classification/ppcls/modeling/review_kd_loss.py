# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import numpy as np
import paddle
from paddle import nn
from paddle import ParamAttr
import paddle.nn.functional as F
from paddle.nn.initializer import KaimingNormal


class ABF(nn.Layer):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super().__init__()
        self.conv1 = nn.Conv2D(
            in_channel,
            mid_channel,
            kernel_size=1,
            bias_attr=False,
            weight_attr=ParamAttr(initializer=KaimingNormal()))
        self.conv1_bn = nn.BatchNorm2D(mid_channel)

        self.conv2 = nn.Conv2D(
            mid_channel,
            out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False,
            weight_attr=ParamAttr(initializer=KaimingNormal()))
        self.conv2_bn = nn.BatchNorm2D(out_channel)
        if fuse:
            self.att_conv = nn.Sequential(
                nn.Conv2D(
                    mid_channel * 2, 2, kernel_size=1),
                nn.Sigmoid(), )
        else:
            self.att_conv = None

    def forward(self, x, y=None, shape=None):
        n, _, h, w = x.shape
        # transform student features
        x = self.conv1(x)
        if self.att_conv is not None:
            # upsample residual features
            y = F.interpolate(y, (shape, shape), mode="nearest")
            # fusion
            z = paddle.concat([x, y], axis=1)
            z = self.att_conv(z)
            x = (x * z[:, 0].reshape([n, 1, h, w]) + y * z[:, 1].reshape(
                [n, 1, h, w]))
        y = self.conv2(x)
        return y, x


class ReviewKDLoss(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channel,
                 shapes=[1, 7, 14, 28, 56],
                 hcl_mode="avg",
                 name="loss_review_kd"):
        super().__init__()
        self.shapes = shapes
        self.name = name

        abfs = nn.LayerList()

        for idx, in_channel in enumerate(in_channels):
            abfs.append(
                ABF(in_channel, mid_channel, out_channels[idx], idx < len(
                    in_channels) - 1))

        self.abfs = abfs[::-1]

        self.hcl = HCL(mode=hcl_mode)

    def forward(self, student_features, teacher_features):
        '''
        student_features: list of tensor, low-level -> high-level
        student_logit: tensor, N x class_num
        '''
        # merge students' feature
        # x is from high-level to low-level
        x = student_features[::-1]
        results = []
        out_features, res_features = self.abfs[0](x[0])
        results.append(out_features)
        for idx in range(1, len(x)):
            features, abf, shape = x[idx], self.abfs[idx], self.shapes[idx]
            out_features, res_features = abf(features, res_features, shape)
            results.insert(0, out_features)

        loss_dict = dict()
        loss_dict[self.name] = self.hcl(results, teacher_features)

        return loss_dict


class HCL(nn.Layer):
    def __init__(self, mode="avg"):
        super(HCL, self).__init__()
        assert mode in ["max", "avg"]
        self.mode = mode

    def forward(self, fstudent, fteacher):
        loss_all = 0.0
        for fs, ft in zip(fstudent, fteacher):
            h = fs.shape[2]
            loss = F.mse_loss(fs, ft)
            cnt = 1.0
            tot = 1.0
            for l in [4, 2, 1]:
                if l >= h:
                    continue
                if self.mode == "max":
                    tmpfs = F.adaptive_max_pool2d(fs, (l, l))
                    tmpft = F.adaptive_max_pool2d(ft, (l, l))
                else:
                    tmpfs = F.adaptive_avg_pool2d(fs, (l, l))
                    tmpft = F.adaptive_avg_pool2d(ft, (l, l))

                cnt /= 2.0
                loss += F.mse_loss(tmpfs, tmpft) * cnt
                tot += cnt
            loss = loss / tot
            loss_all = loss_all + loss
        return loss_all