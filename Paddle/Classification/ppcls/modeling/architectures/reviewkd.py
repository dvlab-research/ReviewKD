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

from ppcls.utils.initializer import kaiming_normal_, kaiming_uniform_

from .mobilenet_v1 import MobileNet

class ABF(nn.Layer):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super().__init__()
        self.conv1 = nn.Conv2D(
            in_channel, mid_channel, kernel_size=1, bias_attr=False)
        self.conv1_bn = nn.BatchNorm2D(mid_channel)

        self.conv2 = nn.Conv2D(
            mid_channel,
            out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False)
        self.conv2_bn = nn.BatchNorm2D(out_channel)
        if fuse:
            self.att_conv = nn.Sequential(
                nn.Conv2D(
                    mid_channel * 2, 2, kernel_size=1),
                nn.Sigmoid(), )
        else:
            self.att_conv = None

        self.init_params()

    def init_params(self, ):
        kaiming_uniform_(self.conv1.weight, a=1)
        kaiming_uniform_(self.conv2.weight, a=1)

    def forward(self, x, y=None, shape=None):
        n, _, h, w = x.shape
        # transform student features
        x = self.conv1(x)
        x = self.conv1_bn(x)
        if self.att_conv is not None:
            # upsample residual features
            y = F.interpolate(y, (shape, shape), mode="nearest")
            # fusion
            z = paddle.concat([x, y], axis=1)
            z = self.att_conv(z)
            x = (x * z[:, 0].reshape([n, 1, h, w]) + y * z[:, 1].reshape(
                [n, 1, h, w]))
        y = self.conv2(x)
        y = self.conv2_bn(y)
        return y, x


class ReviewKD(nn.Layer):
    def __init__(self,
                 student,
                 student_args=dict(),
                 in_channels=[],
                 out_channels=[],
                 mid_channel=[],
                 feat_keepkeys=[],
                 **kargs):
        super().__init__()
        self.shapes = [1, 7, 14, 28, 56]
        self.student = eval(student)(**student_args)
        self.feat_keepkeys = feat_keepkeys

        abfs = nn.LayerList()

        for idx, in_channel in enumerate(in_channels):
            abfs.append(
                ABF(in_channel, mid_channel, out_channels[idx], idx < len(
                    in_channels) - 1))

        self.abfs = abfs[::-1]

    def forward(self, x):
        student_features = self.student(x, is_feat=True)
        logit = student_features[1]
        x = student_features[0][::-1]
        results = []
        out_features, res_features = self.abfs[0](x[0])
        results.append(out_features)
        for features, abf, shape in zip(x[1:], self.abfs[1:], self.shapes[1:]):
            out_features, res_features = abf(features, res_features, shape)
            results.insert(0, out_features)

        return {"feats": results, "logits": logit}

if __name__ == "__main__":
    in_channels = [128, 256, 512, 1024, 1024]
    out_channels = [256, 512, 1024, 2048, 2048]
    mid_channel = 256
    student = "MobileNet"
    student_args = {}
    review_kd = ReviewKD(
        student,
        student_args=student_args,
        in_channels=in_channels,
        out_channels=out_channels,
        mid_channel=mid_channel)

    x = paddle.rand([1, 3, 224, 224])

    results, logits = review_kd(x)
    for r in results:
        print(r.shape)
    print("=====ligits====")
    print(logits.shape)
