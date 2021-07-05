#copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .review_kd_loss import ReviewKDLoss
from .review_kd_loss import HCL


class CELoss(nn.Layer):
    def __init__(self, name="loss_ce", epsilon=None):
        super().__init__()
        self.name = name
        if epsilon is not None and (epsilon <= 0 or epsilon >= 1):
            epsilon = None
        self.epsilon = epsilon

    def _labelsmoothing(self, target, class_num):
        if target.shape[-1] != class_num:
            one_hot_target = F.one_hot(target, class_num)
        else:
            one_hot_target = target
        soft_target = F.label_smooth(one_hot_target, epsilon=self.epsilon)
        soft_target = paddle.reshape(soft_target, shape=[-1, class_num])
        return soft_target

    def __call__(self, logits, label, mode="train"):
        loss_dict = {}
        if self.epsilon is not None:
            class_num = logits.shape[-1]
            label = self._labelsmoothing(label, class_num)

            x = -F.log_softmax(x, axis=-1)
            loss = paddle.sum(x * label, axis=-1)
        else:
            if label.shape[-1] == logits.shape[-1]:
                label = F.softmax(label, axis=-1)
                soft_label = True
            else:
                soft_label = False
            loss = F.cross_entropy(logits, label=label, soft_label=soft_label)

        loss_dict[self.name] = paddle.mean(loss)
        return loss_dict


class S_GT_CELoss(CELoss):
    '''
    for distillation training loss
    return student gt loss
    '''
    def __init__(self, name="loss_ce_s_gt", epsilon=None):
        super().__init__(name=name, epsilon=epsilon)

    def __call__(self, predicts, batch, mode="train"):
        logits = predicts["student"]["logits"]
        label = batch["label"].reshape([-1, 1])
        return super().__call__(logits, label, mode=mode)


class S_T_ReviewKDLoss(ReviewKDLoss):
    def __init__(
            self,
            in_channels=[],
            out_channels=[],
            mid_channel=256,
            shapes=[1, 7, 14, 28, 56],
            student_keepkeys=[],
            teacher_keepkeys=[],
            hcl_mode="avg",
            name="loss_review_kd", ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            mid_channel=mid_channel,
            shapes=shapes,
            hcl_mode=hcl_mode,
            name=name)
        self.student_keepkeys = student_keepkeys
        self.teacher_keepkeys = teacher_keepkeys

    def __call__(self, predicts, batch, mode="train"):
        s_feats = [
            predicts["student"]["feats"][key] for key in self.student_keepkeys
        ]
        t_feats = [
            predicts["teacher"]["feats"][key] for key in self.teacher_keepkeys
        ]
        return super().__call__(s_feats, t_feats)


class S_T_HCLLoss(HCL):
    def __init__(
            self,
            hcl_mode="avg",
            s_keep_keys=None,
            t_keep_keys=None,
            name="loss_review_kd", ):
        super().__init__(mode=hcl_mode)
        self.s_keep_keys = s_keep_keys
        self.t_keep_keys = t_keep_keys
        self.name = name

    def __call__(self, predicts, batch, mode="train"):
        s_feats = predicts["student"]["feats"]
        t_feats = predicts["teacher"]["feats"]
        if self.s_keep_keys is not None:
            s_feats = [s_feats[key] for key in self.s_keep_keys]
        if self.t_keep_keys is not None:
            t_feats = [t_feats[key] for key in self.t_keep_keys]
        
        loss = super().__call__(s_feats, t_feats)
        return {self.name: loss}
