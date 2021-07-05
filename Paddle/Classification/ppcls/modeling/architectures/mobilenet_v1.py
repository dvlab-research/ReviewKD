import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppcls.utils.initializer import xavier_uniform_, kaiming_normal_, kaiming_uniform_, zeros_


class MobileNet(nn.Layer):
    def __init__(self, classes_num=1000, **kwargs):
        super().__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2D(
                    inp, oup, 3, stride, 1, bias_attr=False),
                nn.BatchNorm2D(oup),
                nn.ReLU())

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2D(
                    inp, inp, 3, stride, 1, groups=inp, bias_attr=False),
                nn.BatchNorm2D(inp),
                nn.ReLU(),
                nn.Conv2D(
                    inp, oup, 1, 1, 0, bias_attr=False),
                nn.BatchNorm2D(oup),
                nn.ReLU(), )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2D(7), )
        self.fc = nn.Linear(1024, classes_num)

        self.init_params()

    def init_params(self, ):
        for _, m in self.named_sublayers():
            if hasattr(m, 'bias') and getattr(m, 'bias') is not None:
                zeros_(m.bias)

            if isinstance(m, nn.Conv2D):
                xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                kaiming_uniform_(m.weight, a=1, reverse=True)

    def forward(self, x, is_feat=False):
        if is_feat:
            return self.extract_feature(x, preReLU=True)
        x = self.model(x)
        x = x.reshape([-1, 1024])
        x = self.fc(x)
        return x

    def get_bn_before_relu(self):
        bn1 = self.model[3][-2]
        bn2 = self.model[5][-2]
        bn3 = self.model[11][-2]
        bn4 = self.model[13][-2]

        return [bn1, bn2, bn3, bn4]

    def get_channel_num(self):

        return [128, 256, 512, 1024]

    def extract_feature(self, x, preReLU=False):

        feat1 = self.model[3][:-1](self.model[0:3](x))
        feat2 = self.model[5][:-1](self.model[4:5](F.relu(feat1)))
        feat3 = self.model[11][:-1](self.model[6:11](F.relu(feat2)))
        feat4 = self.model[13][:-1](self.model[12:13](F.relu(feat3)))

        feat5 = self.model[14](F.relu(feat4))
        out = feat5.reshape([-1, 1024])
        out = self.fc(out)

        if not preReLU:
            feat1 = F.relu(feat1)
            feat2 = F.relu(feat2)
            feat3 = F.relu(feat3)
            feat4 = F.relu(feat4)

        return [feat1, feat2, feat3, feat4, feat5], out