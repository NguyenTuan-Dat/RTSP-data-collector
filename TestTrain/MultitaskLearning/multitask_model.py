from mxnet import is_np_array
from mxnet.gluon import nn
import mxnet as mx


class RELU6(nn.HybridBlock):
    """Relu6 used in MobileNetV2."""

    def __init__(self, **kwargs):
        super(RELU6, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.clip(x, 0, 6, name="relu6")


class Softmax(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(Softmax, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        softmax = F.npx.softmax if is_np_array() else F.softmax
        return softmax(x)

    def __repr__(self):
        return self.__class__.__name__


def _add_conv(out, channels=1, kernel=1, stride=1, pad=0,
              num_group=1, active=True, relu6=False):
    out.add(nn.Conv2D(channels, kernel, stride, pad, groups=num_group, use_bias=False))
    out.add(nn.BatchNorm(scale=True))
    if active:
        out.add(RELU6() if relu6 else nn.Activation('relu'))


class LinearBottleneck(nn.HybridBlock):
    r"""LinearBottleneck used in MobileNetV2 model from the
    `"Inverted Residuals and Linear Bottlenecks:
    Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381>`_ paper.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    channels : int
        Number of output channels.
    t : int
        Layer expansion ratio.
    stride : int
        stride
    """

    def __init__(self, in_channels, channels, t, stride, **kwargs):
        super(LinearBottleneck, self).__init__(**kwargs)
        self.use_shortcut = stride == 1 and in_channels == channels
        with self.name_scope():
            self.out = nn.HybridSequential()

            _add_conv(self.out, in_channels * t, relu6=True)
            _add_conv(self.out, in_channels * t, kernel=3, stride=stride,
                      pad=1, num_group=in_channels * t, relu6=True)
            _add_conv(self.out, channels, active=False, relu6=True)

    def hybrid_forward(self, F, x):
        out = self.out(x)
        if self.use_shortcut:
            out = F.elemwise_add(out, x)
        return out


class MobileNetV2Multitask(nn.HybridBlock):
    r"""MobileNetV2 model from the
    `"Inverted Residuals and Linear Bottlenecks:
    Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381>`_ paper.

    Parameters
    ----------
    multiplier : float, default 1.0
        The width multiplier for controling the model size. The actual number of channels
        is equal to the original channel size multiplied by this multiplier.
    classes : int, default 1000
        Number of classes for the output layer.
    """

    def __init__(self, multiplier=1.0, classes=1000, **kwargs):
        super(MobileNetV2Multitask, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='features_')
            with self.features.name_scope():
                _add_conv(self.features, int(32 * multiplier), kernel=3,
                          stride=2, pad=1, relu6=True)

                in_channels_group = [int(x * multiplier) for x in [32] + [16] + [24] * 2
                                     + [32] * 3 + [64] * 4 + [96] * 3 + [160] * 3]
                channels_group = [int(x * multiplier) for x in [16] + [24] * 2 + [32] * 3
                                  + [64] * 4 + [96] * 3 + [160] * 3 + [320]]
                ts = [1] + [6] * 16
                strides = [1, 2] * 2 + [1, 1, 2] + [1] * 6 + [2] + [1] * 3

                for in_c, c, t, s in zip(in_channels_group, channels_group, ts, strides):
                    self.features.add(LinearBottleneck(in_channels=in_c, channels=c,
                                                       t=t, stride=s))

                last_channels = int(1280 * multiplier) if multiplier > 1.0 else 1280
                _add_conv(self.features, last_channels, relu6=True)

                self.features.add(nn.GlobalAvgPool2D())

            self.output_glasses = nn.HybridSequential(prefix='output_glasses_')
            with self.output_glasses.name_scope():
                _add_conv(self.output_glasses, 128, kernel=1, stride=1, relu6=True)
                _add_conv(self.output_glasses, 2, kernel=1, stride=1)

                self.output_glasses.add(
                    nn.Flatten(),
                    Softmax()
                )

            self.output_mask = nn.HybridSequential(prefix='output_mask_')
            with self.output_mask.name_scope():
                _add_conv(self.output_mask, 128, kernel=1, stride=1, relu6=True)
                _add_conv(self.output_mask, 2, kernel=1, stride=1)
                self.output_mask.add(
                    nn.Flatten(),
                    Softmax()
                )

            # self.output_normal = nn.HybridSequential(prefix='output_normal_')
            # with self.output_normal.name_scope():
            #     self.output_normal.add(
            #         nn.Conv2D(2, 1, use_bias=False, prefix='pred_normal_'),
            #         nn.Flatten(),
            #         Softmax()
            #     )

            self.output_hat = nn.HybridSequential(prefix='output_hat_')
            with self.output_hat.name_scope():
                _add_conv(self.output_hat, 128, kernel=1, stride=1, relu6=True)
                _add_conv(self.output_hat, 2, kernel=1, stride=1)
                self.output_hat.add(
                    nn.Flatten(),
                    Softmax()
                )

    def hybrid_forward(self, F, x):
        x = self.features(x)
        pred_glasses = self.output_glasses(x)
        pred_mask = self.output_mask(x)
        # pred_normal = self.output_normal(x)
        pred_hat = self.output_hat(x)
        # return pred_glasses, pred_mask, pred_normal, pred_hat
        output = mx.symbol.Group([pred_glasses, pred_mask, pred_hat])
        return output
