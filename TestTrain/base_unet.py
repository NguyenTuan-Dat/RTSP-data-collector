from mxnet import is_np_array
from mxnet.gluon import nn


class Softmax(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(Softmax, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        softmax = F.npx.softmax if is_np_array() else F.softmax
        return softmax(x)

    def __repr__(self):
        return self.__class__.__name__


class BaseUnet(nn.HybridSequential):
    def __init__(self, num_classes=3, prefix=None, params=None):
        super(BaseUnet, self).__init__(prefix=prefix, params=params)

        self.num_classes = num_classes

        self.layers = nn.HybridSequential()

        self.layers.add(nn.Conv2D(64, kernel_size=1))
        self.layers.add(nn.Conv2D(128, kernel_size=1))
        self.layers.add(nn.Conv2D(256, kernel_size=1))
        self.layers.add(nn.Conv2D(512, kernel_size=1))
        self.layers.add(nn.AvgPool2D(pool_size=7))
        self.layers.add(nn.BatchNorm())
        self.layers.add(nn.Conv2D(128, kernel_size=1))
        self.layers.add(nn.Conv2D(self.num_classes, kernel_size=1))
        self.layers.add(nn.Flatten())
        self.layers.add(Softmax())

    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.layers(x)
