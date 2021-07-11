from mxnet import is_np_array
from mxnet.gluon.loss import Loss
import numpy as np


class MultitaskSoftmaxCrossEntropyLoss(Loss):
    def __init__(self, axis=-1, sparse_label=True, from_logits=False, weight=None,
                 batch_axis=0, **kwargs):
        super(MultitaskSoftmaxCrossEntropyLoss, self).__init__(
            weight, batch_axis, **kwargs)
        self._axis = axis
        self._sparse_label = sparse_label
        self._from_logits = from_logits

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        new_label = np.zeros((3, 2), dtype=np.int)
        new_label[new_label != label][0] = 1
        new_label[label][1] = 1
        
        if is_np_array():
            log_softmax = F.npx.log_softmax
            pick = F.npx.pick
        else:
            log_softmax = F.log_softmax
            pick = F.pick
        if not self._from_logits:
            pred = log_softmax(pred, self._axis)
        if self._sparse_label:
            loss = -pick(pred, label, axis=self._axis, keepdims=True)
        else:
            label = _reshape_like(F, label, pred)
            loss = -(pred * label).sum(axis=self._axis, keepdims=True)
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        if is_np_array():
            if F is ndarray:
                return loss.mean(axis=tuple(range(1, loss.ndim)))
            else:
                return F.npx.batch_flatten(loss).mean(axis=1)
        else:
            return loss.mean(axis=self._batch_axis, exclude=True)
