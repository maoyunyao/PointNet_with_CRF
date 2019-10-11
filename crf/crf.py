#-*- coding: utf-8 -*-

import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels

def crf_process(cloud, gt_prob=0.5, compat=3):
    n_points = cloud.shape[0]
    n_labels = 13
    labels = cloud[:, 3].astype(np.uint32)
    d = dcrf.DenseCRF(n_points, n_labels)

    U = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=False)
    d.setUnaryEnergy(U.astype(np.float32))

    feats = 15 * cloud[:, 0:3].transpose()
    feats = feats.copy(order='C')

    d.addPairwiseEnergy(feats.astype(np.float32), compat=compat,
                    kernel=dcrf.DIAG_KERNEL,
                    normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q_unary = d.inference(12)
    return Q_unary