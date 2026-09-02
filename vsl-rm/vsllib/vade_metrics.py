import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

nmi = normalized_mutual_info_score
ari = adjusted_rand_score

from scipy.optimize import linear_sum_assignment

def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    return w[row_ind, col_ind].sum() / y_pred.size

def cluster_acc(Y_pred, Y):
  assert Y_pred.size == Y.size
  D = max(Y_pred.max(), Y.max())+1
  w = np.zeros((D,D), dtype=np.int64)
  for i in range(Y_pred.size):
    w[Y_pred[i], Y[i]] += 1
  row_ind, col_ind = linear_sum_assignment(w.max() - w)

  return w[row_ind, col_ind].sum() / y_pred.size, w