import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score

def compute_dcr_nndr(X_syn, X_train_real):
    """
    Computes Distance to Closest Record (DCR) and Nearest Neighbor Distance Ratio (NNDR)
    for each synthetic record relative to the real training dataset.
    """
    # X_syn: (N_syn, D), X_train_real: (N_train, D)
    dists = cdist(X_syn, X_train_real, metric='euclidean')
    
    # Efficiently partition to get the two smallest elements in each row
    # np.partition is O(N) instead of O(N log N) sorting
    two_smallest = np.partition(dists, 1, axis=1)[:, :2]
    
    # Sort the two smallest elements to get d1 (nearest) and d2 (second-nearest)
    two_smallest_sorted = np.sort(two_smallest, axis=1)
    
    d1 = two_smallest_sorted[:, 0]
    d2 = two_smallest_sorted[:, 1]
    
    # Compute NNDR with division safety
    nndr = np.where(d2 > 0, d1 / d2, 1.0)
    
    metrics = {
        "dcr_values": d1.tolist(),
        "nndr_values": nndr.tolist(),
        "dcr_mean": float(np.mean(d1)),
        "dcr_median": float(np.median(d1)),
        "dcr_5th_percentile": float(np.percentile(d1, 5)),
        "nndr_mean": float(np.mean(nndr)),
        "nndr_median": float(np.median(nndr))
    }
    
    return metrics

def compute_mia_auc(X_train_real, X_test_real, X_syn):
    """
    Computes distance-based Membership Inference Attack (MIA) ROC-AUC.
    Score is the negative distance of each real record to its closest synthetic record.
    """
    # Compute nearest distance to synthetic set for training records
    dists_train = cdist(X_train_real, X_syn, metric='euclidean')
    d_train = dists_train.min(axis=1)
    
    # Compute nearest distance to synthetic set for test records
    dists_test = cdist(X_test_real, X_syn, metric='euclidean')
    d_test = dists_test.min(axis=1)
    
    # Form binary classification problem:
    # 1 for training set members, 0 for test set non-members
    scores = np.concatenate([-d_train, -d_test])
    labels = np.concatenate([np.ones(len(d_train)), np.zeros(len(d_test))])
    
    mia_auc = roc_auc_score(labels, scores)
    
    return float(mia_auc)
