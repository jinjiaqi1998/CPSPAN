import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, v_measure_score
from sklearn import metrics
from munkres import Munkres
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment


def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))

    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:,j]) # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i,j]
            cost_matrix[j,i] = s-t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    clusterLabels = np.zeros(n_clusters)
    for i in range(n_clusters):
        clusterLabels[i] = indices[i][1]
    return clusterLabels


def get_y_preds(y_true, cluster_assignments, n_clusters):

    confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments, labels=None)
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)

    if np.min(cluster_assignments)!=0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred


def fmetric(y_true, y_pred, n_clusters):
    y_pred_ajusted = get_y_preds(y_true, y_pred, n_clusters)
    # F-score
    f_score = metrics.f1_score(y_true, y_pred_ajusted, average='weighted')
    #f_score = float(np.round(f_score, decimals))
    precision = metrics.precision_score(y_true, y_pred_ajusted, average='weighted')
    recall = metrics.recall_score(y_true, y_pred_ajusted, average='macro')

    return f_score, precision, recall



def Purity_score(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    purity = metrics.accuracy_score(y_true, y_voted_labels)
    return purity


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # ind = linear_assignment(w.max() - w)
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
    return acc


def evaluate(truth, prediction):
    unique = np.unique(truth)
    n_clusters = np.size(unique, axis=0)
    nmi = metrics.normalized_mutual_info_score(truth, prediction)
    acc = cluster_acc(truth, prediction)
    purity = Purity_score(truth, prediction)
    fscore, precision, recall = fmetric(truth, prediction, n_clusters)
    ari = adjusted_rand_score(truth, prediction)
    # print(acc, nmi, purity, fscore, precision, recall)
    acc = float(np.round(acc, 4))
    nmi = float(np.round(nmi, 4))
    purity = float(np.round(purity, 4))
    fscore = float(np.round(fscore, 4))
    precision = float(np.round(precision, 4))
    recall = float(np.round(recall, 4))
    ari = float(np.round(ari, 4))
    return acc, nmi, purity, fscore, precision, recall, ari

