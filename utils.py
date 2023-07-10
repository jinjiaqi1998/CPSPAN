import torch
import numpy as np
import warnings
import random
import torch.nn.functional as F
import scipy.sparse as sp
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from kmeans_gpu import kmeans

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 得到归一化的向量在二维中位于单位圆内, 在高维中位于超球内, 且向量模为1
def NormalizeFeaTorch(features):  # features为nxd维的特征矩阵
    rowsum = torch.tensor((features ** 2).sum(1))
    r_inv = torch.pow(rowsum, -0.5)
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    normalized_feas = torch.mm(r_mat_inv, features)
    return normalized_feas


# 求特征矩阵fea_mat1(nxd)和fea_mat2(nxd)的行向量之间的余弦相似度，得similarity为nxn的相似度矩阵, 元素取值范围为[-1,1]
def get_Similarity(fea_mat1, fea_mat2):
    Sim_mat = F.cosine_similarity(fea_mat1.unsqueeze(1), fea_mat2.unsqueeze(0), dim=-1)
    return Sim_mat


def clustering(feature, cluster_num):
    # predict_labels,  cluster_centers = kmeans(X=feature, num_clusters=cluster_num, distance='euclidean', device=torch.device('cuda'))
    predict_labels,  cluster_centers = kmeans(X=feature, num_clusters=cluster_num, distance='euclidean', device=device)
    return predict_labels.numpy(), cluster_centers
    # acc, nmi, ari, f1 = eva(true_labels, predict_labels.numpy(), show_details=False)
    # return 100 * acc, 100 * nmi, 100 * ari, 100 * f1, predict_labels.numpy(), initial



def euclidean_dist(x, y, root=False):
    """
    Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
    Returns:
        dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    if root:
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

