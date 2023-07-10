import torch
from Nmetrics import evaluate
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import argparse
import random
import time
import torch.nn.functional as F
import load_data as loader
from network import Network
from loss import Proto_Align_Loss, Instance_Align_Loss
from alignment import alignment
from datasets import Data_Sampler, TrainDataset_Com, TrainDataset_All
import os
from sklearn.cluster import KMeans
from utils import NormalizeFeaTorch, get_Similarity, clustering, euclidean_dist
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def seed_everything(SEED=42):  # 应用不同的种子产生可复现的结果
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True  # keep True if all the input have same size.


def pretrain(model, opt_pre, args, device, X_com, Y_com):
    train_dataset = TrainDataset_Com(X_com, Y_com)
    batch_sampler = Data_Sampler(train_dataset, shuffle=True, batch_size=args.batch_size, drop_last=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_sampler=batch_sampler)

    t_progress = tqdm(range(args.pretrain_epochs), desc='Pretraining')
    for epoch in t_progress:
        tot_loss = 0.0
        loss_fn = torch.nn.MSELoss()
        for batch_idx, (xs, ys) in enumerate(train_loader):
            for v in range(args.V):
                xs[v] = torch.squeeze(xs[v]).to(device)
            opt_pre.zero_grad()
            zs, xrs = model(xs)
            loss_list = []
            for v in range(args.V):
                loss_value = loss_fn(xs[v], xrs[v])
                loss_list.append(loss_value)
            loss = sum(loss_list)
            loss.backward()
            opt_pre.step()
            tot_loss += loss.item()
        print('Epoch {}'.format(epoch + 1), 'Loss:{:.6f}'.format(tot_loss / len(train_loader)))

    fea_emb = []
    for v in range(args.V):
        fea_emb.append([])

    all_dataset = TrainDataset_Com(X, Y)
    batch_sampler_all = Data_Sampler(all_dataset, shuffle=False, batch_size=args.batch_size, drop_last=False)
    all_loader = torch.utils.data.DataLoader(dataset=all_dataset, batch_sampler=batch_sampler_all)

    with torch.no_grad():
        for batch_idx2, (xs2, _) in enumerate(all_loader):
            for v in range(args.V):
                xs2[v] = torch.squeeze(xs2[v]).to(device)
            zs2, xrs2 = model(xs2)
            for v in range(args.V):
                zs2[v] = zs2[v].cpu()
                fea_emb[v] = fea_emb[v] + zs2[v].tolist()

    for v in range(args.V):
        fea_emb[v] = torch.tensor(fea_emb[v])

    return fea_emb


def train_align(model, opt_align, args, device, X, Y, Miss_vecs):
    train_dataset = TrainDataset_All(X, Y, Miss_vecs)
    batch_sampler = Data_Sampler(train_dataset, shuffle=True, batch_size=args.Batch_Align, drop_last=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_sampler=batch_sampler)

    t_progress = tqdm(range(args.align_epochs), desc='Alignment')
    for epoch in t_progress:
        for batch_idx, (x, y, miss_vec) in enumerate(train_loader):
            opt_align.zero_grad()
            ###### 计算loss_recon ######
            loss_fn = torch.nn.MSELoss().to(device)
            loss_list_recon = []
            for v in range(args.V):
                x[v] = torch.squeeze(x[v]).to(device)
                y[v] = torch.squeeze(y[v]).to(device)
                miss_vec[v] = torch.squeeze(miss_vec[v]).to(device)
            z, xr = model(x)
            for v in range(args.V):
                loss_list_recon.append(loss_fn(x[v][miss_vec[v]>0], xr[v][miss_vec[v]>0]))
            loss_recon = sum(loss_list_recon)

            ###### 计算loss_ins_align ######
            criterion_ins = Instance_Align_Loss().to(device)
            loss_list_ins = []
            for v1 in range(args.V):
                v2_start = v1 + 1
                for v2 in range(v2_start, args.V):
                    align_index = []
                    for i in range(x[0].shape[0]):
                        if miss_vec[v1][i] == 1 and miss_vec[v2][i] == 1:
                            align_index.append(i)

                    z1 = z[v1][align_index]  # 改
                    z2 = z[v2][align_index]  # 改
                    Dx = F.cosine_similarity(z1, z2, dim=1)
                    gt = torch.ones(z1.shape[0]).to(device)
                    l_tmp2 = criterion_ins(gt, Dx)
                    loss_list_ins.append(l_tmp2)
            loss_ins_align = sum(loss_list_ins)

            criterion_proto = Proto_Align_Loss().to(device)
            loss_list_pro = []
            for v1 in range(args.V):
                v2_start = v1 + 1
                for v2 in range(v2_start, args.V):
                    align_index = []
                    for i in range(z[0].shape[0]):
                        if miss_vec[v1][i] == 1 and miss_vec[v2][i] == 1:
                            align_index.append(i)

                    p1 = z[v1][align_index].t()
                    p2 = z[v2][align_index].t()
                    gt = torch.ones(p1.shape[0]).to(device)
                    Dp = get_Similarity(p1, p2)
                    l_tmp = criterion_proto(gt, Dp)
                    loss_list_pro.append(l_tmp)
            loss_pro_align = sum(loss_list_pro)
            loss_total = loss_recon + para_loss[0] * loss_pro_align + para_loss[1] * loss_ins_align  # 改

            loss_total.backward()
            opt_align.step()

    fea_all = []
    for v in range(args.V):
        fea_all.append([])

    all_dataset = TrainDataset_Com(X, Y)
    batch_sampler_all = Data_Sampler(all_dataset, shuffle=False, batch_size=args.batch_size, drop_last=False)
    all_loader = torch.utils.data.DataLoader(dataset=all_dataset, batch_sampler=batch_sampler_all)

    with torch.no_grad():
        for batch_idx2, (xs2, _) in enumerate(all_loader):
            for v in range(args.V):
                xs2[v] = torch.squeeze(xs2[v]).to(device)
            zs2, xrs2 = model(xs2)
            for v in range(args.V):
                zs2[v] = zs2[v].cpu()
                fea_all[v] = fea_all[v] + zs2[v].tolist()

    for v in range(args.V):
        fea_all[v] = torch.tensor(fea_all[v])

    return fea_all



if __name__=='__main__':
    my_data_dic = loader.ALL_data
    for i_d in my_data_dic:
        data_para = my_data_dic[i_d]  # 改
        print(data_para)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        missrate = 0.5  # 缺失率
        align_epochs = 50
        lr_pre = 0.0005  # 0.0005 预训练学习率
        lr_align = 0.0001  # 0.0001 对齐学习率
        Batch = 256  # 256  预训练阶段batch_size
        Batch_Align = 256  # 对齐阶段batch_size
        para_loss = [1e-3, 1e-3]  # 超参数
        pre_epochs = 200  # 200 预训练epoch
        feature_dim = 10  # embedding维度, 等于clusters

        seed_everything(42)  # 应用不同的种子产生可复现的结果

        parser = argparse.ArgumentParser(description='main')
        parser.add_argument('--dataset', default=data_para)
        parser.add_argument('--batch_size', default=Batch, type=int)
        parser.add_argument('--Batch_Align', default=Batch_Align, type=int)
        parser.add_argument('--missrate', default=missrate, type=float)
        parser.add_argument('--lr_pre', default=lr_pre, type=float)
        parser.add_argument('--lr_align', default=lr_align, type=float)
        parser.add_argument('--para_loss', default=para_loss, type=float)
        parser.add_argument('--pretrain_epochs', default=pre_epochs, type=int)
        parser.add_argument('--align_epochs', default=align_epochs, type=int)
        parser.add_argument("--feature_dim", default=feature_dim)
        parser.add_argument("--V", default=data_para['V'])
        parser.add_argument("--K", default=data_para['K'])
        parser.add_argument("--N", default=data_para['N'])
        parser.add_argument("--view_dims", default=data_para['n_input'])
        parser.add_argument("--view_meaning", default=data_para['view_meaning'])

        args = parser.parse_args()
        print('+' * 30, ' Parameters ', '+' * 30)
        print(args)
        print('+' * 75)

        X, Y, missindex, X_com, Y_com, index_com, index_incom = loader.load_data(args.dataset, args.missrate)
        Miss_vecs = []
        for v in range(args.V):
            Miss_vecs.append(missindex[:, v])

        model = Network(args.V, args.view_dims, args.feature_dim).to(device)
        optimizer_pretrain = torch.optim.Adam(model.parameters(), lr=args.lr_pre)
        fea_emb = pretrain(model, optimizer_pretrain, args, device, X_com, Y_com)
        optimizer_align = torch.optim.Adam(model.parameters(), lr=args.lr_align)
        fea_end = train_align(model, optimizer_align, args, device, X, Y, Miss_vecs)

        for v in range(args.V):
            fea_end[v] = fea_end[v].cpu()
        fea_final = []
        for v in range(args.V):
            fea_final.append([])

        final_batch = 2000  # 改 计算相似矩阵的batch大小
        all_dataset2 = TrainDataset_Com(fea_end, Y)
        batch_sampler_all2 = Data_Sampler(all_dataset2, shuffle=False, batch_size=final_batch, drop_last=False)
        all_loader2 = torch.utils.data.DataLoader(dataset=all_dataset2, batch_sampler=batch_sampler_all2)
        for batch_idx, (xs, ys) in enumerate(all_loader2):
            for v in range(args.V):
                xs[v] = torch.squeeze(xs[v]).to(device)
            # 计算batch内各视图的batchsize x batchsize的余弦相似度矩阵的列表cossim_mat #
            cossim_mat = []
            for v in range(args.V):
                sim_mat = get_Similarity(xs[v], xs[v])
                diag = torch.diag(sim_mat)
                sim_diag = torch.diag_embed(diag)
                sim_mat = sim_mat - sim_diag  # 得到的sim_mat为主对角线为0的相似矩阵
                for i in range(xs[0].shape[0]):
                    if missindex[final_batch * batch_idx + i, v] == 0:
                        sim_mat[:, i] = 0  # 将缺失实例所在的相似度矩阵的整列置为0
                cossim_mat.append(sim_mat)
            # 用最大相似度的完整视图填补缺失视图 #
            for i in range(xs[0].shape[0]):
                for v in range(args.V):
                    if missindex[final_batch * batch_idx + i, v] == 0:
                        vec_tmp = cossim_mat[v][i]
                        _, indices = torch.sort(vec_tmp, descending=True)
                        xs[v][i] = xs[v][indices[0]]  # 改 NN_i最优为0

            for v in range(args.V):
                fea_final[v] = fea_final[v] + xs[v].tolist()

        for v in range(args.V):
            fea_final[v] = torch.tensor(fea_final[v])

        Labels = Y[0]
        estimator = KMeans(n_clusters=args.K)

        fea_cluster = fea_final[0]
        for i in range(1, len(fea_final)):
            fea_cluster = np.concatenate((fea_cluster, fea_final[i]), axis=1)

        estimator.fit(fea_cluster)
        pred_final = estimator.labels_
        acc, nmi, purity, fscore, precision, recall, ari = evaluate(Labels, pred_final)
        print('ACC=%.4f, NMI=%.4f, PUR=%.4f, Fscore=%.4f, Prec=%.4f, Recall=%.4f, ARI=%.4f' %
              (acc, nmi, purity, fscore, precision, recall, ari))





