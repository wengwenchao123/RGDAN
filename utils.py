import numpy as np
import pandas as pd
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
import pickle

# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

# metric
def metric(pred, label):
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
    return mae, rmse, mape

def seq2instance(data, P, Q):
    num_step, dims = data.shape
    num_sample = num_step - P - Q + 1         #这里减一是为了方便预测特定的一步，做真正的多步预测要把这里的-1和y的+1去掉
    x = np.zeros(shape = (num_sample, P, dims))
    y = np.zeros(shape = (num_sample, Q, dims))
    for i in range(num_sample):
        x[i] = data[i : i + P]
        y[i] = data[i + P : i + P + Q ]
    return x, y

def loadData(args):
    # Traffic
    if args.dataset == 'PeMS' or args.dataset == 'METR':
        TRAFFIC_FILE = args.path+'data/'+args.dataset+'.h5'
        SE_FILE = args.path+'data/SE('+args.dataset+').txt'
        df = pd.read_hdf(TRAFFIC_FILE)
        Traffic = df.values
    elif args.dataset == 'BJ500':
        df = pd.read_csv('data/BJ500.csv', header=0, index_col=0)
        df.index = pd.to_datetime(df.index)
        SE_FILE = args.path + 'data/SE(' + args.dataset + ').txt'
        Traffic = df.values


    print("Initial loaded traffic Shape is: ", Traffic.shape)
    # train/val/test
    num_step = df.shape[0]
    train_steps = round(args.train_ratio * num_step)
    # train_steps1 = round(0.7 * num_step)
    test_steps = round(args.test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps
    train = Traffic[: train_steps]
    val = Traffic[train_steps : train_steps + val_steps]
    test = Traffic[-test_steps :]
    # X, Y 
    trainX, trainY = seq2instance(train, args.P, args.Q)
    valX, valY = seq2instance(val, args.P, args.Q)
    testX, testY = seq2instance(test, args.P, args.Q)

    print("trainX Shape is: ", trainX.shape)
    print("trainY Shape is: ", trainY.shape)
    print("valX Shape is: ", valX.shape)
    print("valY Shape is: ", valY.shape)
    # normalization
    mean, std = np.mean(trainX), np.std(trainX)
    trainX = (trainX - mean) / std
    valX = (valX - mean) / std
    testX = (testX - mean) / std

    # spatial embedding 
    f = open(SE_FILE, mode = 'r')
    lines = f.readlines()
    temp = lines[0].split(' ')
    N, dims = int(temp[0]), int(temp[1])
    SE = np.zeros(shape = (N, dims), dtype = np.float32)
    for line in lines[1 :]:
        temp = line.split(' ')
        index = int(temp[0])
        SE[index] = temp[1 :]

    print("SE Shape is: ", SE.shape)
    # temporal embedding
    Time = df.index
    dayofweek =  np.reshape(Time.weekday, newshape = (-1, 1))
    timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) \
                // 300 #Time.freq.delta.total_seconds()
    #timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second)
    timeofday = np.reshape(timeofday, newshape = (-1, 1))
    Time = np.concatenate((dayofweek, timeofday), axis = -1)
    # train/val/test
    train = Time[: train_steps]
    val = Time[train_steps : train_steps + val_steps]
    test = Time[-test_steps :]
    # shape = (num_sample, P + Q, 2)
    trainTE = seq2instance(train, args.P, args.Q)
    trainTE = np.concatenate(trainTE, axis = 1).astype(np.int32)
    valTE = seq2instance(val, args.P, args.Q)
    valTE = np.concatenate(valTE, axis = 1).astype(np.int32)
    testTE = seq2instance(test, args.P, args.Q)
    testTE = np.concatenate(testTE, axis = 1).astype(np.int32)

    print("train Shape is: ", train.shape)
    print("trainTE Shape is: ", trainTE.shape)
    print("valTE Shape is: ", valTE.shape)
    return (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY,
            SE, mean, std)

# [A * D^(-1/2)]^T * D^(-1/2) = D^(-1/2) * A * D^(-1/2)
def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)   # 转换为coordinate形式的压缩邻接矩阵
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()    # 将n*1的矩阵转换为1个向量
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)   # ndarray类型
    # toarray returns an ndarray; todense returns a matrix. If you want a matrix, use todense otherwise, use toarray
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

# D^(-1/2) * A
def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()    #行向量相加
    d_inv = np.power(rowsum, -1).flatten()     #取每个元素的-1次平方
    d_inv[np.isinf(d_inv)] = 0.                #溢出部分赋值为0
    d_mat= sp.diags(d_inv)                     #变成一个对角矩阵形式应该就是对应那个D
    return d_mat.dot(adj).astype(np.float32).todense()   #这里应该是等于np.dot(d_mat,adj) ，做矩阵乘法

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        # ‘LM’ : Largest (in magnitude) eigenvalues.
        # 返回1个绝对值最大的特征值与特征向量
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    # 转换为稀疏矩阵
    L = sp.csr_matrix(L)
    M, _ = L.shape         # 原始矩阵的行数
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_adj(pkl_filename, adjtype):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return sensor_ids, sensor_id_to_ind, adj
