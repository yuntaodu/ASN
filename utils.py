import numpy as np
import scipy.sparse as sp
import torch
import itertools
from scipy import sparse
import scipy.io as sio
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
import scipy
from sklearn.decomposition import PCA
import torch.nn.functional as F

def adj_matrix(graph):
    nodes = []
    for src, v in graph.items():
        nodes.extend([[src, v_] for v_ in v])
        nodes.extend([[v_, src] for v_ in v])
    nodes = [k for k, _ in itertools.groupby(sorted(nodes))]
    nodes = np.array(nodes)
    return sparse.coo_matrix((np.ones(nodes.shape[0]), (nodes[:, 0], nodes[:, 1])),
                             (len(graph), len(graph)))


def norm_x(x):
    return np.diag(np.power(x.sum(axis=1), -1).flatten()).dot(x)


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def norm_adj_matrix(matrix):
    matrix += sparse.eye(matrix.shape[0])
    degree = np.array(matrix.sum(axis=1))
    d_sqrt = sparse.diags(np.power(degree, -0.5).flatten())
    return d_sqrt.dot(matrix).dot(d_sqrt)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def csr_2_sparse_tensor_tuple(csr_matrix):
    if not isinstance(csr_matrix, scipy.sparse.lil_matrix):
        csr_matrix = lil_matrix(csr_matrix)
    coo_matrix = csr_matrix.tocoo()
    indices = np.transpose(np.vstack((coo_matrix.row, coo_matrix.col)))
    values = coo_matrix.data
    shape = csr_matrix.shape
    return indices, values, shape

def feature_compression(features, dim=200):
    """Preprcessing of features"""
    features = features.toarray()
    feat = lil_matrix(PCA(n_components=dim, random_state=0).fit_transform(features))
    return feat.toarray()

def load_data_citation(path="data/",dataset="citation"):
    file = str(path) + str(dataset)
    net = sio.loadmat(file)
    features, adj, labels = net['attrb'], net['network'], net['group']
    if not isinstance(features, scipy.sparse.lil_matrix):
        features = lil_matrix(features)
    labels = np.array(labels)

    '''compute PPMI'''
    A_k = AggTranProbMat(adj, 3)
    PPMI_ = ComputePPMI(A_k)
    n_PPMI_ = MyScaleSimMat(PPMI_)  # row normalized PPMI
    n_PPMI_mx = lil_matrix(n_PPMI_)
    X_n = sparse_mx_to_torch_sparse_tensor(n_PPMI_mx)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    features = normalize(features.toarray())
    adj = normalize(adj + sp.eye(adj.shape[0]))
    features = torch.FloatTensor(features)
    labels = np.argmax(labels, 1)
    labels = torch.LongTensor(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    idxs = np.random.permutation(len(features))
    idx_all = torch.LongTensor(idxs)

    return adj, features, labels,idx_all,X_n

def load_network(path="data/",dataset="citation"):
    file = str(path) + str(dataset)
    net = sio.loadmat(file)
    X, A, Y = net['attrb'], net['network'], net['group']
    if not isinstance(X, scipy.sparse.lil_matrix):
        X = lil_matrix(X)
    return A, X, Y

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

class ConditionalEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(ConditionalEntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(dim=1)
        return -1.0 * b.mean(dim=0)

def MyScaleSimMat(W):
    '''L1 row norm of a matrix'''
    rowsum = np.array(np.sum(W, axis=1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    W = r_mat_inv.dot(W)
    return W

def AggTranProbMat(G, step):
    '''aggregated K-step transition probality'''
    G = MyScaleSimMat(G)
    G = csc_matrix.toarray(G)
    A_k = G
    A = G
    for k in np.arange(2, step + 1):
        A_k = np.matmul(A_k, G)
        A = A + A_k / k
    return A

def ComputePPMI(A):
    '''compute PPMI, given aggregated K-step transition probality matrix as input'''
    np.fill_diagonal(A, 0)
    A = MyScaleSimMat(A)
    (p, q) = np.shape(A)
    col = np.sum(A, axis=0)
    col[col == 0] = 1
    PPMI = np.log((float(p) * A) / col[None, :])
    IdxNan = np.isnan(PPMI)
    PPMI[IdxNan] = 0
    PPMI[PPMI < 0] = 0
    return PPMI


def load_adj_label_for_reconstruction(dataset_name):
    A,_,_ = load_network(dataset=dataset_name)

    adj_label = A + sp.eye(A.shape[0])
    adj_label = torch.FloatTensor(adj_label.toarray())

    pos_weight = float(A.shape[0] * A.shape[0] - A.sum()) / A.sum()
    pos_weight = np.array(pos_weight).reshape(1, 1)
    pos_weight = torch.from_numpy(pos_weight)
    norm = A.shape[0] * A.shape[0] / float((A.shape[0] * A.shape[0] - A.sum()) * 2)

    return adj_label,pos_weight,norm