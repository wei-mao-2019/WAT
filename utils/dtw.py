"""adapted from https://github.com/pierre-rouanet/dtw/blob/master/dtw/dtw.py"""

from numpy import array, zeros, full, argmin, inf, ndim
from scipy.spatial.distance import cdist
from math import isinf
import numpy as np
import torch
import time
from numba import njit, prange, cuda
import math

def dtw(x, y, dist, warp=1, w=inf, s=1.0):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.
    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    :param int warp: how many shifts are computed.
    :param int w: window size limiting the maximal distance between indices of matched entries |i,j|.
    :param float s: weight applied on off-diagonal moves of the path. As s gets larger, the warping path is increasingly biased towards the diagonal
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    assert isinf(w) or (w >= abs(len(x) - len(y)))
    assert s > 0
    r, c = len(x), len(y)
    if not isinf(w):
        D0 = full((r + 1, c + 1), inf)
        for i in range(1, r + 1):
            D0[i, max(1, i - w):min(c + 1, i + w + 1)] = 0
        D0[0, 0] = 0
    else:
        D0 = zeros((r + 1, c + 1))
        D0[0, 1:] = inf
        D0[1:, 0] = inf
    D1 = D0[1:, 1:]  # view
    for i in range(r):
        for j in range(c):
            if (isinf(w) or (max(0, i - w) <= j <= min(c, i + w))):
                D1[i, j] = dist(x[i], y[j])
    C = D1.copy()
    jrange = range(c)
    for i in range(r):
        if not isinf(w):
            jrange = range(max(0, i - w), min(c, i + w + 1))
        for j in jrange:
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                i_k = min(i + k, r)
                j_k = min(j + k, c)
                min_list += [D0[i_k, j] * s, D0[i, j_k] * s]
            D1[i, j] += min(min_list)
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1], C, D1, path


def accelerated_dtw(x, y, dist, warp=1):
    """
    Computes Dynamic Time Warping (DTW) of two sequences in a faster way.
    Instead of iterating through each element and calculating each distance,
    this uses the cdist function from scipy (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)
    :param array x: N1*M array
    :param array y: N2*M array
    :param string or func dist: distance parameter for cdist. When string is given, cdist uses optimized functions for the distance metrics.
    If a string is passed, the distance function can be 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.
    :param int warp: how many shifts are computed.
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    if ndim(x) == 1:
        x = x.reshape(-1, 1)
    if ndim(y) == 1:
        y = y.reshape(-1, 1)
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:]
    D0[1:, 1:] = cdist(x, y, dist)
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                min_list += [D0[min(i + k, r), j],
                             D0[i, min(j + k, c)]]
            D1[i, j] += min(min_list)
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1], C, D1, path


def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)


def batch_dtw_torch(x, y):
    """
    Computes Dynamic Time Warping (DTW) of two sequences in a faster way.
    Instead of iterating through each element and calculating each distance,
    this uses the cdist function from scipy (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)
    :param array x: batch*N1*M array
    :param array y: batch*N2*M array
    :param string or func dist: distance parameter for cdist. When string is given, cdist uses optimized functions for the distance metrics.
    If a string is passed, the distance function can be 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.
    :param int warp: how many shifts are computed.
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x.shape) == 3
    assert len(y.shape) == 3
    bs,r,_ = x.shape
    _,c,_ = y.shape
    D0 = torch.zeros([bs,r + 1, c + 1],dtype=x.dtype,device=x.device)
    D0[:,0, 1:] = 1e10
    D0[:,1:, 0] = 1e10
    D1 = D0[:,1:, 1:]
    st = time.time()
    D0[:,1:, 1:] = torch.cdist(x, y, p=2, compute_mode='donot_use_mm_for_euclid_dist')
    print(f'cdist time {time.time()-st:.1f}')
    # D0[:,1:, 1:] = torch.norm(x[:,:,None,:]-y[:,None,:,:],dim=-1)
    # C = D1.clone()

    # the following use cpu
    st = time.time()
    D1 = D1.cpu()
    D0 = D0.cpu()
    len_path = torch.ones_like(D0)
    len_path[D0==0] = 0
    len_path[D0>=1e5] = 0
    for i in range(r):
        for j in range(c):
            min_list = torch.cat([D0[:,i, j:j+1],
                                  D0[:,min(i + 1, r), j:j+1],
                                  D0[:,i:i+1, min(j + 1, c)]],dim=1)
            tmp = torch.min(min_list,dim=1)
            min_val = tmp[0]
            min_idx = tmp[1]
            D1[:, i, j] += min_val
            len_path[min_idx==0,i+1,j+1] += len_path[min_idx==0,i, j]
            len_path[min_idx==1,i+1,j+1] += len_path[min_idx==1,min(i + 1, r), j]
            len_path[min_idx==2,i+1,j+1] += len_path[min_idx==2,i,min(j + 1, c)]
    print(f'intergrate the cost {time.time()-st:.1f}')
    # if len(x) == 1:
    #     path = zeros(len(y)), range(len(y))
    # elif len(y) == 1:
    #     path = range(len(x)), zeros(len(x))
    # else:
    # path_p,path_q,seq_len = batch_traceback_torch(D0)
    # seq_len = torch.from_numpy(seq_len).to(dtype=x.dtype,device=x.device)
    seq_len = len_path[:,-1,-1]
    return D1[:,-1, -1], seq_len


def batch_dtw_torch_parallel(x, y, seq_x, seq_y,power=1):
    """
    Computes Dynamic Time Warping (DTW) of two sequences in a faster way.
    Instead of iterating through each element and calculating each distance,
    this uses the cdist function from scipy (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)
    :param array x: batch*N1*M array
    :param array y: batch*N2*M array
    :param string or func dist: distance parameter for cdist. When string is given, cdist uses optimized functions for the distance metrics.
    If a string is passed, the distance function can be 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.
    :param int warp: how many shifts are computed.
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x.shape) == 3
    assert len(y.shape) == 3
    bs,r,_ = x.shape
    _,c,_ = y.shape
    # D1 = D0[:,1:, 1:]
    # st = time.time()
    dtmp = torch.cdist(x, y, p=2, compute_mode='donot_use_mm_for_euclid_dist')
    # print(f'cdist time {time.time()-st:.1f}')
    # D0[:,1:, 1:] = torch.norm(x[:,:,None,:]-y[:,None,:,:],dim=-1)
    # C = D1.clone()
    # torch.cuda.synchronize()

    # the following use cpu parallel
    # st = time.time()
    # D1 = D1.cpu().data.numpy()
    d0 = np.zeros([bs, r + 1, c + 1])
    d0[:, 0, 1:] = 1e10
    d0[:, 1:, 0] = 1e10
    d0[:,1:,1:] = dtmp.cpu().data.numpy()

    len_tmp = np.ones_like(d0)
    len_tmp[d0==0] = 0
    len_tmp[d0>=1e5] = 0
    cost = np.zeros([bs])
    path_len = np.zeros([bs])
    seq_x = seq_x.astype(np.int)
    seq_y = seq_y.astype(np.int)
    cost, path_len = integrate_cost(d0,len_tmp,seq_len1=seq_x,seq_len2=seq_y,
                                   cost=cost,path_len=path_len)
    # print(f'intergrate the cost {time.time()-st:.1f}')



    # if len(x) == 1:
    #     path = zeros(len(y)), range(len(y))
    # elif len(y) == 1:
    #     path = range(len(x)), zeros(len(x))
    # else:
    # path_p,path_q,seq_len = batch_traceback_torch(D0)
    # seq_len = torch.from_numpy(seq_len).to(dtype=x.dtype,device=x.device)
    # seq_len = len_path[:,-1,-1]
    return cost, path_len


def batch_dtw_torch_parallel_custom_dist(x, y, seq_x, seq_y,power=1):
    """
    Computes Dynamic Time Warping (DTW) of two sequences in a faster way.
    Instead of iterating through each element and calculating each distance,
    this uses the cdist function from scipy (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)
    :param array x: batch*N1*M array
    :param array y: batch*N2*M array
    :param string or func dist: distance parameter for cdist. When string is given, cdist uses optimized functions for the distance metrics.
    If a string is passed, the distance function can be 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.
    :param int warp: how many shifts are computed.
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x.shape) == 3
    assert len(y.shape) == 3
    bs,r,_ = x.shape
    _,c,_ = y.shape
    # D1 = D0[:,1:, 1:]
    # st = time.time()
    dtmp = []
    sidx = 0
    sbs = 10
    x = x.reshape(bs,r,-1,3)
    y = y.reshape(bs,r,-1,3)
    while True:
        x_tmp = x[sidx:sidx+sbs]
        y_tmp = y[sidx:sidx+sbs]
        if x_tmp.shape[0] == 0:
            break
        dtmp.append(torch.norm(x_tmp[:,:,None,:,:]-y_tmp[:,None,:,:,:],dim=-1).mean(dim=-1))
        sidx += sbs
    dtmp = torch.cat(dtmp,dim=0)
    # dtmp = torch.cdist(x, y, p=2, compute_mode='donot_use_mm_for_euclid_dist')
    # print(f'cdist time {time.time()-st:.1f}')
    # D0[:,1:, 1:] = torch.norm(x[:,:,None,:]-y[:,None,:,:],dim=-1)
    # C = D1.clone()
    # torch.cuda.synchronize()

    # the following use cpu parallel
    # st = time.time()
    # D1 = D1.cpu().data.numpy()
    d0 = np.zeros([bs, r + 1, c + 1])
    d0[:, 0, 1:] = 1e10
    d0[:, 1:, 0] = 1e10
    d0[:,1:,1:] = dtmp.cpu().data.numpy()

    len_tmp = np.ones_like(d0)
    len_tmp[d0==0] = 0
    len_tmp[d0>=1e5] = 0
    cost = np.zeros([bs])
    path_len = np.zeros([bs])
    seq_x = seq_x.astype(np.int)
    seq_y = seq_y.astype(np.int)
    cost, path_len = integrate_cost(d0,len_tmp,seq_len1=seq_x,seq_len2=seq_y,
                                   cost=cost,path_len=path_len)
    # print(f'intergrate the cost {time.time()-st:.1f}')



    # if len(x) == 1:
    #     path = zeros(len(y)), range(len(y))
    # elif len(y) == 1:
    #     path = range(len(x)), zeros(len(x))
    # else:
    # path_p,path_q,seq_len = batch_traceback_torch(D0)
    # seq_len = torch.from_numpy(seq_len).to(dtype=x.dtype,device=x.device)
    # seq_len = len_path[:,-1,-1]
    return cost, path_len

def batch_dtw_cpu_parallel(x, y, seq_x, seq_y):
    """
    Computes Dynamic Time Warping (DTW) of two sequences in a faster way.
    Instead of iterating through each element and calculating each distance,
    this uses the cdist function from scipy (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)
    :param array x: batch*N1*M array
    :param array y: batch*N2*M array
    :param string or func dist: distance parameter for cdist. When string is given, cdist uses optimized functions for the distance metrics.
    If a string is passed, the distance function can be 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.
    :param int warp: how many shifts are computed.
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x.shape) == 3
    assert len(y.shape) == 3
    bs,r,_ = x.shape
    _,c,_ = y.shape
    d0 = np.zeros([bs,r + 1, c + 1])
    # d0[0,0,0] = 0
    d0[:,0, 1:] = 1e10
    d0[:,1:, 0] = 1e10
    seq_x = seq_x.astype(np.int)
    seq_y = seq_y.astype(np.int)
    # D1 = D0[:,1:, 1:]
    # for i in range(10):
    # st = time.time()
    threadsperblock = (8,8,8)
    blockspergrid_x = math.ceil(bs / threadsperblock[0])
    blockspergrid_y = math.ceil(r / threadsperblock[1])
    blockspergrid_z = math.ceil(c / threadsperblock[2])
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
    tmp = np.zeros([bs, r, c])
    cdist_parallel_cuda[blockspergrid, threadsperblock](np.ascontiguousarray(x), np.ascontiguousarray(y), tmp, seq_x, seq_y)
    d0[:,1:,1:] = tmp
    # print(f"{time.time()-st:.3f}")
    
    # st = time.time()
    # d0[:,1:,1:] = cdist_parallel(x,y,d0[:,1:,1:],seq_x,seq_y)
    # print(f"{time.time() - st:.3f}")

    # # validation
    # xx = torch.from_numpy(x).cuda()
    # yy = torch.from_numpy(y).cuda()
    # dtmp = torch.cdist(xx, yy, p=2, compute_mode='donot_use_mm_for_euclid_dist')
    # dtmp = dtmp.cpu().data.numpy()
    # d0[:,1:,1:] = dtmp
    # print(f'cdist time {time.time()-st:.1f} err {np.max(np.abs(dtmp-tmp))}')

    # the following use cpu parallel
    # st = time.time()
    # D1 = D1.cpu().data.numpy()
    # d0 = D0.cpu().data.numpy()
    len_tmp = np.ones_like(d0)
    len_tmp[d0==0] = 0
    len_tmp[d0>=1e5] = 0
    cost = np.zeros([bs])
    path_len = np.zeros([bs])
    # seq_x = seq_x.astype(np.int)
    # seq_y = seq_y.astype(np.int)
    cost, path_len = integrate_cost(d0,len_tmp,seq_len1=seq_x,seq_len2=seq_y,
                                   cost=cost,path_len=path_len)
    # print(f'intergrate the cost {time.time()-st:.1f}')



    # if len(x) == 1:
    #     path = zeros(len(y)), range(len(y))
    # elif len(y) == 1:
    #     path = range(len(x)), zeros(len(x))
    # else:
    # path_p,path_q,seq_len = batch_traceback_torch(D0)
    # seq_len = torch.from_numpy(seq_len).to(dtype=x.dtype,device=x.device)
    # seq_len = len_path[:,-1,-1]
    return cost, path_len


@njit(parallel=True)
def cdist_parallel(x, y, cost,seq_x,seq_y, compute_mode='l2'):
    """
    x bs*seq_n1*feat
    y bs*seq_n2*feat
    cost: bs*seq1*seq2
    seq_len1: bs
    seq_len2: bs
    """
    bs,seq1,c = x.shape
    _,seq2,_ = y.shape
    for bb in prange(bs):
        # cost[bb] = np.sqrt(np.sum((np.expand_dims(x[bb],axis=1) - np.expand_dims(y[bb],axis=0)) ** 2,axis=2))
        r = seq_x[bb]
        c = seq_y[bb]
        for i in prange(r):
            for j in prange(c):
                cost[bb,i,j] = np.sqrt(np.sum((x[bb,i]-y[bb,j])**2))
    return cost


@cuda.jit
def cdist_parallel_cuda(seqx, seqy, cost, seqlx, seqly):
    """
    x bs*seq_n1*feat
    y bs*seq_n2*feat
    cost: bs*seq1*seq2
    seq_len1: bs
    seq_len2: bs
    """

    x,y,z = cost.shape
    hn = seqx.shape[-1]

    # tx = cuda.threadIdx.x
    # ty = cuda.threadIdx.y
    # tz = cuda.threadIdx.z
    #
    # bx = cuda.blockIdx.x
    # by = cuda.blockIdx.y
    # bz = cuda.blockIdx.z
    #
    # tpbx = cuda.blockDim.x  # number of threads per block
    # bpgx = cuda.gridDim.x    # number of blocks in the grid
    #
    # tpby = cuda.blockDim.y  # number of threads per block
    # bpgy = cuda.gridDim.y    # number of blocks in the grid
    #
    # tpbz = cuda.blockDim.z  # number of threads per block
    # bpgz = cuda.gridDim.z    # number of blocks in the grid
    #
    # sx = tx + bx * tpbx
    # stx = tpbx * bpgx
    #
    # sy = ty + by * tpby
    # sty = tpby * bpgy
    #
    # sz = tz + bz * tpbz
    # stz = tpbz * bpgz

    sx,sy,sz = cuda.grid(3)
    stx,sty,stz = cuda.gridsize(3)


    for i in range(sx,x,stx):
        ry = seqlx[i]
        rz = seqly[i]
        for j in range(sy,ry,sty):
            for k in range(sz,rz,stz):
                tmp = 0.
                for l in range(hn):
                    tmp += (seqx[i,j,l]-seqy[i,k,l])**2
                cost[i, j, k] = math.sqrt(tmp)

@njit(parallel=True)
def integrate_cost(D0, len_path, seq_len1=None, seq_len2=None,
                    cost=None,path_len=None):
    """
    D0 bs*seq_n*seq_m
    seq_len1: bs
    seq_len2: bs
    """
    D1 = D0[:,1:,1:]
    bs,r,c = D1.shape
    for bb in prange(bs):
        d1 = D1[bb]
        d0 = D0[bb]
        lp = len_path[bb]
        r = seq_len1[bb]
        c = seq_len2[bb]
        for i in range(r):
            for j in range(c):
                min_idx = 0
                min_val = d0[i, j]

                if d0[min(i + 1, r), j] < min_val:
                    min_idx = 1
                    min_val = d0[min(i + 1, r), j]

                if d0[i, min(j + 1, c)] < min_val:
                    min_idx = 2
                    min_val = d0[i, min(j + 1, c)]
                # min_val = min(d0[i, j],d0[min(i + 1, r), j],d0[i, min(j + 1, c)])
                # min_idx = np.argmin([d0[i, j],d0[min(i + 1, r), j],d0[i, min(j + 1, c)]])
                d1[i, j] += min_val
                if min_idx == 0:
                    lp[i+1,j+1] += lp[i,j]
                elif min_idx == 1:
                    lp[i+1,j+1] += lp[min(i + 1, r), j]
                elif min_idx ==2:
                    lp[i+1,j+1] += lp[i, min(j + 1, c)]
        cost[bb] = d1[i,j]
        path_len[bb] = lp[i+1,j+1]
    return cost, path_len

def batch_traceback_torch(D):
    bs, i, j = D.shape

    path_p = []
    path_q = []
    seq_len = []

    for b in range(bs):
        dd = D[b]
        it = i - 2
        jt = j - 2
        p, q = [it], [jt]
        while (it > 0) or (jt > 0):
            tb = torch.min(torch.hstack([dd[it, jt], dd[it, jt + 1], dd[it + 1, jt]]),dim=0)[1]

            if tb == 0:
                it -= 1
                jt -= 1
            elif tb == 1:
                it -= 1
            else:  # (tb == 2):
                jt -= 1
            p.insert(0, it)
            q.insert(0, jt)
        seq_len.append(len(p))
        path_p.append(array(p))
        path_q.append(array(q))

    return path_p, path_q, array(seq_len)


if __name__ == '__main__':
    # w = inf
    # s = 1.0
    # manhattan_distances = lambda x,y: np.abs(x-y)
    # euclidean_distances = lambda x,y: np.linalg.norm(x-y, axis=-1)
    # if 1:  # 1-D numeric
    #     # from sklearn.metrics.pairwise import manhattan_distances
    #     x = np.array([0, 0, 1, 1, 2, 4, 2, 1, 2, 0])
    #     y = np.array([1, 1, 1, 2, 2, 2, 2, 3, 2, 0])
    #     # x = np.random.randn(1000)
    #     # y = np.random.randn(1000)
    #     dist_fun = manhattan_distances
    #     # w = 2
    #     # s = 1.2
    # elif 0:  # 2-D numeric
    #     # from sklearn.metrics.pairwise import euclidean_distances
    #     x = np.array([[0, 0], [0, 1], [1, 1], [1, 2], [2, 2], [4, 3], [2, 3], [1, 1], [2, 2], [0, 1]])
    #     y = np.array([[1, 0], [1, 1], [1, 1], [2, 1], [4, 3], [4, 3], [2, 3], [3, 1], [1, 2], [1, 0]])
    #     dist_fun = euclidean_distances
    # else:  # 1-D list of strings
    #     # from nltk.metrics.distance import edit_distance
    #     # x = ['we', 'shelled', 'clams', 'for', 'the', 'chowder']
    #     # y = ['class', 'too']
    #     x = ['i', 'soon', 'found', 'myself', 'muttering', 'to', 'the', 'walls']
    #     y = ['see', 'drown', 'himself']
    #     # x = 'we talked about the situation'.split()
    #     # y = 'we talked about the situation'.split()
    #     dist_fun = edit_distance
    import time
    # st = time.time()
    # dist, cost, acc, path = dtw(x, y, dist_fun, w=w, s=s)
    # print(f'dtw time used {time.time()-st:.1f}')
    #
    # st = time.time()
    # dist1, cost1, acc1, path1 = accelerated_dtw(x, y, dist_fun)
    # print(f'fast dtw time used {time.time()-st:.1f}')
    import torch
    x = torch.randn([100,1000,2]).float().cuda()
    y = torch.randn([100,800,2]).float().cuda()

    st = time.time()
    cost, seq_len = batch_dtw_torch_parallel(x, y)
    print(f'fast dtw time used {time.time()-st:.1f}')
    dist_fun = lambda x,y: np.abs(x-y)
    cost, C, D1, path = accelerated_dtw(x[0], y[0],dist_fun)

    print(1)
    # # Vizualize
    # from matplotlib import pyplot as plt
    # plt.imshow(cost.T, origin='lower', cmap=plt.cm.Reds, interpolation='nearest')
    # plt.plot(path[0], path[1], '-o')  # relation
    # plt.xticks(range(len(x)), x)
    # plt.yticks(range(len(y)), y)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.axis('tight')
    # if isinf(w):
    #     plt.title('Minimum distance: {}, slope weight: {}'.format(dist, s))
    # else:
    #     plt.title('Minimum distance: {}, window widht: {}, slope weight: {}'.format(dist, w, s))
    # plt.show()