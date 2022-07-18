import numpy as np
from numba import njit,prange

# @njit(parallel=True)
def acceleration(seq, seq_len, acce):
    """
    seq: bs * seq_max * dim
    seq_len: bs
    """
    bs = seq.shape[0]
    # acce = np.zeros([bs])
    for i in range(bs):
        len_tmp = seq_len[i]
        seq_tmp = seq[i, :len_tmp]
        acce_tmp = seq_tmp[:-1]-seq_tmp[1:]
        acce_tmp = acce_tmp[:-1]-acce_tmp[1:]
        # acce[i] = np.mean(np.sqrt(np.sum(acce_tmp**2,axis=1)))
        acce[i] = np.mean(np.linalg.norm(acce_tmp,axis=1))
    return acce.mean()