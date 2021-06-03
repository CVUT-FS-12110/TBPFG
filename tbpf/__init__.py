import numpy as np

def mask_matrix(var_n, order, bias=False):
    if order < 2:
        raise(Exception('Order has to be greater then 1.'))
    n = var_n + 1
    fout = np.arange(1,n+1)
    for i in range(2,order):
        fout = np.cumsum(fout)
    fout = fout.astype(int)
    fm = np.zeros((n, fout[-1]), dtype=bool)
    if bias:
        start = 0
    else:
        start = 1
    for i in range(start, n):
        fm[i, 0:int(fout[i])] = 1
    return fm

def tbpf(v_in, order, mask):
    v_in = np.hstack(([1],v_in))
    v_out = v_in
    for o in range(0, (order-2)):
        xy = v_out[:, np.newaxis].dot(v_out[np.newaxis, :])
        v_out = xy[mask[o] != 0]
    xy = v_in[:, np.newaxis].dot(v_out[np.newaxis, :])
    v_out = xy[mask[order-2] != 0]

    return v_out