import numpy as np
import tensorflow as tf

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


@tf.function
def PFG2(mask, v_in):
    qnu = tf.linalg.matmul(v_in, v_in, transpose_a=True, name="qnu_mul")
    v_out = tf.reshape(tf.boolean_mask(qnu, mask[0]), [1,-1], name="qnu_mask")
    return v_out

@tf.function
def PFG3(mask, v_in):
    v_out = tf.linalg.matmul(v_in, v_in, transpose_a=True, name="qnu_mul")
    v_out = tf.reshape(tf.boolean_mask(v_out, mask[0]), [1, -1], name="qnu_mask")
    v_out = tf.linalg.matmul(v_in, v_out, transpose_a=True, name="cnu_mul")
    v_out = tf.reshape(tf.boolean_mask(v_out, mask[1]), [1, -1], name="cnu_mask")
    return v_out

@tf.function
def PFG4(mask, v_in):
    v_out = tf.linalg.matmul(v_in, v_in, transpose_a=True, name="qnu_mul")
    v_out = tf.reshape(tf.boolean_mask(v_out, mask[0]), [1, -1], name="qnu_mask")
    v_out = tf.linalg.matmul(v_in, v_out, transpose_a=True, name="cnu_mul")
    v_out = tf.reshape(tf.boolean_mask(v_out, mask[1]), [1, -1], name="cnu_mask")
    v_out = tf.linalg.matmul(v_in, v_out, transpose_a=True, name="4nu_mul")
    v_out = tf.reshape(tf.boolean_mask(v_out, mask[2]), [1, -1], name="4nu_mask")
    return v_out

@tf.function
def PFG5(mask, v_in):
    v_out = tf.linalg.matmul(v_in, v_in, transpose_a=True, name="qnu_mul")
    v_out = tf.reshape(tf.boolean_mask(v_out, mask[0]), [1, -1], name="qnu_mask")
    v_out = tf.linalg.matmul(v_in, v_out, transpose_a=True, name="cnu_mul")
    v_out = tf.reshape(tf.boolean_mask(v_out, mask[1]), [1, -1], name="cnu_mask")
    v_out = tf.linalg.matmul(v_in, v_out, transpose_a=True, name="4nu_mul")
    v_out = tf.reshape(tf.boolean_mask(v_out, mask[2]), [1, -1], name="4nu_mask")
    v_out = tf.linalg.matmul(v_in, v_out, transpose_a=True, name="5nu_mul")
    v_out = tf.reshape(tf.boolean_mask(v_out, mask[3]), [1, -1], name="5nu_mask")
    return v_out
