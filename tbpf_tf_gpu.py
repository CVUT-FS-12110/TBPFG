import argparse
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf

import tbpf_tf

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--degree", help="Degree of polynomial features", default=2, type=int)
parser.add_argument("-i", "--iterations", help="Number of iterations over one number of inputs",
                    default=10000, type=int)
parser.add_argument("--start", help="Number of inputs start", default=1, type=int)
parser.add_argument("--stop", help="Number of inputs stop", default=1001, type=int)
parser.add_argument("--step", help="Number of inputs step", default=100, type=int)

args = parser.parse_args()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

order = args.degree
inputs = range(args.start, args.stop, args.step)
iterations = args.iterations

times = []
pfgs = [tbpf_tf.PFG2, tbpf_tf.PFG3, tbpf_tf.PFG4, tbpf_tf.PFG5]

sub_iterations = 100

if iterations/100 > 0:
    iterations /= 100
else:
    sub_iterations = 1

for i in inputs:
    times_in = []
    input_tensor = tf.random.normal([i], 0, 1, tf.float32)
    x = tf.reshape(tf.concat([[1], input_tensor], 0), [1, -1])

    M = []

    for it in range(2, order + 1):
        M.append(tf.convert_to_tensor(tbpf_tf.mask_matrix(i, it, True)))

    for idx in range(int(iterations)):
        start = timer()
        for idx2 in range(sub_iterations):
            pfg = pfgs[order-2](M, x)
        end = timer()
        times_in.append((end-start)/sub_iterations)
    times.append([order, i, np.sum(times_in), np.min(times_in), np.mean(times_in), np.max(times_in),
                  pfgs[order-2](M, x).numpy().size])
    print('Order: {}, Inputs: {}, Overall test time: {}, Test mean time: {}, # of PF: {}'.format(
        times[-1][0], times[-1][1], times[-1][2], times[-1][4], times[-1][6]))

np.savetxt(f'res_tbpf_tf_gpu_d{order}_it{iterations*sub_iterations}.csv', times,
           header='order,inputs,overall time,min time,mean time,max time,num of PF',
           delimiter=',')


