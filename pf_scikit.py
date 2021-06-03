import argparse
from timeit import default_timer as timer

import numpy as np
from sklearn.preprocessing import PolynomialFeatures


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--degree", help="Degree of polynomial features", default=2, type=int)
parser.add_argument("-i", "--iterations", help="Number of iterations over one number of inputs",
                    default=10000, type=int)
parser.add_argument("--start", help="Number of inputs start", default=1, type=int)
parser.add_argument("--stop", help="Number of inputs stop", default=1001, type=int)
parser.add_argument("--step", help="Number of inputs step", default=100, type=int)

args = parser.parse_args()


order = args.degree
inputs = range(args.start, args.stop, args.step)
iterations = args.iterations

times = []

sub_iterations = 100

if iterations/100 > 0:
    iterations /= 100
else:
    sub_iterations = 1

for i in inputs:
    times_in = []
    x = np.random.rand(i).reshape(1, -1)
    poly = PolynomialFeatures(order)
    poly.fit(x)

    for idx in range(int(iterations)):
        start = timer()
        for idx2 in range(sub_iterations):
            pfg = poly.transform(list(x))
        end = timer()
        times_in.append((end-start)/sub_iterations)
    times.append([order, i, np.sum(times_in), np.min(times_in), np.mean(times_in), np.max(times_in),
                  len(poly.transform(x)[0])])
    print('Order: {}, Inputs: {}, Overall test time: {}, Test mean time: {}, # of PF: {}'.format(
        times[-1][0], times[-1][1], times[-1][2], times[-1][4], times[-1][6]))

np.savetxt(f'res_pf_scikit_d{order}_it{iterations*sub_iterations}.csv', times,
           header='order,inputs,overall time,min time,mean time,max time,num of PF',
           delimiter=',')


