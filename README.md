# TBPFG - Tensor-based Polynomial Features Generator

The algorithm of TBPFG is implemented in modules *tbpf* (numpy implementation) 
and *tbpf_tf* (tensorflow implementation up to pf degree 5).

Scripts in the root are for testing of algorithms performance. 

* tbpf_numpy.py - Numpy implementation of TBPFG
* tbpf_tf_cpu.py - Tensorflow implementation of TBPFG, TF is forced to use CPU
* tbpf_tf_gpu.py - Tensorflow implementation of TBPFG, TF will use GPU if it is available
* pf_numpy.py - Simple recursive algorithm of PF generation
* pf_scikit.py - Algorithm from scikit-learn library *sklearn.preprocessing.PolynomialFeatures*

All testing scripts are CMD-line scripts that write the results into the console and 
saves results into CSV file. The naming convention for results file is *res_{script name}_d{degree}_it{number of iterations}.csv*.
  
Input parameters of testing scripts are:

  * -h, --help            show help message and exit
  * -d DEGREE, --degree DEGREE
                        Degree of polynomial features
  * -i ITERATIONS, --iterations ITERATIONS
                        Number of iterations over one number of inputs
  * --start START         Number of inputs start
  * --stop STOP           Number of inputs stop
  * --step STEP           Number of inputs step
