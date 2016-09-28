import sys
sys.path.append('../P2')
from linear_regression      import calculate_polynomial_phi, plot_regression, get_polynomial_regression_fn
from linear_regression      import calculate_mle_weight         as calculate_mle_weight_no_lambda
from loadFittingDataP2      import getData
import matplotlib.pyplot    as plt
import numpy
import math

def calculate_mle_weight(x_vector, y_vector, calculate_phi_fn, M, lambda_val):
    phi = calculate_phi_fn(x_vector, M)
    phi_transpose = numpy.transpose(phi)
    pre_inverse = numpy.dot(lambda_val, numpy.identity(M+1)) + numpy.dot(phi_transpose, phi)
    inversed = numpy.linalg.inv(pre_inverse)
    w_mle = numpy.dot(numpy.dot(inversed, phi_transpose), y_vector)
    return w_mle

if __name__ == '__main__':
    if len(sys.argv) > 1:
        M = int(sys.argv[1])
        lambda_val = float(sys.argv[2])
    else:
        M = 3
        lambda_val = 0.1

    x, y = getData(ifPlotData=False)
    real_fn = lambda x: math.cos(math.pi * x) + 1.5 * math.cos(2 * math.pi * x)

    ## Maximum likelihood calculations

    w_mle = calculate_mle_weight(x, y, calculate_polynomial_phi, M, lambda_val)
    regression_fn = get_polynomial_regression_fn(w_mle)
    print "w_mle: ", w_mle

    w_mle_no_lambda = calculate_mle_weight_no_lambda(x, y, calculate_polynomial_phi, M)
    regression_fn_no_lambda = get_polynomial_regression_fn(w_mle_no_lambda)
    print "w_mle_no_lambda: ", w_mle_no_lambda

    graph_title = "Linear regression (M=" + str(M) + ", lambda=" + str(lambda_val) + ")"
    fns_to_plot = {
        "Actual": real_fn,
        "Linear regression": regression_fn,
        "Ridge regression": regression_fn_no_lambda
    }
    plot_regression(x, y, fns_to_plot, graph_title)




