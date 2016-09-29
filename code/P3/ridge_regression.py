import sys
sys.path.append('../P2')
from linear_regression      import calculate_polynomial_phi, plot_regression, get_polynomial_regression_fn, make_sse_objective_fn, make_mse_objective_fn
from linear_regression      import calculate_mle_weight         as calculate_mle_weight_no_lambda
from loadFittingDataP2      import getData
from regressData            import regressAData, regressBData, validateData
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

def train_model(x_training, y_training):
    w_mles = {}
    M_values = range(0, 11)
    lambda_values = [ i * 0.1 for i in range(20) ]

    for M in M_values:
        for lambda_val in lambda_values:
            w_mle = calculate_mle_weight(x_training, y_training, calculate_polynomial_phi, M, lambda_val)
            w_mles[(M, lambda_val)] = w_mle

    return w_mles

def validate_models(w_mles, sse_fn):
    min_sse = float("inf")
    M, lambda_val, w_mle = 0, 0, []
    all_values = {}

    for hyperparameters in w_mles.keys():
        sse = sse_fn(w_mles[hyperparameters])
        all_values[hyperparameters] = sse

        if sse < min_sse:
            min_sse = sse
            M, lambda_val = hyperparameters
            w_mle = w_mles[hyperparameters]

    return (M, lambda_val, w_mle, all_values)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        M = int(sys.argv[1])
        lambda_val = float(sys.argv[2])
    else:
        M = 3
        lambda_val = 0.1

    x, y = getData(ifPlotData=False)
    real_fn = lambda x: math.cos(math.pi * x) + 1.5 * math.cos(2 * math.pi * x)
    list_of_basis_functions = [
        lambda x: x ** 0,
        lambda x: x ** 1,
        lambda x: x ** 2,
        lambda x: x ** 3,
        lambda x: x ** 4,
        lambda x: x ** 5,
        lambda x: x ** 6,
        lambda x: x ** 7,
        lambda x: x ** 8,
        lambda x: x ** 9,
        lambda x: x ** 10
    ]


    ## Maximum likelihood calculations

    # w_mle = calculate_mle_weight(x, y, calculate_polynomial_phi, M, lambda_val)
    # regression_fn = get_polynomial_regression_fn(w_mle)
    # print "w_mle: ", w_mle

    # w_mle_no_lambda = calculate_mle_weight_no_lambda(x, y, calculate_polynomial_phi, M)
    # regression_fn_no_lambda = get_polynomial_regression_fn(w_mle_no_lambda)
    # print "w_mle_no_lambda: ", w_mle_no_lambda

    # graph_title = "Linear regression (M=" + str(M) + ", lambda=" + str(lambda_val) + ")"
    # fns_to_plot = {
    #     "Actual": real_fn,
    #     "Linear regression": regression_fn,
    #     "Ridge regression": regression_fn_no_lambda
    # }
    # plot_regression(x, y, fns_to_plot, graph_title)


    ## Train, validate, test
    data_b = regressAData()
    data_a = regressBData()
    validate_data = validateData()

    x_training, y_training = [ data[0] for data in data_a[0] ], [ data[0] for data in data_a[1] ]
    x_testing, y_testing = [ data[0] for data in data_b[0] ], [ data[0] for data in data_b[1] ]
    x_validate, y_validate = [ data[0] for data in validate_data[0] ], [ data[0] for data in validate_data[1] ]

    sse_fn = make_sse_objective_fn(x_validate, y_validate, list_of_basis_functions)
    sse_fn_testing = make_sse_objective_fn(x_testing, y_testing, list_of_basis_functions)
    # sse_fn = make_mse_objective_fn(x_validate, y_validate, list_of_basis_functions)
    # sse_fn_testing = make_mse_objective_fn(x_testing, y_testing, list_of_basis_functions)

    w_mles = train_model(x_training, y_training)
    best_M, best_lambda_val, best_w_mle, all_values = validate_models(w_mles, sse_fn)
    best_regression_fn = get_polynomial_regression_fn(best_w_mle)
    print "best_value: ", all_values[(best_M, best_lambda_val)]
    print "best_w_mle: ", best_w_mle
    print "best testing value: ", sse_fn_testing(best_w_mle)

    fns_to_plot = {
        "Best regression": best_regression_fn
    }
    plot_regression(x_training, y_training, fns_to_plot, "Training, M=" + str(best_M) + ", lambda=" + str(best_lambda_val))
    plot_regression(x_validate, y_validate, fns_to_plot, "Validation, M=" + str(best_M) + ", lambda=" + str(best_lambda_val))
    plot_regression(x_testing, y_testing, fns_to_plot, "Testing, M=" + str(best_M) + ", lambda=" + str(best_lambda_val))


"""
Training a, testing b:
M=2, lambda=0
best_w_mle: [ 0.997 0.888 -0.0374 ]

min_sse: 2.35
testing_sse: 25.753
min_mse: 0.1067
testing_mse: 2.575



Training b, testing a:
M=3, lambda = 0.6
best_w_mle: [ 0.728 1.332 -0.184 ]

min_sse: 28.095
testing_sse: 40.087
min_mse: 1.277
testing_sse: 3.084

"""

