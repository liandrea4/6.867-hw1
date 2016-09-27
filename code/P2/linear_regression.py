import sys
sys.path.append('../P1')
from gradient_descent       import gradient_descent
from sgd                    import plot_data
from loadFittingDataP2      import getData
import matplotlib.pyplot    as plt
import numpy
import math

##### Maximum likelihood #####

def calculate_phi(x_vector, y_vector, M):
    phi = []
    for x in x_vector:
        phi_x = []
        for i in range(M+1):
            phi_x.append(x ** i)
        phi.append(phi_x)
    return phi

def calculate_mle_weight(x_vector, y_vector, M):
    phi = calculate_phi(x_vector, y_vector, M)
    phi_transpose = numpy.transpose(phi)
    inversed = numpy.linalg.inv(numpy.dot(phi_transpose, phi))
    w_mle = numpy.dot(numpy.dot(inversed, phi_transpose), y_vector)
    return w_mle

def get_regression_fn(w_mle):
    def regression_fn(x):
        fn = 0
        for index in range(len(w_mle)):
            fn += w_mle[index] * (x ** index)
        return fn
    return regression_fn

def plot_regression(x, y, real_fn, regression_fn):
    plt.figure()

    ## Plot data points
    plt.plot(x, y, 'o')

    x_fn_values = numpy.linspace(min(x), max(x), 1000)

    ## Plot real function
    real_fn_y_values = [ real_fn(x_i) for x_i in x_fn_values ]
    plt.plot(x_fn_values, real_fn_y_values, 'g-', linewidth=2)

    ## Plot regression function
    regression_fn_y_values = [ regression_fn(x_i) for x_i in x_fn_values ]
    plt.plot(x_fn_values, regression_fn_y_values, 'r-', linewidth=2)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear regression (M=' + str(M) + ')')
    plt.show()

##### Sum of square error #####
def make_sse_objective_fn(x, y, list_of_basis_functions):
    def sse_objective_fn(weight_vector):
        regression_fn = get_generic_regression_fn(list_of_basis_functions, weight_vector)

        sse = 0
        for x_i, y_i in zip(x, y):
            difference = y_i - regression_fn(x_i)
            sse += (difference ** 2)
        return sse

    return sse_objective_fn

def make_sse_gradient_fn(x, y, list_of_basis_functions, M):
    def sse_gradient_fn(weight_vector):
        sse_gradient = []

        for i in range(M + 1):
            regression_fn = get_generic_regression_fn(list_of_basis_functions[:i+1], weight_vector[:i+1])
            gradient = 0
            for x_i, y_i in zip(x, y):
                gradient += (2 * (y_i - regression_fn(x_i)))
            sse_gradient.append(gradient)

        return numpy.array(sse_gradient)
    return sse_gradient_fn

def get_generic_regression_fn(list_of_basis_functions, w_mle):
    if len(list_of_basis_functions) != len(w_mle):
        print "list: ", list_of_basis_functions, "   weight_vector: ", w_mle
        print "len(list): ", len(list_of_basis_functions), "   len(weight_vector): ", len(w_mle)
        raise Exception("Wrong dimensions of basis functions and weight vector")

    def generic_regression_fn(x):
        fn = 0
        for index in range(len(w_mle)):
            fn += w_mle[index] * list_of_basis_functions[index](x)
        return fn
    return generic_regression_fn

if __name__ == '__main__':
    M = 3
    x, y = getData(ifPlotData=False)
    real_fn = lambda x: math.cos(math.pi * x) + 1.5 * math.cos(2 * math.pi * x)

    ## Maximum likelihood calculations

    w_mle = calculate_mle_weight(x, y, M)
    print "w_mle: ", w_mle
    regression_fn = get_regression_fn(w_mle)

    # plot_regression(x, y, real_fn, regression_fn)


    ## Gradient descent
    list_of_basis_functions = [ lambda x: x ** i for i in range(M+1) ]
    weight_vector = numpy.array([1] * (M+1))
    step_size = 0.0005
    threshold = 1

    objective_f = make_sse_objective_fn(x, y, list_of_basis_functions)
    gradient_f = make_sse_gradient_fn(x, y, list_of_basis_functions, M)

    previous_values = gradient_descent(objective_f, gradient_f, weight_vector, step_size, threshold)
    min_x, min_y = (previous_values[-1][0], previous_values[-1][1])
    print "min_x: ", min_x, "  min_y",  min_y
    print "number of steps: ", len(previous_values)

    plot_data(previous_values, 0)
