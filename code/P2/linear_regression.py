from loadFittingDataP2      import getData
import matplotlib.pyplot    as plt
import numpy
import math

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


if __name__ == '__main__':
    M = 3
    x, y = getData(ifPlotData=False)
    real_fn = lambda x: math.cos(math.pi * x) + 1.5 * math.cos(2 * math.pi * x)

    w_mle = calculate_mle_weight(x, y, M)
    regression_fn = get_regression_fn(w_mle)

    plot_regression(x, y, real_fn, regression_fn)
