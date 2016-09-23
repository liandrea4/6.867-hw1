from loadFittingDataP1      import getData
from gradient_descent       import gradient_descent
from gradient_descent       import make_numeric_gradient_calculator
import matplotlib.pyplot    as plt
import numpy

####### Least squares error ########

def make_least_square_error(x_matrix, y_vector):
    def least_square_error(theta):
        length = len(y_vector)
        sum = 0
        for i in range(length):
            x_transposed = numpy.matrix.transpose(x_matrix[i])
            partial_sum = numpy.dot(x_transposed, theta) - y_vector[i]
            partial_sum_squared = partial_sum ** 2
            sum = sum +partial_sum_squared

        return sum
    return least_square_error

def make_least_square_derivative(x, y):
    def least_square_derivative(theta):
        scaling_factor = 0
        for i in range(len(y)):
            scaling_factor += numpy.dot(numpy.matrix.transpose(x[i]), theta) - y[i]
        scaling_factor *= 2
        return numpy.dot(scaling_factor, numpy.matrix.transpose(x[i]))
    return least_square_derivative

####### SGD update ##########
def calc_next_theta(old_theta, x, y):
    n = 2
    return old_theta - numpy.dot(2 * n, numpy.dot(numpy.matrix.transpose(x[i]), old_theta) - y[i])

def sgd(x, y, next_theta_f, threshold):
    pass


if __name__ == '__main__':
    step_size = 0.001
    threshold = 0.5

    fitting_data = getData()
    x_matrix = fitting_data[0]
    y_vector = fitting_data[1]
    theta = numpy.array([0.0] * 10)

    objective_f = make_least_square_error(x_matrix, y_vector)
    gradient_f = make_numeric_gradient_calculator(objective_f, 0.01)

    previous_values = gradient_descent(objective_f, gradient_f, theta, step_size, threshold)
    min_x, min_y = (previous_values[-1][0], previous_values[-1][1])
    
    print "min_x: ", min_x, "  min_y",  min_y
    print "number of steps: ", len(previous_values)
