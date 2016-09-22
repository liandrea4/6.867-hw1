from loadParametersP1 import getData
import matplotlib.pyplot as plt
import numpy
import math

def gradient_descent(objective_f, gradient_f, x0, step_size, threshold):
    previous_values = [(x0, objective_f(x0))]
    difference = threshold + 1

    while abs(difference) > threshold:
        old_x = previous_values[-1][0]
        old_y = previous_values[-1][1]
        new_x = old_x - (step_size * gradient_f(old_x))
        new_y = objective_f(new_x)
        difference = old_y - new_y
        previous_values.append((new_x, new_y))

    return previous_values

def plot_gradient_descent(objective_f, previous_values):
    fig, ax = plt.subplots()

    gradient_descent_x = [ value[0][0] for value in previous_values ]
    gradient_descent_y = [ value[1] for value in previous_values ]
    labels = range(1, len(previous_values) + 1)
    plt.plot(gradient_descent_x, gradient_descent_y, 'ro')
    # for i, label in enumerate(labels):
    #     # if i > 5:
    #     #     break
    #     ax.annotate(label, (gradient_descent_x[i], gradient_descent_y[i]))

    objective_x = numpy.arange(-50, 50, 0.1)
    # objective_x = numpy.arange(min(gradient_descent_x), max(gradient_descent_x), 0.1)
    objective_y = [ objective_f(numpy.array([x_i, x_i])) for x_i in objective_x ]
    plt.plot(objective_x, objective_y, 'b-')

    plt.show()


####### Gradient descent testing functions ########

def make_negative_gaussian(mean, covariance):
    def negative_gaussian(x):
        n = 2
        exponential_part = numpy.exp(-1/2. * numpy.dot(numpy.dot(numpy.matrix.transpose(x-mean), numpy.linalg.inv(covariance)), (x-mean)))
        return -1./numpy.sqrt((2*numpy.pi)**n * numpy.linalg.det(covariance)) * exponential_part
    return negative_gaussian

def make_negative_gaussian_derivative(negative_gaussian, mean, covariance):
    def negative_gaussian_derivative(x):
        return numpy.dot(numpy.dot(-negative_gaussian(x), numpy.linalg.inv(covariance)), (x-mean))
    return negative_gaussian_derivative

def make_quadratic_bowl(A, b):
    def quadratic_bowl(x):
        y = (1/2.) * numpy.dot(numpy.dot(numpy.matrix.transpose(x), A), x) - numpy.dot(numpy.matrix.transpose(x), b)
        return y
    return quadratic_bowl

def make_quadratic_bowl_derivative(A, b):
    def quadratic_bowl_derivative(x):
        return numpy.dot(A, x) - b
    return quadratic_bowl_derivative


######## Numerical approx for gradient ########

def calculate_gradient_numerically(f, x, y, delta):
    original = numpy.array([x,y])
    x_new = numpy.array([(x+delta), y])
    y_new = numpy.array([x, (y+delta)])
    x_slope = (f(x_new) - f(original))/delta
    y_slope = (f(y_new) - f(original))/delta
    return (x_slope, y_slope)


####### Least squares error ########

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


if __name__ == '__main__':
    parameters = getData()
    initial_guess = numpy.array([40, 40])
    step_size = 100000
    threshold = 0.0000000005

    gaussian_mean = parameters[0]
    gaussian_cov = parameters[1]
    objective_f = make_negative_gaussian(gaussian_mean, gaussian_cov)
    gradient_f = make_negative_gaussian_derivative(objective_f, gaussian_mean, gaussian_cov)

    # objective_f = make_quadratic_bowl(parameters[2], parameters[3])
    # gradient_f = make_quadratic_bowl_derivative(parameters[2], parameters[3])

    previous_values = gradient_descent(objective_f, gradient_f, initial_guess, step_size, threshold)
    min_x, min_y = (previous_values[-1][0], previous_values[-1][1])
    print "min_x: ", min_x, "  min_y",  min_y
    print "number of steps: ", len(previous_values)

    plot_gradient_descent(objective_f, previous_values)

