from loadParametersP1 import getData
import numpy
import math

def gradient_descent(objective_f, gradient_f, x0, step_size, threshold):
    old_x = x0
    old_y = objective_f(x0)
    difference = threshold + 1

    print "difference: ", difference, "   threshold: ", threshold
    print "old_x: ", old_x, "old_y: ", old_y
    while abs(difference) > threshold:
        new_x = old_x - step_size * gradient_f(old_x)
        new_y = objective_f(new_x)
        print "gradient: ", gradient_f(old_x)
        print "new_x: ", new_x, "  new_y: ", new_y
        difference = old_y - new_y
        print "difference: ", difference
        old_x = new_x
        old_y = new_y

    return (old_x, old_y)

def make_negative_gaussian(mean, covariance):
    def negative_gaussian(x):
        n = 2
        exponential_part = math.exp(-1/2. * numpy.dot(numpy.dot(numpy.matrix.transpose(x-mean), numpy.linalg.inv(covariance)), (x-mean)))
        return -1/numpy.sqrt((2*math.pi)**n * numpy.linalg.norm(covariance)) * exponential_part
    return negative_gaussian

def make_negative_gaussian_derivative(negative_gaussian, mean, covariance):
    def negative_gaussian_derivative(x):
        return numpy.dot(numpy.dot(-negative_gaussian(x), numpy.linalg.inv(covariance)), x-mean)
    return negative_gaussian_derivative

def make_quadratic_bowl(A, b):
    def quadratic_bowl(x):
        y = (1/2.)* numpy.dot(numpy.dot(numpy.matrix.transpose(x), A), x) - numpy.dot(numpy.matrix.transpose(x), b)
        return y
    return quadratic_bowl

def make_quadratic_bowl_derivative(A, b):
    def quadratic_bowl_derivative(x):
        return numpy.dot(A, x) - b
    return quadratic_bowl_derivative

def calculate_gradient_numerically(f, x, y, delta):
    original = numpy.array([x,y])
    x_new = numpy.array([(x+delta), y])
    y_new = numpy.array([x, (y+delta)])
    x_slope = (f(x_new) - f(original))/delta
    y_slope = (f(y_new) - f(original))/delta
    return (x_slope, y_slope)


if __name__ == '__main__':
    parameters = getData()
    initial_guess = numpy.array([6, 8])
    step_size = 0.1
    threshold = 0.01

    # gaussian_mean = parameters[0]
    # gaussian_cov = parameters[1]
    # negative_gaussian = make_negative_gaussian(gaussian_mean, gaussian_cov)
    # negative_gaussian_derivative = make_negative_gaussian_derivative(negative_gaussian, gaussian_mean, gaussian_cov)

    quadratic_bowl = make_quadratic_bowl(parameters[2], parameters[3])
    quadratic_bowl_derivative = make_quadratic_bowl_derivative(parameters[2], parameters[3])
    print quadratic_bowl_derivative(initial_guess)
    print calculate_gradient_numerically(quadratic_bowl, 6, 8, 0.2) 


    # min_x, min_y = gradient_descent(negative_gaussian, negative_gaussian_derivative, initial_guess, step_size, threshold)
    # print "min_x: ", min_x, "  min_y",  min_y

