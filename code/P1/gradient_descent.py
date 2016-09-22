from loadParametersP1 import getData
import numpy
import math

def gradient_descent(objective_f, gradient_f, x0, step_size, threshold):
    old_x = x0
    old_y = objective_f(x0)
    difference = threshold + 1

    print "difference: ", difference, "   threshold: ", threshold
    while difference > threshold:
        new_x = old_x - step_size * gradient_f(old_x)
        new_y = objective_f(new_x)
        difference = old_y - new_y
        old_x = new_x
        old_y = new_y

    return (old_x, old_y)

def make_negative_gaussian(mean, covariance):
    def negative_gaussian(x):
        return -1/math.sqrt((2*math.pi)**n * abs(covariance)) * math.exp(-1/2. * numpy.matrix.transpose(x-mean) / covariance * (x-mean))
    return negative_gaussian

def make_negative_gaussian_derivative(negative_gaussian, mean, covariance):
    def negative_gaussian_derivative(x):
        return -negative_gaussian(x) / covariance * (x-mean)
    return negative_gaussian_derivative

def make_quadratic_bowl(A, b):
    def quadratic_bowl(x):
        y = (1/2)*numpy.matrix.transpose(x)*A*x - numpy.matrix.transpose(x)*b
        print "y: ", y
        print "first part: ", numpy.matrix.transpose(x), A, x
        print "second part: ", numpy.matrix.transpose(x)*b
        return y
    return quadratic_bowl

def make_quadratic_bowl_derivative(A, b):
    def quadratic_bowl_derivative(x):
        return A*x - b
    return quadratic_bowl_derivative

if __name__ == '__main__':
    parameters = getData()
    print parameters
    # print "mean: ", parameters[0][0]
    # print "cov: ", abs(parameters[1])

    # gaussian_mean = parameters[0]
    # gaussian_cov = parameters[1]
    # negative_gaussian = make_negative_gaussian(gaussian_mean, parameters[1])
    # negative_gaussian_derivative = make_negative_gaussian_derivative(par)

    quadratic_bowl = make_quadratic_bowl(parameters[2], parameters[3])
    quadratic_bowl_derivative = make_quadratic_bowl_derivative(parameters[2], parameters[3])


    min_x, min_y = gradient_descent(quadratic_bowl, quadratic_bowl_derivative, numpy.array([0, 0]), 0.1, 0.01)
    print min_x, min_y

