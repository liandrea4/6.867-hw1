from loadParametersP1 import getData
import numpy
import math

def gradient_descent(objective_f, gradient_f, x0, step_size, threshold):
    old_x = x0
    old_y = objective_f(x0)
    difference = threshold + 1

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

if __name__ == '__main__':
    parameters = getData()
    print "mean: ", parameters[0][0]
    print "cov: ", abs(parameters[1])

    gaussian_mean = parameters[0]
    gaussian_cov = parameters[1]
    negative_gaussian = make_negative_gaussian(gaussian_mean, parameters[1])
    negative_gaussian_derivative = make_negative_gaussian_derivative(par)
    min_x, min_y = gradient_descent(negative_gaussian)

