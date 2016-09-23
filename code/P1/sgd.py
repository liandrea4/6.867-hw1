from loadFittingDataP1      import getData
from gradient_descent       import gradient_descent
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
            sum += partial_sum_squared
        return sum
    return least_square_error

def make_least_square_derivative(x, y):
    def least_square_derivative(theta):
        gradient_sum = numpy.dot(numpy.dot(numpy.matrix.transpose(x[0]), theta) - y[0], x[0])
        # least_square = []
        for i in range(1, len(y)):
            # least_square.append((numpy.dot(numpy.matrix.transpose(x[i]), theta) - y[i]))
            gradient_sum += (numpy.dot(numpy.dot(numpy.matrix.transpose(x[i]), theta) - y[i], x[i]))
        gradient_sum *= 2
        # print "least_square: ", least_square, "   sum: ", sum(least_square)
        return gradient_sum
    return least_square_derivative

####### SGD update ##########
def calc_next_theta(old_theta, x, y, index):
    t0 = 1.
    k = 0.6
    n = lambda t: (1.+t)**(-k)
    return old_theta - numpy.dot(2 * n(index), numpy.dot(numpy.matrix.transpose(x[i]), old_theta) - y[i])

def sgd(x, y, theta, threshold):
    t = 0
    for x_i, y_i in zip(x,y):
        theta = calc_next_theta(theta, x_i, y_i, t)
        t += 1


def plot_data(previous_values, x_channel): #, x_matrix, y_vector):
    x = [ value[0][x_channel] for value in previous_values ]
    y = [ value[1] for value in previous_values ]
    # data_x = [ value[x_channel] for value in x_matrix ]
    plt.figure()
    plt.plot(x, y, 'ro')
    # plt.plot(data_x, y_vector, 'bo')
    plt.show()

if __name__ == '__main__':
    step_size = 0.000001
    threshold = 0.5
    x_channel = 2

    fitting_data = getData()
    x_matrix = fitting_data[0]
    y_vector = fitting_data[1]
    theta = numpy.array([0.0] * 10)

    objective_f = make_least_square_error(x_matrix, y_vector)
    gradient_f = make_least_square_derivative(x_matrix, y_vector)

    previous_values = gradient_descent(objective_f, gradient_f, theta, step_size, threshold)
    min_x, min_y = (previous_values[-1][0], previous_values[-1][1])
    print "min_x: ", min_x, "  min_y",  min_y
    print "number of steps: ", len(previous_values)

    plot_data(previous_values, x_channel) #, x_matrix, y_vector)


