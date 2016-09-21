def gradient_descent(objective_f, gradient_f, x0, step_size, threshold):
    old_x = x0
    old_y = objective_f(x0)
    difference = threshold + 1:

    while difference > threshold:
        new_x = old_x - step_size * gradient_f(old_x)
        new_y = objective_f(new_x)
        difference = old_y - new_y
        old_x = new_x
        old_y = new_y

    return (old_x, old_y)

def negative_gaussian(x):
    pass