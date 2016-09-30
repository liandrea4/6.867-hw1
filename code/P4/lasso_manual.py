import sys
sys.path.append('../P2')
from linear_regression          import get_generic_regression_fn, plot_regression
from sklearn                    import linear_model
import numpy                    as np
import matplotlib.pyplot        as plt
import matplotlib.pylab         as pylab
import scipy.optimize           as spo
import math
import lassoData

#### Ridge Regression Loss ####
def ridge_loss(x,y,lambda1,M):
    def evalRL(theta):
        X = sin_features(x,M)
        theta = np.array(theta).reshape(-1,1)
        y_hat = np.dot(X,theta)
        diff = y - y_hat
        loss = 1./(len(y))*np.sum(np.dot(np.transpose(diff),diff)) + lambda1 * np.sum(np.dot(np.transpose(theta),theta))
        return loss
    return evalRL

#### Lasso Loss Function ####
def lasso_loss(x,y,lambda1,M):
    def evalLL(theta):
        X = sin_features(x,M)
        theta = np.array(theta).reshape(-1,1)
        y_hat = np.dot(X,theta)
        diff = y - y_hat
        loss = 1./(len(y))*np.sum(np.dot(np.transpose(diff),diff)) + lambda1 * np.sum(np.absolute(theta))
        return loss
    return evalLL

#### Predict y_hat ####
def predict(X,theta):
    theta = np.array(theta).reshape(-1,1)
    y_hat = np.dot(X,theta)
    return y_hat

#### Mean Squared Error ####
def MSE(y_hat,y):
    mse = np.dot(np.transpose(y_hat-y),(y_hat-y))/float(len(y_hat))
    return mse
#### Using Sin Basis ####
def create_features(X = np.zeros(1),M=1):
    if len(X) == 1:
        n = len(X.T)
    else:
        n = len(X)
    X_matrix = np.zeros((n,M))
    for i in range(n):
        for j in range(M):
            if len(X) == 1:
                if j == 0 :
                    X_matrix[i,j] = X.T[i]
                else:
                    X_matrix[i,j] = math.sin(math.pi*0.4*(X.T[i])*i)
            else:
                if j == 0:
                    X_matrix[i,j] = X[i]
                else:
                    X_matrix[i,j] = math.sin(math.pi*0.4*(X[i])*i)
    return X_matrix

def sin_features(X,M):
    X_matrix = np.zeros((len(X),M))
    for i in range(len(X)):
        for j in range(M):
            if j == 0:
                X_matrix[i,j] = X[i]
            else:
                X_matrix[i,j] = math.sin(math.pi * 0.4 * X[i] * j)
    return X_matrix
##### Lambda Optimization #####

def optimize(lambdas,M,theta_guess, x_train,x_test,x_valid,y_train,y_test,y_valid):
    opt_lasso_lambda,opt_lasso_mse = 0,100
    opt_lasso_theta = np.copy(theta_guess)
    opt_ridge_lambda,opt_ridge_mse = 0,100
    opt_ridge_theta = np.copy(theta_guess)
    for l in lambdas:
        ## We could use our gradient descent method here but SciPy's BFGS is just better and will give let us test better ##

        lasso_f = lasso_loss(x_train,y_train,l,M)
        ridge_f = ridge_loss(x_train,y_train,l,M)

        lasso_theta = np.array(spo.fmin_bfgs(lasso_f,theta_guess,gtol=10**-6,disp=0)).reshape(-1,1)
        ridge_theta = np.array(spo.fmin_bfgs(ridge_f,theta_guess,gtol=10**-6,disp=0)).reshape(-1,1)

        X = create_features(x_valid,M)
        mse_lasso = MSE(predict(X,lasso_theta),y_valid)
        mse_ridge = MSE(predict(X,ridge_theta),y_valid)

        if mse_lasso < opt_lasso_mse:
            opt_lasso_lambda = l
            opt_lasso_mse = mse_lasso
            opt_lasso_theta = lasso_theta
        if mse_ridge < opt_ridge_mse:
            opt_ridge_lambda = l
            opt_ridge_mse = mse_ridge
            opt_ridge_theta = ridge_theta
    # return [lasso,ridge]
    # lasso = optimal (lambda,mse,theta)
    # ridge = optimatl (lambda,mse,theta)
    return [(opt_lasso_lambda,opt_lasso_mse,opt_lasso_theta),(opt_ridge_lambda,opt_ridge_mse,opt_ridge_theta)]

if __name__ == '__main__':
    x_valid,y_valid = lassoData.lassoValData()
    x_train,y_train = lassoData.lassoTrainData()
    x_test,y_test = lassoData.lassoTestData()
    w_true = lassoData.lassoW()

    M = len(w_true)

    sin_basis = [
        lambda x: x,
        lambda x: math.sin(0.4 * math.pi * x * 1),
        lambda x: math.sin(0.4 * math.pi * x * 2),
        lambda x: math.sin(0.4 * math.pi * x * 3),
        lambda x: math.sin(0.4 * math.pi * x * 4),
        lambda x: math.sin(0.4 * math.pi * x * 5),
        lambda x: math.sin(0.4 * math.pi * x * 6),
        lambda x: math.sin(0.4 * math.pi * x * 7),
        lambda x: math.sin(0.4 * math.pi * x * 8),
        lambda x: math.sin(0.4 * math.pi * x * 9),
        lambda x: math.sin(0.4 * math.pi * x * 10),
        lambda x: math.sin(0.4 * math.pi * x * 11),
        lambda x: math.sin(0.4 * math.pi * x * 12)
    ]

    lambdas = [0, .1, .2, .3, .4, .5, .6, .7 ,.8, .9, 1]
    theta_guess = np.repeat(0,M).reshape(-1,1)
    results = optimize(lambdas,M,theta_guess, x_train,x_test,x_valid,y_train,y_test,y_valid)

    print results[0][2],results[1][2]
    lasso_theta = results[0][2]
    ridge_theta = results[1][2]

    print "lasso_theta: ", lasso_theta, "  ridge_theta: ", ridge_theta
    lasso_w = [ w[0] for w in lasso_theta ]
    ridge_w = [ w[0] for w in ridge_theta ]

    lasso_regression_fn = get_generic_regression_fn(sin_basis, lasso_w)
    ridge_regression_fn = get_generic_regression_fn(sin_basis, ridge_w)
    actual_fn = get_generic_regression_fn(sin_basis, w_true)

    fns_to_plot = {
        "Actual": actual_fn,
        "LASSO": lasso_regression_fn,
        "Ridge": ridge_regression_fn
    }
    best_lambda = 0.1
    plot_regression(x_train, y_train, fns_to_plot, "Training, lambda=" + str(best_lambda))
    plot_regression(x_valid, y_valid, fns_to_plot, "Validation, lambda=" + str(best_lambda))
    plot_regression(x_test, y_test, fns_to_plot, "Testing, lambda=" + str(best_lambda))



    # #ridge_f = ridge_loss(x_train,y_train,0.2,5)
    # #ridge_theta = np.array(spo.fmin_bfgs(ridge_f,theta_guess,gtol=10**-6,disp=0)).reshape(-1,1)
    # X = create_features(x_train,M)

    # lasso_predictions = predict(X,lasso_theta)
    # ridge_predictions = predict(X,ridge_theta)

    # X_dom = np.linspace(-1,1,50)
    # #X_dom = np.array([X_dom, np.sin(0.4*np.pi*X_dom*1), np.sin(0.4*np.pi*X_dom*2), np.sin(0.4*np.pi*X_dom*3), np.sin(0.4*np.pi*X_dom*4), np.sin(0.4*np.pi*X_dom*5), np.sin(0.4*np.pi*X_dom*6), np.sin(0.4*np.pi*X_dom*7), np.sin(0.4*np.pi*X_dom*8), np.sin(0.4*np.pi*X_dom*9), np.sin(0.4*np.pi*X_dom*10), np.sin(0.4*np.pi*X_dom*11),np.sin(0.4*np.pi*X_dom*12)])
    # X_matrix = sin_features(X_dom,M)

    # actual_values = predict(X_matrix,np.transpose(w_true))

    # lasso_curve = predict(X_matrix,lasso_theta)
    # ridge_curve = predict(X_matrix,ridge_theta)


    # For the optimal thetas ridge and lasso are the same! #
    # f1 = plt.figure()
    # f1.suptitle('Lasso vs Actual')
    # ax1 = f1.add_subplot(111)
    # ax1.plot(X_dom,lasso_curve,'b',X_dom,actual_values,'r')

    # f2 = plt.figure()
    # f2.suptitle('Ridge vs Actual')
    # ax2 = f2.add_subplot(111)
    # ax2.plot(X_dom,ridge_curve,'g',X_dom,actual_values,'r')

    # f3 = plt.figure()
    # f3.suptitle('Lasso vs Ridge')
    # ax3 = f3.add_subplot(111)
    # ax3.plot(X_dom,lasso_curve,'b',X_dom,ridge_curve,'g', X_dom, actual_values, 'r')

    # plt.plot(X_dom,lasso_curve,'b'),'r')
    # plt.plot(X_dom,ridge_curve,'g')


    # plt.plot(X_dom,actual_values)

    # clf = linear_model.Lasso(alpha=1)
    # X_matrix = sin_features(x_train, M)
    # clf.fit(X_matrix, y_train)
    # print clf.coef_

    # plt.show()











