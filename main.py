import numpy as np # type: ignore
from numpy import linalg as la # type: ignore
from scipy.linalg import norm # type: ignore
import matplotlib.pyplot as plt # type: ignore
from numba import njit, jit # type: ignore #b# type: ignore, jitclass  # A just in time compiler to speed things up!

import warnings
warnings.filterwarnings("ignore")

from numpy.random import multivariate_normal, randn # type: ignore
from scipy.linalg.special_matrices import toeplitz # type: ignore

from LinReg import LinReg 

def simu_linreg(w, n, std=1., corr=0.5):
    d = w.shape[0]
    cov = toeplitz(corr ** np.arange(0, d))
    X = multivariate_normal(np.zeros(d), cov, size=n)
    noise = std * randn(n)
    y = X.dot(w) + noise
    return X, y

def simu_logreg(w, n, std=1., corr=0.5):
    X, y = simu_linreg(w, n, std=1., corr=0.5)
    return X, np.sign(y)

d = 50
n = 1000
idx = np.arange(d)

w_model_truth = (-1)**idx * np.exp(-idx / 10.)
X, y = simu_logreg(w_model_truth, n, std=1., corr=0.7)


lbda = 1. / n ** (0.5)
model = LinReg(X, y, lbda)

grad_error = []
for i in range(n):
    ind = np.random.choice(n,1)
    w =  np.random.randn(d)
    vec =  np.random.randn(d)
    eps = pow(10.0, -7.0)
    model.f_i(ind[0],w)
    grad_error.append((model.f_i( ind[0], w+eps*vec) - model.f_i( ind[0], w))/eps - np.dot(model.grad_i(ind[0],w),vec))
plt.stem(grad_error);
print(f" mean grad_error {np.mean(grad_error)}")

from scipy.optimize import check_grad
modellin = LinReg(X, y, lbda)
check_grad(modellin.f, modellin.grad, np.random.randn(d))


from scipy.optimize import fmin_l_bfgs_b
w_init = np.zeros(d)
w_min, obj_min, _ = fmin_l_bfgs_b(model.f, w_init, model.grad, args=(), pgtol=1e-30, factr =1e-30)

print(f"obj_min {obj_min}")
print(norm(model.grad(w_min)))

def sgd(w0, model, indices, steps, w_min, n_iter=100, averaging_on=False, momentum =0 ,verbose=True, start_averaging = 0):
    """Stochastic gradient descent algorithm
    """
    w = w0.copy()
    w_new = w0.copy()
    n_samples, n_features = X.shape
    avg_list = []
    # average x
    w_average = w0.copy()
    # estimation error history
    errors = []
    err = 1.0
    # objective history
    objectives = []
    # Current estimation error
    if np.any(w_min):
        err = norm(w - w_min) / norm(w_min)
        errors.append(err)
    # Current objective
    obj = model.f(w)
    objectives.append(obj)
    if verbose:
        print("Lauching SGD solver...")
        print(' | '.join([name.center(8) for name in ["it", "obj", "err"]]))
    vel = np.zeros_like(w0)
    for t in range(n_iter):
        idx = indices[t]
        vel = momentum * vel - steps[t] * model.grad_i(idx, w)
        w[:] = w + vel

        # Compute the average iterate
        if averaging_on:
            if t > start_averaging:
              w_average = ((w_average * (t - start_averaging)) + w )/ (t-start_averaging+1)
            else:
              w_average = w
            w_test = w_average.copy()
        else:
            w_test = w.copy()

        obj = model.f(w_test)
        if np.any(w_min):
            err = norm(w_test - w_min) / norm(w_min)
            errors.append(err)
        objectives.append(obj)
        k = t
        if k % n_samples == 0 and verbose:
            if(sum(w_min)):
                print(' | '.join([("%d" % k).rjust(8),
                              ("%.2e" % obj).rjust(8),
                              ("%.2e" % err).rjust(8)]))
            else:
                print(' | '.join([("%d" % k).rjust(8),
                              ("%.2e" % obj).rjust(8)]))
    if averaging_on:
        w_output = w_average.copy()
    else:
        w_output = w.copy()
    return w_output, np.array(objectives), np.array(errors)

def gd(w0, model, step, w_min =[], n_iter=100, verbose=True):
    """Gradient descent algorithm
    """
    w = w0.copy()
    w_new = w0.copy()
    n_samples, n_features = X.shape
    # estimation error history
    errors = []
    err = 1.
    # objective history
    objectives = []
    # Current estimation error
    if np.any(w_min):
        err = norm(w - w_min) / norm(w_min)
        errors.append(err)
    # Current objective
    obj = model.f(w)
    objectives.append(obj)
    if verbose:
        print("Lauching GD solver...")
        print(' | '.join([name.center(8) for name in ["it", "obj", "err"]]))
    for k in range(n_iter ):
        ##### TODO ######################
        ##### Compute gradient step update
        grad = model.grad(w)
        w[:] = w - step * grad
        w_new[:] = w
        ##### END TODO ##################
        obj = model.f(w)
        if (sum(w_min)):
            err = norm(w - w_min) / norm(w_min)
            errors.append(err)
        objectives.append(obj)
        if verbose:
            print(' | '.join([("%d" % k).rjust(8),
                              ("%.2e" % obj).rjust(8),
                              ("%.2e" % err).rjust(8)]))
    return w, np.array(objectives), np.array(errors)


datapasses = 30  
n_iter = int(datapasses * n)
Lmax = model.L_max_constant(); 


def main():
    indices_replace = np.random.randint(1,model.n,size = n_iter)
    constant_steps = np.array([1/(2*Lmax) for _ in range(n_iter)])
    w_sgdcr, obj_sgdcr, err_sgdcr = sgd(w_init, model, indices_replace, constant_steps, w_min, n_iter, averaging_on=False, momentum=0, verbose=True, start_averaging=0)

    c = 2 * Lmax
    decreasing_steps = np.array([1/(c + t) for t in range(1,n_iter+1)])
    w_sgdsr, obj_sgdsr, err_sgdsr = sgd(w_init, model, indices_replace, decreasing_steps, w_min, n_iter, averaging_on=False, momentum=0, verbose=True, start_averaging=0)

    # Error of objective on a logarithmic scale
    plt.figure(figsize=(7, 5))
    plt.semilogy(obj_sgdcr - obj_min, label="SGD const", lw=2)
    plt.semilogy(obj_sgdsr - obj_min, label="SGD shrink", lw=2)
    plt.title("Convergence plot", fontsize=16)
    plt.xlabel("#iterations", fontsize=14)
    plt.ylabel("Error of objective", fontsize=14)
    plt.legend()
    plt.show()
    # Distance to the minimizer on a logarithmic scale
    plt.figure(figsize=(7, 5))
    plt.yscale("log")
    plt.semilogy(err_sgdcr , label="SGD const", lw=2)
    plt.semilogy(err_sgdsr , label="SGD shrink", lw=2)
    plt.title("Convergence plot", fontsize=16)
    plt.xlabel("#iterations", fontsize=14)
    plt.ylabel("Distance to the minimum", fontsize=14)
    plt.legend()
    plt.show()


    mu = model.mu_constant();
    Kappa = Lmax/mu;
    tstar = 4*int(np.ceil(Kappa));

    shrinking_steps = np.array([ 1/(2 * Lmax) if t <= tstar else (2 * t + 1)/(((t+1)**2) * mu) for t in range(n_iter)])
    w_sgdsr, obj_sgdss, err_sgdss = sgd(w_init, model, indices_replace, shrinking_steps, w_min, n_iter, averaging_on=False, momentum=0, verbose=True, start_averaging =0)

    plt.figure(figsize=(7, 5))
    plt.semilogy(obj_sgdss - obj_min, label="SGD switch", lw=2)
    plt.semilogy(obj_sgdsr - obj_min, label="SGD shrink", lw=2)
    plt.title("Convergence plot", fontsize=16)
    plt.xlabel("#iterations", fontsize=14)
    plt.ylabel("Error of objective", fontsize=14)
    plt.legend()
    plt.show()
    # Distance to the minimizer on a logarithmic scale
    plt.figure(figsize=(7, 5))
    plt.yscale("log")
    plt.semilogy(err_sgdss , label="SGD switch", lw=2)
    plt.semilogy(err_sgdsr , label="SGD shrink", lw=2)
    plt.title("Convergence plot", fontsize=16)
    plt.xlabel("#iterations", fontsize=14)
    plt.ylabel("Distance to the minimum", fontsize=14)
    plt.legend()
    plt.show()


    start_averaging = n_iter * 0.75
    w_sgdar, obj_sgdar, err_sgdar = sgd(w_init, model, indices_replace, shrinking_steps, w_min, n_iter, averaging_on=True, momentum=0, verbose=True, start_averaging = int(start_averaging))

    plt.figure(figsize=(7, 5))
    plt.semilogy(obj_sgdss - obj_min, label="SGD switch", lw=2)
    plt.semilogy(obj_sgdar - obj_min, label="SGD average end", lw=2)
    plt.semilogy(obj_sgdsr - obj_min, label="SGD shrink", lw=2)
    plt.title("Convergence plot", fontsize=16)
    plt.xlabel("#iterations", fontsize=14)
    plt.ylabel("Loss function", fontsize=14)
    plt.legend()
    plt.show()
    # Distance to the minimizer on a logarithmic scale
    plt.figure(figsize=(7, 5))
    plt.semilogy(err_sgdss , label="SGD switch", lw=2)
    plt.semilogy(err_sgdar , label="SGD average end", lw=2)
    plt.semilogy(obj_sgdsr - obj_min, label="SGD shrink", lw=2)

    plt.title("Convergence plot", fontsize=16)
    plt.xlabel("#iterations", fontsize=14)
    plt.ylabel("Distance to the minimum", fontsize=14)
    plt.legend()
    plt.show()


    start_averaging = n_iter * 0.75
    w_sgdm, obj_sgdm, err_sgdm = sgd(w_init, model, indices_replace, decreasing_steps, w_min, n_iter, averaging_on=True, momentum=0.7, verbose=True, start_averaging = int(start_averaging))

    datapasses = 30 
    step = 1. / model.lipschitz_constant()
    w_gd, obj_gd, err_gd = gd(w_init, model, step, w_min, datapasses)
    complexityofGD = n * np.arange(0, datapasses + 1)
    
    # Error of objective on a logarithmic scale
    plt.figure(figsize=(7, 5))
    plt.semilogy(complexityofGD, obj_gd - obj_min, label="gd", lw=2)
    plt.semilogy(obj_sgdss - obj_min, label="sgd switch", lw=2)
    plt.semilogy(obj_sgdm - obj_min, label="sgdm", lw=2)
    plt.title("Convergence plot", fontsize=16)
    plt.xlabel("# SGD iterations", fontsize=14)
    plt.ylabel("Loss function", fontsize=14)
    plt.legend()
    plt.show()
    # Distance to the minimum on a logarithmic scale
    plt.figure(figsize=(7, 5))
    plt.semilogy(complexityofGD, err_gd, label="gd", lw=2)
    plt.semilogy(err_sgdss , label="sgd switch", lw=2)
    plt.semilogy(err_sgdm , label="sgdm", lw=2)
    plt.title("Convergence plot", fontsize=16)
    plt.xlabel("# SGD iterations", fontsize=14)
    plt.ylabel("Distance to the minimum", fontsize=14)
    plt.legend()
    plt.show()


    datapasses = 30
    indices_no_replace = []
    for t in range(datapasses):
        index = np.random.permutation(model.n)
        for item in index:
            indices_no_replace.append(item)


    w_sgdsw, obj_sgdsw, err_sgdsw = sgd(w_init, model, indices_no_replace, shrinking_steps, w_min, n_iter, averaging_on=False, momentum=0, verbose=True, start_averaging =0)

    # Error of objective on a logarithmic scale
    plt.figure(figsize=(7, 5))
    plt.yscale("log")
    plt.plot(obj_sgdss - obj_min, label="SGD switch replace", lw=2)
    plt.plot(obj_sgdsw - obj_min, label="SGD withoutreplace", lw=2)
    plt.title("Convergence plot", fontsize=16)
    plt.xlabel("#iterations", fontsize=14)
    plt.ylabel("Loss function", fontsize=14)
    plt.legend()
    plt.show()
    # Distance to the minimizer on a logarithmic scale
    plt.figure(figsize=(7, 5))
    plt.yscale("log")
    plt.plot(err_sgdss , label="SGD replace", lw=2)
    plt.plot(err_sgdsw , label="SGD withoutreplace", lw=2)
    plt.title("Convergence plot", fontsize=16)
    plt.xlabel("#iterations", fontsize=14)
    plt.ylabel("Distance to the minimum", fontsize=14)
    plt.legend()
    plt.show()

    w_sgdsaw, obj_sgdsaw, err_sgdsaw = sgd(w_init, model, indices_no_replace, shrinking_steps, w_min, n_iter, averaging_on=True, momentum=0, verbose=True, start_averaging = int(start_averaging))
    complexityofGD = n * np.arange(0, datapasses + 1)
    
    plt.figure(figsize=(7, 5))
    plt.yscale("log")
    plt.semilogy(err_sgdsw , label="SGD withoutreplace", lw=2)
    plt.semilogy(err_sgdsaw , label="SGD averaging end", lw=2)
    plt.semilogy(complexityofGD, err_gd , label="GD", lw=2)
    plt.title("Convergence plot", fontsize=16)
    plt.xlabel("#iterations", fontsize=14)
    plt.ylabel("Distance to true model", fontsize=14)
    plt.legend()
    plt.show()



    w_sgdar, obj_sgdar, err_sgdar = sgd(w_init, model, indices_no_replace, decreasing_steps, w_model_truth, n_iter, averaging_on=True, momentum=0.7, verbose=True, start_averaging = int(start_averaging))

    w_sgdsw, obj_sgdsw, err_sgdsw = sgd(w_init, model, indices_no_replace, decreasing_steps, w_model_truth, n_iter, averaging_on=False, momentum=0, verbose=True, start_averaging = 0)

    w_gd, obj_gd, err_gd = gd(w_init, model, step, w_model_truth, datapasses)


    # Distance to the minimizer on a logarithmic scale
    plt.figure(figsize=(7, 5))
    plt.yscale("log")
    plt.semilogy(err_sgdsw , label="SGD withoutreplace", lw=2)
    plt.semilogy(err_sgdar , label="SGD averaging end", lw=2)
    plt.semilogy(complexityofGD, err_gd , label="GD", lw=2)
    plt.title("Convergence plot", fontsize=16)
    plt.xlabel("#iterations", fontsize=14)
    plt.ylabel("Distance to true model", fontsize=14)
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()