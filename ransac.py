import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm 

def run_ransac(data, estimate, is_inlier, sample_size, goal_inliers, max_iterations, stop_at_goal=True, random_seed=None):
    best_ic = 0
    best_model = None
    best_inliers = []
    random.seed(random_seed)
    # random.sample cannot deal with "data" being a numpy array
    data_np = data;
    data = list(data)
    for i in trange(max_iterations):
        s = random.sample(data, int(sample_size))
        m = estimate(s)
        ic = 0
        inliers = np.zeros((data_np.shape[0],1))
        for j in range(len(data)):
            if is_inlier(m, data[j]):
                ic += 1
                inliers[j] = 1;
        # print(s)
        # print('estimate:', m,)
        # print('# inliers:', ic)
        #
        if ic > best_ic:
            best_ic = ic
            best_model = m
            best_inliers = inliers
            if ic > goal_inliers and stop_at_goal:
                break
    print('took iterations:', i+1, 'best model:', best_model, 'explains:', best_ic)
    return best_model, best_ic, best_inliers

def augment(xys):
    axy = np.ones((len(xys), 3))
    axy[:, :2] = xys
    return axy

def estimate(xys):
    axy = augment(xys[:2])
    return np.linalg.svd(axy)[-1][-1, :]

def is_inlier(coeffs, xy, threshold):
    return np.abs(coeffs.dot(augment([xy]).T)) < threshold

def ransac(X,Y):

    n = X.shape[0]
    max_iterations = 300
    goal_inliers = n * 0.6

    # test data


    ax = plt.scatter(X, Y)
    xys = np.vstack((X,Y))

    # RANSAC
    # m, b, i = run_ransac(xys.transpose(), estimate, lambda x, y: is_inlier(x, y, 0.04), 10, goal_inliers, max_iterations, random_seed = 2)
    m, b, i = run_ransac(xys.transpose(), estimate, lambda x, y: is_inlier(x, y, 0.3), 5, goal_inliers, max_iterations, random_seed = 2)

    a, b, c = m
    left,right = plt.xlim()
    plt.plot([left, right], [(-c-left*a)/b, -(c+right*a)/b], color=(0, 1, 0))

    plt.show()
    return i, m
