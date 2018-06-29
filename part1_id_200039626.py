import scipy.io as spio
import numpy as np
from matplotlib import pyplot as plt
import utils
from skimage import feature, transform as tf
from sklearn.metrics import mean_squared_error, euclidean_distances
from scipy import signal
from mpl_toolkits.mplot3d import Axes3D  # do not remove, needed for surface plot


def load_data():
    ''':return brain_fixed and brain_moving numpy.ndarrays'''
    mat = spio.loadmat('brain.mat')
    brain_fixed = mat['brain_fixed']
    brain_moving = mat['brain_moving']
    return brain_fixed, brain_moving


def display_edges_on_image(image1, image2):
    '''
    Display the edges of image2 on top of image1.
    :param image1: Original [numpy.ndarray 2-d of shape 512,512]
    :param image2: Transformed [numpy.ndarray 2-d of shape 512,512]
    '''
    plt.figure()
    edges = feature.canny(image2, sigma=9).astype(int) * 255
    image1 -= np.min(image1)
    image1 = (image1 / np.max(image1)) * 255
    image1 += edges
    plt.imshow(image1, cmap='gray')
    plt.show()


def display_matching_points(image1, image2, points1=None, points2=None, pad=False, inliers=[], outliers=[]):
    '''
    Display matching points on two subplots.
    :param image1: Brain Fixed [numpy.ndarray 2-d of shape 512,512]
    :param image2: Brain Moving [numpy.ndarray 2-d of shape 512,512]
    :param points1: numpy.ndarray of shape (N,2) with N points
    :param points2: numpy.ndarray of shape (N,2) with N points
    :param pad: [bool] if True, add padding to image2 display
    :param inliers: [list] indices of points which are inliers
    :param outliers: [list] indices of points which are outliers
    '''
    if points1 is None:
        points1 = np.array([])
    if points2 is None:
        points2 = np.array([])

    # breakup into inliers and outliers
    if outliers:
        points1_outliers = points1[np.array(outliers)]
        points1 = points1[np.array(inliers)]
        points2_outliers = points2[np.array(outliers)]
        points2 = points2[np.array(inliers)]

    points_txt = range(len(points1))
    fig = plt.figure()

    ax1 = fig.add_subplot(121)
    ax1.imshow(image1, cmap='gray')
    if points1.any():
        ax1.scatter(points1[:, 0], points1[:, 1], color='white', label='inliers')
    if outliers:
        ax1.scatter(points1_outliers[:, 0], points1_outliers[:, 1], color='green', label='outliers')
        ax1.legend()
    ax1.set_title('Brain Fixed')
    for i, txt in enumerate(points_txt):
        ax1.annotate(txt, (points1[i, 0], points1[i, 1]), fontweight='bold')

    # pad image2 if necessary
    if pad:
        n = image2.shape[0]
        offset = int(n / 2)
        frame = np.zeros((n + n, n + n))
        frame[offset: -offset, offset: -offset] = image2
        points2 += offset
    else:
        frame = image2

    ax2 = fig.add_subplot(122)
    ax2.imshow(frame, cmap='gray')
    if points2.any():
        ax2.scatter(points2[:, 0], points2[:, 1], color='white', label='inliers')
    if outliers:
        ax2.scatter(points2_outliers[:, 0], points2_outliers[:, 1], color='green', label='outliers')
        ax2.legend()
    ax2.set_title('Brain Moving')
    for i, txt in enumerate(points_txt):
        ax2.annotate(txt, (points2[i, 0], points2[i, 1]), fontweight='bold')

    plt.show()


def calcPointBasedReg(fixed_points, moving_points):
    '''
    Find registration between points using SVD which minimizes lle.
    :param fixed_points: numpy.ndarray of shape (N,2) with N points
    :param moving_points: numpy.ndarray of shape (N,2) with N points
    :return: rigidReg [numpy.ndarray shape (3,3)]
    '''
    # compute weighted centroids of points
    fixed_mean = np.mean(fixed_points, axis=0)
    moving_mean = np.mean(moving_points, axis=0)

    # compute centered points
    centered_fixed = fixed_points - fixed_mean
    centered_moving = moving_points - moving_mean

    # compute covariance matrix, 2x2
    weights = np.identity(fixed_points.shape[0])
    covariance = centered_fixed.T.dot(weights).dot(centered_moving)

    # compute SVD
    U, S, V = np.linalg.svd(covariance)
    middle = np.identity(V.shape[1])
    uv_det = np.linalg.det(V.dot(U.transpose()))
    middle[-1, -1] = uv_det

    # extract rotation and translation matrices
    rotation = V.dot(middle.dot(U.transpose()))
    translation = (moving_mean.dot(rotation)).reshape((1, 2))

    # for re-centering
    translation[:, 0] = fixed_mean[0] - translation[:, 0]
    translation[:, 1] = fixed_mean[1] - translation[:, 1]

    # compile together as rigid transformation
    rotation = np.concatenate((rotation, np.zeros((2, 1))), axis=1).reshape((2, 3))
    translation = np.concatenate((translation, np.ones((1, 1))), axis=1).reshape((1, 3))
    return np.vstack((rotation, translation))


def register_points(points, registration):
    '''
    Perform registration operation on points (rotation and translation).
    :param points: numpy.ndarray of shape (N,2) with N points
    :param registration: numpy.ndarray of shape (3,3)
    :return: moved_points: numpy.ndarray of shape (N,2) with N points
    '''
    N = points.shape[0]
    points = np.concatenate((points, np.ones((N, 1))), axis=1).reshape((N, 3))
    transformed = points.dot(registration)
    return transformed[:, :-1]


def calcDist(fixed_points, moving_points, rigidReg, rmse=False):
    '''
    Compute distance of each registered point from the original fixed point.
    :param fixed_points: numpy.ndarray of shape (N,2) with N points
    :param moving_points: numpy.ndarray of shape (N,2) with N points
    :param rigidReg: numpy.ndarray of shape (3,3)
    :param rmse: [bool] if True, calculate and return also the RMSE
    :return: distances vector [N,2]
    '''
    registrated_points = register_points(moving_points, rigidReg)
    diff = euclidean_distances(registrated_points, fixed_points)
    # round up in order to look at metric of number of pixels
    pixels_diff = np.ceil(np.diag(diff))
    if rmse:
        error = root_mse(fixed_points, registrated_points)
        return pixels_diff, error
    return pixels_diff


def root_mse(real, predicted):
    '''
    Calculate the root mean squared error.
    :param real: numpy.ndarray of shape (N,2) with N points
    :param predicted: numpy.ndarray of shape (N,2) with N points
    :return: RMSE [float]
    '''
    return np.sqrt(mean_squared_error(real, predicted))


def calcRobustPointBasedReg(moving_points, fixed_points):
    f, inlier_index = utils.ransac(fixed_points, moving_points, calcPointBasedReg, calcDist, 10, 100, 10, 0.01)
    return f, inlier_index


def register_and_compare_image(fixed_image, moving_image, rigidReg):
    '''
    Perform registration on moved image and compare with original fixed image.
    :param fixed_image: numpy.ndarray of shape (N,N)
    :param moving_image: numpy.ndarray of shape (N,N) 
    :param rigidReg: numpy.ndarray of shape (3,3) 
    '''
    transformation = np.linalg.inv(rigidReg.T)
    warped_img = tf.warp(moving_image, transformation, preserve_range=True)
    # display_matching_points(fixed_image, warped_img, pad=True)
    display_edges_on_image(fixed_image, warped_img)
    NCC(fixed_image, warped_img)

def fastCorrelate(img1, img2):
    # normalize images first
    img1 = (img1 - np.mean(img1)) / np.std(img1)
    img2 = (img2 - np.mean(img2)) / np.std(img2)
    img2flipped = img2[::-1, ::-1]
    convolved = signal.fftconvolve(img1, img2flipped, mode='same')

    normalized_conv = (((convolved - np.min(convolved)) / np.max(convolved)) * 2 - 1)
    return normalized_conv

def NCC(image1, image2):
    '''
    Calculate  alized cross correlation between two images, and plot result.
    :param image1: numpy.ndarray of shape (N,N)
    :param image2: numpy.ndarray of shape (N,N)
    '''
    ncc = fastCorrelate(image1, image2)

    # show ncc histogram
    # plt.figure()
    # plt.hist(ncc.flatten())
    # plt.show()

    # surface plot the NCC score
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(image1.shape[0])
    X, Y = np.meshgrid(x, y)
    zs = np.array([ncc[x, y] for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('NCC Score')
    plt.title('NCC surface')
    plt.show()

def image_registration(with_outliers='no_outliers', robust=False):
    '''
    Perform image registration between images.
    :param with_outliers: [str] whether to use outliers in feature points.
    :param robust: [bool] whether to use robust RANSAC algorithm
    '''

    brain_fixed, brain_moving = load_data()
    fixed_points, moving_points = utils.getPoints(with_outliers)

    # show difference between points
    # display_matching_points(brain_fixed, brain_moving, fixed_points, moving_points)

    if robust:
        rigidReg, inliers = calcRobustPointBasedReg(moving_points, fixed_points)
        outliers = [i for i in range(len(fixed_points)) if i not in inliers]
    else:
        rigidReg = calcPointBasedReg(fixed_points, moving_points)
        inliers, outliers = [], []

    diff, rmse = calcDist(fixed_points, moving_points, rigidReg, rmse=True)
    print(diff, rmse)

    # show registrated points
    registrated_points = register_points(moving_points, rigidReg)
    display_matching_points(brain_fixed, brain_fixed, fixed_points, registrated_points, inliers=inliers, outliers=outliers)

    # show moved image comparison
    register_and_compare_image(brain_fixed, brain_moving, rigidReg)


if __name__ == '__main__':

    image_registration(with_outliers='with_outliers', robust=True)
