'''
Perform rigid registration between CT images using radon transformation.
'''

import norm_xcorr
import numpy as np
import scipy.io as spio
from scipy import ndimage
from skimage import feature
from matplotlib import pyplot as plt
from skimage.transform import AffineTransform, warp, radon, rotate


def find_rotation_translation_per_angle(fixed_sinogram, moving_vector):
    '''
    Using rigid registration algorithm, find matching angle and translation parameter delta which yield the 
    maximal cross correlation between the moving vector and the fixed sinogram.
    :param fixed_sinogram: [numpy.ndarray (N,M)] sinogram of an image computed by radon transform.
    :param moving_vector: [numpy.ndarray (N,1)] a vector extracted from moving_sinogram at a certain angle.
    :return: alpha_tag (index of a vector from fixed_sinogram) and delta (translation parameter)
             where (alpha_tag, delta) = argmax CrossCorrelation (moving_vector, delta * fixed_sinogram[alpha_tag]) 
    '''
    # compute NCC for moving_vector with entire fixed_sinogram
    moving_vector = moving_vector[:, None]
    template_match = norm_xcorr.TemplateMatch(moving_vector, 'ncc')
    ncc = template_match(fixed_sinogram)
    max_idx = np.argwhere(ncc == np.max(ncc))[0]  # take first occurrence
    alpha_tag = max_idx[1]
    delta = (fixed_sinogram.shape[0] / 2) - max_idx[0]
    return alpha_tag, delta


def radon_register(fixed_sinogram, fixed_angles_deg, moving_sinogram, moving_angles_deg):
    '''
    Compute the translation pixels and rotation degree used to create the sinogram.
    Uses rigid registration algorithm in radon plane.
    :param fixed_sinogram: sinogram of an image computed by radon transform with fixed_angles_deg as angle
    :param fixed_angles_deg: [float] degree used for radon degree to create fixed_sinogram
    :param moving_sinogram: sinogram of an image computed by radon transform with moving_angles_deg as angle
    :param moving_angles_deg: [float] degree used for radon degree to create moving_sinogram
    :return: rotation_ccw_deg, translation_pixels
    '''
    # find several options of theta (rotation angle) and matching deltas (translation)
    all_alpha_tags, all_deltas = [], []
    all_thetas = []  # difference alpha - alpha tag
    for idx, alpha in enumerate(moving_angles_deg):
        alpha_tag, delta = find_rotation_translation_per_angle(fixed_sinogram, moving_sinogram[:, idx])

        # alpha_tag is in fixed_angles_deg space, translate to [0,180] space
        real_alpha_tag = fixed_angles_deg[alpha_tag]

        all_alpha_tags.append(real_alpha_tag)
        all_deltas.append(delta)
        # take modulo to deal with cyclic rotations
        all_thetas.append((alpha - real_alpha_tag) % np.max(fixed_angles_deg))

    # find highest occurring theta value and corresponding deltas
    hist, bins = np.histogram(all_thetas, bins=fixed_angles_deg)
    theta_max = bins[np.argmax(hist)]

    # construct pairs of delta-alpha matching values, corresponding to the max theta
    closeness_threshold = 1
    matching_deltas_alphas = np.array([[all_deltas[i], moving_angles_deg[i]] for i in range(len(all_deltas)) if
                                       np.abs(all_thetas[i] - theta_max) <= closeness_threshold])

    # construct the normalized direction vector
    normed_alphas = np.zeros((matching_deltas_alphas.shape[0], 2))
    for j in range(matching_deltas_alphas.shape[0]):
        angle = np.deg2rad(matching_deltas_alphas[j, 1])
        normed_alphas[j] = [-(np.cos(angle)), np.sin(angle)]

    # solve the linear set of equations
    deltas = matching_deltas_alphas[:, 0]
    deltas = deltas.reshape(deltas.shape[0], 1)

    xy_delta, _, _, _ = np.linalg.lstsq(normed_alphas, deltas)  # 2x1
    xy_delta = list(xy_delta.flatten())
    return 180 - theta_max, xy_delta


def display_edges_on_image(image1, image2, resize_shape):
    '''
    Display the edges of image2 on top of image1.
    :param image1: Original [numpy.ndarray 2-d]
    :param image2: Transformed [numpy.ndarray 2-d]
    :param resize_shape: [tuple] crop image to display in this shape
    '''
    new_width, new_height = resize_shape
    width, height = image1.shape
    pad_width = (width - new_width) / 2
    pad_height = (height - new_height) / 2

    edges = feature.canny(image2, sigma=5).astype(int) * 255
    image1 -= np.min(image1)
    image1 = (image1 / np.max(image1)) * 255
    image1 += edges

    canvas = image1[pad_width:-pad_width, pad_height:-pad_height]
    plt.figure()
    plt.imshow(canvas, cmap='gray')
    plt.show()


def test_mirrored_transformation(fixed_image, moved_image, rotation, translation):
    '''
    Test whether rotation should be as is, or plus a factor of 180 degrees. Perform reconstruction by both options 
    and choose the one which gives higher correlation between original image and reconstructed image.
    :param fixed_image: [numpy.ndarray] 2d image
    :param moved_image: image after an affine transform was performed on fixed_image
    :param rotation: [float] degrees rotation to perform on moved_image
    :param translation: [list of ints] translation in x and y to perform on moved_image
    :return: [numpy.ndarray] correct reconstructed image
    '''
    # reconstruct image in both ways
    found_transform = AffineTransform(translation=translation)
    registered_img = warp(moved_image, found_transform)
    original_reconstruction = rotate(registered_img, rotation)
    mirror_reconstruction = rotate(registered_img, 180+rotation)

    # compare correlation to fixed image
    template_match = norm_xcorr.TemplateMatch(fixed_image, 'ncc')
    original_ncc = template_match(original_reconstruction)
    mirror_ncc = template_match(mirror_reconstruction)

    if np.max(original_ncc) > np.max(mirror_ncc):
        return original_reconstruction
    else:
        return mirror_reconstruction


def show_radon_registration(img1, rotation_ccw_deg, translation_pixels, num_angles):
    '''
    Perform transformation on first image and use radon registration to retrieve first image from second image.
    Display img1 before and after registration.
    :param img1: [numpy.ndarray] 2d image
    :param rotation_ccw_deg: [int] degree to rotate image
    :param translation_pixels: (x,y) translation to perform on image
    :param num_angles: number of angles for which to perform radon transform on img2
    '''
    # normalize img
    img1 = normalize_image(img1)

    # pad image so that edges will not be cut off
    original_shape = img1.shape
    img1 = pad_image(img1)

    # create transformed image and matching sinograms
    # use inverse transform for AffineTransform
    inverse_translation = [-translation_pixels[0], -translation_pixels[1]]
    transform = AffineTransform(translation=inverse_translation)
    img2 = warp(img1, transform)
    img2 = rotate(img2, angle=-rotation_ccw_deg)  # AffineTransform doesn't rotate around center

    thetas_img1 = np.arange(180)
    img1_sinogram = radon(img1, thetas_img1, circle=True)
    thetas_img2 = np.arange(1, 181, 180/num_angles)
    img2_sinogram = radon(img2, thetas_img2, circle=True)

    # perform registration on img2
    rotation_result, translation_result = radon_register(img1_sinogram, thetas_img1, img2_sinogram, thetas_img2)
    registered_img = test_mirrored_transformation(img1, img2, rotation_result, translation_result)

    # display results
    display_edges_on_image(img1, registered_img, original_shape)


def pad_image(img):
    '''
    Pad given image with zeros at least twice the size of input.
    :param img: [numpy.ndarray] of shape (N,M)
    :return: padded [numpy.ndarray] of shape (2N, 2M)
    '''
    assert img.ndim == 2
    height, width = img.shape
    half_height, half_width = np.floor(height / 2), np.floor(width / 2)
    padded = np.zeros((2 * height, 2 * width))
    padded[half_height: half_height + height, half_width: half_width + width] = img
    return padded


def normalize_image(img):
    '''
    Normalize image values between -1 and 1.
    :param img: [numpy.ndarray] with values between 0 and 255.
    :return: normalized image
    '''
    img = img.astype(float) + np.min(img)
    return img / np.max(img)

if __name__ == '__main__':
    phatom_img = ndimage.imread('Phantom.png', flatten=True)
    show_radon_registration(phatom_img, -60, [75,90], 90)

    mat = spio.loadmat('brain.mat')
    ct_img = mat['brain_fixed']
    show_radon_registration(ct_img, 180, [40, 15], 60)
