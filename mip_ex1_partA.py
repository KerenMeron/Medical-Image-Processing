'''
Medical Image Processing Course, Exercise 1, Part 1
Segmentation of MRI scans, using thresholds.
Keren Meron
'''

import numpy as np
import nibabel as nib
from scipy.ndimage import morphology
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

IMAX = 1300


def SegmentationByTH(img_data, Imin, Imax, seg_file_name='', original_filename='', save_result=False):
    '''
    Create segmentation file of bone skeleton from given nifti file. 
    :param img_data: [numpy.ndarray] input image
    :param Imin: minimal threshold for bones
    :param Imax: maximal threshold for bones
    :param seg_file_name: name to save segmentation file
    :param original_filename: name of original nifti file.
    :param save_result: [bool] if True, will save resulting segmentation.
    :return: [numpy.ndarray] segmentation image
    '''
    segmentation = np.zeros(img_data.shape)
    segmentation[np.where((img_data > Imin) & (img_data < Imax))] = 1
    if save_result:
        save_nifti_and_calculate_CC(segmentation, original_filename, seg_file_name, '')
    return segmentation


def SkeletonTHFinder(img_data, Imax, original_filename, show_plot=False):
    '''
    Find best lower threshold for segmentation.
    Plot and count connected components of segmentation formed by different lower threshold values.
    :param img_data: [numpy.ndarray] input image
    :param Imax: maximal threshold to use for segmentation
    :param original_filename: name of original nifti file.
    :param show_plot: [boo] if True, will display plot of thresholds.
    :return list of connected components per each Imin checked
    '''
    num_connected_components = []
    for Imin in range(150, 500, 5):
        seg_file = strip_nifti_filename(original_filename, '_seg_%d_%d' % (Imin, IMAX))
        segmentation = SegmentationByTH(img_data, Imin, Imax, seg_file, original_filename)
        _, num_labels = label(segmentation, return_num=True)
        num_connected_components.append(num_labels)

    if show_plot:
        plt.figure()
        plt.scatter(list(range(150, 500, 5)), num_connected_components)
        plt.show()
        plt.ion()

    return num_connected_components


def strip_nifti_filename(filename, suffix):
    '''
    Prepare unique filename according to given parameters, for nii.gz file.
    :param filename: [str] original file name.
    :param suffix: [str] ending to add to file name
    :return: [str] file name
    '''
    stripped_filename = filename
    if stripped_filename.endswith('.gz'):
        stripped_filename = stripped_filename[:-3]
    if stripped_filename.endswith('.nii'):
        stripped_filename = stripped_filename[:-4]
    stripped_filename += suffix + '.nii.gz'
    return stripped_filename


def morphological_operations_on_segmentation(segmented_img, original_file='', get_data_array=False):
    '''
    Perform various morphological operations on given segmentation, 
    in order to reduce connected components, close and fill holes.
    :param segmented_img: [ndarray] of segmented image
    :param original_file: [ndarray] of original image.
    :return: file name of segmentation file after morphological operations.
    '''

    _, orig_num_labels = label(segmented_img, return_num=True)
    saved_imgs = {}

    for i in range(5):
        segmented_img = morphology.binary_erosion(segmented_img)
        segmented_img = morphology.binary_dilation(segmented_img)
        segmented_img = morphology.binary_dilation(segmented_img)
        segmented_img = remove_small_objects(segmented_img, min_size=12100)
        labels, num_labels = label(segmented_img, return_num=True)
        saved_imgs[num_labels] = segmented_img

    # uncomment following in order to see size of segmentation components
    # for region in regionprops(labels):
    #     print(region.area)

    min_label = min(saved_imgs.keys())
    if get_data_array:
        return saved_imgs[min_label]
    return save_nifti_and_calculate_CC(saved_imgs[min_label], original_file, original_file, '_SkeletonSegmentation')


def save_nifti_and_calculate_CC(data_array, original_img, file_name, file_name_suffix, calculate_CC=False):
    '''
    Calculate number of connected components in given image, and save as nifti file.
    :param data_array: image to save, a segmentation.
    :param original_img: [str] original image file, which data_array was segmented from.
    :param file_name: base file name
    :param file_name_suffix: ending for file name
    :param calculate_CC: [bool] if True, calculate number of connected components and print result.
    :return name of saved file
    '''
    orig = nib.load(original_img)
    data_nii = nib.Nifti1Image(data_array, orig.affine, orig.header)
    new_filename = strip_nifti_filename(file_name, file_name_suffix)
    nib.save(data_nii, new_filename)
    if calculate_CC:
        _, num_labels = label(data_array, return_num=True)
        print(new_filename + ' with Num CC: ' + str(num_labels))
    return new_filename

def choose_min_thresh(num_cc):
    '''
    Choose the best Imin from a list of connected components calculated with each threshold.
    :param num_cc: list of cc, matching Imin of: range(150, 500, 5)
    :return: [int] Imin
    '''
    # exclude 'tail' of too high values
    less_num_cc = np.array(num_cc[:28])
    best_idx = np.argmin(less_num_cc)
    best_idx = (best_idx * 5) + 150
    return best_idx


def perform_segmentation(nifti_path, Imin=None):
    '''
    Perform entire segmentation process: choose threshold, do segmentation and add fixes to result.
    :param nifti_path: [str] name of nifti file to open 
    :param Imin: [int] pass minimum threshold value if known, otherwise optimal value will be calculated.
    :return: name of file containing result segmentation
    '''
    # open nifti file
    try:
        img = nib.load(nifti_path)
    except FileNotFoundError:
        nifti_path += '.gz'
        img = nib.load(nifti_path)
    img_data = np.array(img.get_data())

    # choose Imin threshold
    if not Imin:
        print("=> Searching for optimal Imin threshold...")
        cc_list = SkeletonTHFinder(img_data, IMAX, nifti_path)
        Imin = choose_min_thresh(cc_list)
        print("Found: %d" % Imin)

    # create segmentation
    seg_file = strip_nifti_filename(nifti_path, '_seg_%d_%d' % (Imin, IMAX))
    print("==> Performing segmentation by threshold...")
    segmentation_img = SegmentationByTH(img_data, Imin, IMAX, seg_file, nifti_path)
    print("===> Fixing segmentation with morphological operations...")
    final_file = morphological_operations_on_segmentation(segmentation_img, nifti_path)
    print("====> Saved segmentation file at: " + final_file)
    return final_file

if __name__ == '__main__':

    ########################################
    # please edit the following lines in order to perform segmentation on your file
    # pass optional Imin value, otherwise method will find optimal threshold value.
    ########################################

    path = 'Case1_CT.nii'
    perform_segmentation(path, Imin=250)

    # UNCOMMENT IN ORDER TO SEGMENT ALL CASE_i_CT files
    # for i in range(5):
    #     path = 'Case%d_CT.nii' % (i+1)
    #     perform_segmentation(path)
