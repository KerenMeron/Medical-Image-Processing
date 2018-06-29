'''
Medical Image Processing Course, Exercise 1, Part 2
Create ROI for spine and lungs.
Keren Meron
'''

import numpy as np
import nibabel as nib
from scipy.ndimage import morphology
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, convex_hull_image
import mip_ex1_partA

MIN_THRESH = -500
MAX_THRESH = 2000


def IsolateBody(ct_scan):
    '''
    Isolate the patient's body from air and scan gantry.
    :param ct_scan: [numpy.ndarray]
    :return: Body segmentation [numpy.ndarray]
    '''
    segmented = mip_ex1_partA.SegmentationByTH(ct_scan, MIN_THRESH, MAX_THRESH).astype(int)

    # return largest connected component which is also segmented (not air)
    labels, num_labels = label(segmented, return_num=True)
    largest_bin = largest_CC(labels, num_labels, segmented)

    final_segmented = np.zeros(segmented.shape)
    final_segmented[np.where(labels == largest_bin)] = 1
    return final_segmented


def largest_CC(labeled_data, num_labels, segmentation, second=False):
    '''
    Find largest connected component.
    :param labeled_data: [numpy.ndarray] as labeled by skimage.measure.label
    :param num_labels: [int] number of labels in labeled_data
    :param segmentation: [numpy.ndarray] original segmentation on which labels was found
    :param second: (optional) [bool] If true, find also the second largest CC.
    :return: label of largest connected component
             (optional) label of second largest connected component
    '''
    hist, bins = np.histogram(labeled_data, bins=np.arange(num_labels))
    if len(hist) == 1:
        return 0

    labels_sorted = bins[np.argsort(hist)[::-1]]
    largest = largest_CC_not_bg(segmentation, labels_sorted, labeled_data)
    if not second:
        return largest

    # if required, find second largest connected component
    largest_idx = np.where(labels_sorted == largest)[0]
    second_largest = largest_CC_not_bg(segmentation, labels_sorted, labeled_data, largest_idx+1)

    return largest, second_largest


def largest_CC_not_bg(segmentation, labels_sorted, labels, largest_idx=0):
    ''':return largest connected component label which is not background (air).'''

    height, width, slices = segmentation.shape
    label_idx = largest_idx
    largest = labels_sorted[label_idx]
    iters = 0

    # find first zero (not segmented) index
    first_non_segmented_idx = (0, 0, 0)
    for i, j, k in zip(range(height), range(width), range(slices)):
        if segmentation[i, j, k] == 0:
            first_non_segmented_idx = (i, j, k)
            break

    # don't choose label which covers more than 45% of scan
    threshold_partial = segmentation.size * 0.45
    num_labeled_largest = np.count_nonzero(labels == largest)

    # assume there are max 2 large background connected components
    while iters < 3 and label_idx < len(labels_sorted) and (labels[first_non_segmented_idx] == largest or
            labels[0,0,0] == largest or labels[height-1, width-1, slices-1] == largest
                or num_labeled_largest > threshold_partial):
        iters += 1
        label_idx += 1
        largest = labels_sorted[label_idx]
        num_labeled_largest = np.count_nonzero(labels == largest)
    return largest


def inferior_slice(segmentation):
    '''
    Define lower cross section to the segmented parts in the data.
    :param segmentation: [numpy.ndarray] segmentation of lungs.
    :return: index of lowest slice with segmentation (by axial z-axis)
    '''
    labels, num_labels = label(segmentation, return_num=True)
    largest_label = largest_CC(labels, num_labels, segmentation)

    # z-axis is the axial plane
    segmented_area_per_slice = np.sum(segmentation, axis=(0,1))
    for i in range(segmented_area_per_slice.size):
        # find first slice with area above threshold, which is lowest slice of lungs
        if segmented_area_per_slice[i] > 10:
            # make sure than slice includes both lungs, in case they are a single CC and there is noise
            if np.argwhere(labels[:, :, i] == largest_label).any():
                return i


def widest_slice(segmentation):
    '''
    Define widest cross section to the segmented parts in the data.
    Analogous to 'deepest' slice in sagittal plane (y-axis).
    :param segmentation: [numpy.ndarray] segmentation of lungs.
    :return: index of widest slice with segmentation (by axial z-axis)
    '''
    # z-axis is the axial plane
    segmented_area_per_slice = np.sum(segmentation, axis=(0,1))
    return np.argmax(segmented_area_per_slice)


def IsolateBS(body_segmentation):
    '''
    Isolate the breathing system from the given body segmentation.
    :param body_segmentation: [numpy.ndarray]
    :return: lungs segmentation [numpy.ndarray], BB, CC
    '''
    # find 1/2 holes inside body which should be lungs
    inverse_body = 1 - body_segmentation

    labels, num_labels = label(inverse_body, return_num=True)
    largest, second_largest = largest_CC(labels, num_labels, inverse_body, second=True)

    lungs_segmentation = np.zeros(body_segmentation.shape).astype(int)
    lungs_segmentation[np.where((labels == largest) | (labels == second_largest))] = 1

    # get indices of BB and CC slices, on z-axis (axial)
    BB = inferior_slice(lungs_segmentation)
    CC = widest_slice(lungs_segmentation)

    return lungs_segmentation, BB, CC


def ThreeDBand(body_segmentation, lungs_segmentation, BB, CC):
    '''
    Create 3D band around breathing system.
    :param body_segmentation: [numpy.ndarray] from IsolateBody
    :param lungs_segmentation: [numpy.ndarray] from IsolateBS
    :param BB: index of lower slice of lungs
    :param CC: index of widest slice of lungs
    :return: [numpy.ndarray] with 3d band around lungs
    '''
    # fill up the lungs so that they will have a cubic shape
    dist = (CC - BB) / 3
    fill_in_shape = (lungs_segmentation.shape[0], lungs_segmentation.shape[1], dist)
    widest_slice_lungs = np.repeat(lungs_segmentation[:, :, CC], dist).reshape(fill_in_shape)
    try:
        lungs_segmentation[:, :, CC-dist:CC] = widest_slice_lungs
    except ValueError:
        lungs_segmentation[:, :, CC-dist+1:CC] = widest_slice_lungs

    # traverse over slices from BB to CC and get convex hull of each slice over lungs
    lungs_hull = np.zeros(lungs_segmentation.shape)
    body_hull = np.zeros(body_segmentation.shape)

    # prep arrays as order 'C' for convex_hull_image
    lungs_segmentation = np.copy(lungs_segmentation, order='C')
    body_segmentation = np.copy(body_segmentation, order='C')

    for i in range(BB, CC):

        if np.count_nonzero(lungs_segmentation[:, :, i]) != 0:
            try:
                lungs_hull[:, :, i] = convex_hull_image(lungs_segmentation[:, :, i])
            except ValueError:
                if i != BB:
                    lungs_hull[:, :, i] = lungs_hull[:, :, i-1]

        if np.count_nonzero(body_segmentation[:, :, i]) != 0:
            try:
                body_hull[:, :, i] = convex_hull_image(body_segmentation[:, :, i])
            except ValueError:
                if i != BB:
                    body_hull[:, :, i] = body_hull[:, :, i-1]

        if i % 30 == 0:
            print (i)

    if np.count_nonzero(body_hull) == 0:
        body_hull = body_segmentation

    # remove the convex hull from the body hull in order to get the inner region
    inner_area = body_hull - lungs_hull
    inner_area[np.where(inner_area == -1)] = 0

    # zero all body parts not in area of interest
    inner_area[:, :, :BB] = 0
    inner_area[:, :, CC:] = 0

    return inner_area


def spineROI(skeleton_seg, aorta_seg):
    '''
    Create ROI of the spine.
    :param skeleton_seg: [numpy.ndarray] segmentation of skeleton
    :param aorta_seg: [numpy.ndarray] segmentation of aorta
    :return: [numpy.ndarray] spine ROI
    '''
    length_properties = np.unique(np.where(aorta_seg != 0)[2])
    lowest_slice = length_properties[0]
    highest_slice = length_properties[-1]
    num_slices = highest_slice - lowest_slice

    # find coordinates bounding the skeleton
    horizontal_y_axis_bounds = np.zeros((num_slices, 2)).astype(int)
    horizontal_x_axis_bounds = np.zeros((num_slices, 2)).astype(int)

    for i in range(lowest_slice, highest_slice):
        slice = skeleton_seg[:, :, i]

        # try to disconnect the ribs from the spine
        slice = morphology.binary_erosion(slice, iterations=2)

        # choose spine bulk as largest connected component
        labels, num_labels = label(slice, return_num=True)

        # if spine is missing, set middle of slice
        if num_labels == 1:
            horizontal_x_axis_bounds[i - lowest_slice] = horizontal_x_axis_bounds[i - lowest_slice - 1]
            horizontal_y_axis_bounds[i - lowest_slice] = horizontal_y_axis_bounds[i - lowest_slice - 1]
            continue

        # take biggest CC
        hist, bins = np.histogram(labels, bins=np.arange(num_labels + 1))
        sorted_bins = bins[np.argsort(hist)[::-1]]
        biggest_label = sorted_bins[1]

        clean_slice = np.zeros(slice.shape)
        clean_slice[np.where(labels == biggest_label)] = 1

        x, y = np.where(clean_slice == 1)

        horizontal_x_axis_bounds[i - lowest_slice] = np.array([np.min(x), np.max(x)])
        horizontal_y_axis_bounds[i - lowest_slice] = np.array([np.min(y), np.max(y)])


    horiz_x_axis_avg_inner = np.average(horizontal_x_axis_bounds[:, 0]).astype(int)
    horiz_x_axis_avg_outer = np.average(horizontal_x_axis_bounds[:, 1]).astype(int)

    # build ROI
    empty = np.zeros(skeleton_seg.shape)
    for i in range(lowest_slice, highest_slice):
        empty[horiz_x_axis_avg_inner:horiz_x_axis_avg_outer,
            horizontal_y_axis_bounds[i-lowest_slice,0]:horizontal_y_axis_bounds[i-lowest_slice,1], i] = 1

    return empty


def mergedROI(body_scan, aorta_scan, body_path):
    '''
    Extract ROI from given scans.
    :return: file name of saved file. 
    '''
    skeleton_file = mip_ex1_partA.perform_segmentation(body_path)
    skeleton = np.array(nib.load(skeleton_file).get_data())
    spine_roi = spineROI(skeleton, aorta_scan)

    body_segmentation = IsolateBody(body_scan)
    lungs, BB, CC = IsolateBS(body_segmentation)
    lungs_roi = ThreeDBand(body_segmentation, lungs, BB, CC)

    final_roi = np.zeros(lungs_roi.shape)
    final_roi[np.where(np.rint(spine_roi) == 1)] = 1
    final_roi[np.where(np.rint(lungs_roi) == 1)] = 1
    return mip_ex1_partA.save_nifti_and_calculate_CC(final_roi, body_path, body_path, '_ROI')


if __name__ == '__main__':
    #######################################################################
    # please edit body_path and aorta_path in order to create merged ROI.
    #######################################################################

    body_path = 'files/Case3_CT.nii.gz'
    img_body = np.array(nib.load(body_path).get_data())
    aorta_path = 'files/Case3_Aorta.nii.gz'
    img_aorta = np.array(nib.load(aorta_path).get_data())
    print(mergedROI(img_body, img_aorta, body_path))
