'''
Medical Image Processing Course, Exercise 1, Part C
Perform Seeded Region Growing algorithm for Bone Segmentation.
Keren Meron
'''

import numpy as np
import nibabel as nib
from scipy.ndimage import morphology
import mip_ex1_partB
import mip_ex1_partA


NUM_SEEDS = 200


def multipleSeedsRG(ct_scan, roi, skeleton):
    '''
    Perform seeded regions growing inside given region of interest.
    :param ct_scan: [numpy.ndarray]
    :param roi: [numpy.ndarray] same shape as ct_scan
    :param skeleton: segmentation of skeleton based on thresholding
    :return: Bone segmentation [numpy.ndarray]
    '''
    # indices (rows, cols, slices) of seeds in scan
    seeds = extractSeeds(ct_scan, roi, skeleton)

    labeled = np.zeros(ct_scan.shape)
    labeled[seeds] = 1
    new_labeled = singleRegionGrowing(ct_scan, np.copy(labeled))

    counter = 0

    while (labeled != new_labeled).any():
        labeled = new_labeled
        new_labeled = singleRegionGrowing(ct_scan, np.copy(labeled))
        counter += 1

    return new_labeled


def percentage_labeled(segmentation):
    ''':return percentage of given image which is labeled.'''
    num_labeled = np.sum(segmentation)
    return float(num_labeled / segmentation.size)


def reg_homogeneity(original, pixels, region):
    '''
    Apply homogeneity function over all pixels in given data array.
    :param original: ct_scan based on
    :param pixels: [numpy.ndarray] array with scan values only for pixels in question
    :param region: (rows, cols, slices) indices inside existing ROI
    :return: results, same shape as pixels
    '''
    mu = (np.mean(original[region]))
    ans = np.abs(pixels - mu)
    return ans


def homogeneity(original, pixels, region):
    '''
    Apply homogeneity function over all pixels in given data array.
    :param original: ct_scan based on
    :param pixels: [numpy.ndarray] array with scan values only for pixels in question
    :param region: (rows, cols, slices) indices inside existing ROI
    :return: results, same shape as pixels
    '''
    mu = np.mean(original[region])
    if mu > 200:
        mu = 200
    dist = pixels - mu
    std = np.std(original[region])
    ans = np.abs(dist / std)
    return ans


def singleRegionGrowing(ct_scan, labeled):
    '''
    Perform a single round of region growing.
    :param ct_scan: original scan
    :param labeled: same shape as ct_scan, with 1s only on already labeled inside region.
    :return: updated labeled.
    '''
    # remove original labels from dilation, to get the new neighbors
    dilated = morphology.binary_dilation(labeled)
    neighbors = dilated - labeled
    ct_new_region = (ct_scan * neighbors).astype(int)  # ct_scan values only for new region pixels

    # take pixels whose value is larger than the mean of the existing region
    mu = np.mean(ct_scan[np.where(labeled == 1)])
    if mu > 200:
        mu = 200
    new_in_region = np.where(ct_new_region > mu)
    labeled[new_in_region] = 1

    return labeled


def extractSeeds(ct_scan, roi, skeleton):
    '''
    Choose seeds from given scan for region growing algorithm.
    extract N seeds uniformly at random from the brightest pixels from ct which are inside the ROI.
    :param ct_scan: [numpy.ndarray]
    :param roi: [numpy.ndarray] same shape as ct_scan
    :param skeleton: segmentation of skeleton based on thresholding
    :return: (rows, cols, slices) indices of seeds.
    '''

    roi_skeleton = (roi * skeleton)
    roi_skeleton_ct = (roi_skeleton * ct_scan).astype(float)

    # find 10% brightest pixels
    roi_skeleton_ct_nonzero = roi_skeleton_ct[np.nonzero(roi_skeleton_ct)]
    hist, bins = np.histogram(roi_skeleton_ct_nonzero, bins=int(roi_skeleton_ct_nonzero.size * 0.1))

    hist_shift = np.concatenate(([hist[0]], hist))[:-1]
    diff = hist - hist_shift
    brightest_idx = np.argmax(diff)
    brightest_threshold = bins[brightest_idx]

    seeds_db = np.where((roi_skeleton_ct > brightest_threshold))

    # choose uniformly at random from the base of points
    random_indices = np.rint(np.random.uniform(0, seeds_db[0].size-1, NUM_SEEDS)).astype(int)
    final_seeds = seeds_db[0][random_indices], seeds_db[1][random_indices], seeds_db[2][random_indices]

    return final_seeds


def segmentBones(ctFileName, AortaFileName, outputFileName):
    '''
    Perform all operations for segmenting bone skeleton of ct scan.
    :param ctFileName: name of ct file
    :param AortaFileName:  segmentation of aorta
    :param outputFileName: name of file to output result
    :return: outputFileName
    '''
    ct_nii = nib.load(ctFileName)
    body_scan = np.array(ct_nii.get_data())
    aorta_scan = np.array(nib.load(AortaFileName).get_data())
    roi_file = mip_ex1_partB.mergedROI(body_scan, aorta_scan, ctFileName)
    roi = np.array(nib.load(roi_file).get_data())
    skeleton_file = mip_ex1_partA.perform_segmentation(ctFileName)
    skeleton = np.array(nib.load(skeleton_file).get_data())
    bone_segmentation = multipleSeedsRG(body_scan, roi, skeleton)

    # save result
    bone_nii = nib.Nifti1Image(bone_segmentation, ct_nii.affine, ct_nii.header)
    nib.save(bone_nii, outputFileName)
    return outputFileName


def evaluateSegmentation(ground_truth, segmentation):
    '''
    Calculate the volume overlap between a created and the ground truth segmentation.
    Error: 1 - 2 * (Intersection / Union)
    :return: Overlap Error
    '''
    # find enclosing box
    truth_nonzero = np.nonzero(ground_truth)
    x_min, y_min, z_min = np.min(truth_nonzero[0]), np.min(truth_nonzero[1]), np.min(truth_nonzero[2])
    x_max, y_max, z_max = np.max(truth_nonzero[0]), np.max(truth_nonzero[1]), np.max(truth_nonzero[2])
    box_segmentation = segmentation[x_min:x_max, y_min:y_max, z_min:z_max]
    box_truth = ground_truth[x_min:x_max, y_min:y_max, z_min:z_max]

    # calculate overlap error
    intersection = np.count_nonzero((box_truth * box_segmentation).astype(int))
    union = np.count_nonzero(box_truth) + np.count_nonzero(box_segmentation)
    # real union definition:
    # union_mat = box_truth + box_segmentation
    # union_mat[np.where(union_mat > 0)] = 1  # turn 2s into 1s
    # union = np.count_nonzero(union_mat)
    return 1 - 2 * (intersection / union)


if __name__ == '__main__':

    # EXAMPLE RUN
    ct_file = 'Case1_CT.nii.gz'
    roi_file = 'Case1_CT_ROI.nii.gz'
    output_file = 'Case1_CT_bone_segmentation.nii.gz'
    output_file = segmentBones(ct_file, roi_file, output_file)
    result = np.array(nib.load(output_file.get_data()))
    truth = np.array(nib.load('Case1_L1.nii').get_data())
    error = evaluateSegmentation(truth, result)
    print("Segmentation error $.4f" % error)
