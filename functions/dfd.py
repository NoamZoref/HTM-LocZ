import numpy as np
import os
import cv2
import pandas as pd

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from numpy.linalg import norm
from scipy.signal import fftconvolve


def bg_removal_frame(image, ratio=1, reduce=0):
    bg = np.sum(image[0,:]) + np.sum(image[-1,:]) + np.sum(image[:,0]) +  np.sum(image[:,-1])
    bg = bg/(2*image.shape[0] + 2*image.shape[1] - 4)
    bg = bg*ratio
    bg = bg - reduce
    image = np.maximum(0, np.int16(image - bg))
    return image

def bg_removal_16(image, ratio=1):
    bg = np.int16(image[0, 0] + image[0, -1] + image[-1, 0] + image[-1, -1] +
                  image[0, 1] + image[0, -2] + image[-1, 1] + image[-1, -2] +
                  image[1, 0] + image[1, -1] + image[-2, 0] + image[-2, -1] +
                  image[1, 1] + image[1, -2] + image[-2, 1] + image[-2, -2])
    bg = bg/16
    bg = bg*ratio
    image = np.maximum(0, np.int16(image - bg))
    return image

def deconvolution_lucy_richardson(image, psf, n):
    orig_image = image
    curr_deconv_image = np.ones_like(image) * (np.sum(image) / (np.shape(image)[0] * np.shape(image)[1]))
    epsilon = 1e-10

    for i_iter in range(n):
        curr_deconv_image = np.multiply(curr_deconv_image,
                                        fftconvolve(np.divide(orig_image, fftconvolve(curr_deconv_image, psf,
                                                                                      mode='same') + epsilon),
                                                    psf[::-1, ::-1], mode='same'))
    return curr_deconv_image

def z_dfd(bead_tif_path, delta_z, bead_center_slice, crop_bead, num_couple_slices, image_z0_path, image_z0_plus_path,
          num_iterations, delta_from_each_image=0):

    bead_tiff_images = cv2.imreadmulti(bead_tif_path, [], cv2.IMREAD_ANYDEPTH)[1]

    index_start = (bead_center_slice - 1) - num_couple_slices / 2 - delta_z / 2 + 1
    if (index_start - bead_center_slice) < (-75):
        index_start = bead_center_slice - 75

    index_end = index_start + num_couple_slices - 1
    if (index_end - bead_center_slice) > 75:
        index_end = bead_center_slice + 75

    num_couple_slices = index_end + 1 - index_start

    image_z0 = cv2.imread(image_z0_path,cv2.IMREAD_ANYDEPTH)
    image_z0_plus = cv2.imread(image_z0_plus_path, cv2.IMREAD_ANYDEPTH)

    image_z0 = bg_removal_frame(image_z0, ratio=0.98)
    image_z0_plus = bg_removal_frame(image_z0_plus, ratio=0.98)

    M, N = image_z0.shape

    correlation_volume = np.zeros((M, N, num_couple_slices))

    for bead_idx in range(index_start, index_end):

        # bead image
        bead_tiff_image = bead_tiff_images[bead_idx]
        # if it's -50 & 50
        if delta_from_each_image:
            bead_tiff_image = bead_tiff_images[bead_idx-delta_z]

        bead_tiff_image = bg_removal_16(bead_tiff_image)

        if crop_bead:
            bead_tiff_image = bead_tiff_image[crop_bead:-crop_bead, crop_bead:-crop_bead]

        # bead image plus
        bead_tiff_image_plus = bead_tiff_images[bead_idx+delta_z]
        bead_tiff_image_plus = bg_removal_16(bead_tiff_image_plus)

        if crop_bead:
            bead_tiff_image_plus = bead_tiff_image_plus[crop_bead:-crop_bead, crop_bead:-crop_bead]

        deconv_lucy = deconvolution_lucy_richardson(image_z0/norm(image_z0),
                                                  bead_tiff_image/norm(bead_tiff_image),
                                                  n=num_iterations)
        deconv_lucy_plus = deconvolution_lucy_richardson(image_z0_plus/norm(image_z0_plus),
                                                       bead_tiff_image_plus/norm(bead_tiff_image_plus),
                                                       n=num_iterations)

        correlation_volume[:,:,bead_idx-index_start] = fftconvolve(deconv_lucy[::-1, ::-1]/norm(deconv_lucy),
                                                                   deconv_lucy_plus/norm(deconv_lucy_plus), mode='same')

    correlation_z_slice = np.max(correlation_volume, axis=(0,1))

    smooth_correlation_z_slice = gaussian_filter1d(correlation_z_slice[:-1], sigma=3)
    peaks, _ = find_peaks(smooth_correlation_z_slice,
                            prominence=0.001)

    peaks_with_edges = np.concatenate(([0], peaks, [149]))
    peak_values_with_edges = smooth_correlation_z_slice[peaks_with_edges]

    corr_score = np.max(peak_values_with_edges)
    idx_max = peaks_with_edges[np.argmax(peak_values_with_edges)]

    idx_max_relative = index_start + idx_max - bead_center_slice
    z = -idx_max_relative

    return z, corr_score


def localizations_from_dfd(images_path, bead_tif_path, delta_z, bead_center_slice, crop_bead,
                           num_couple_slices, num_iterations,
                           delta_from_each_image, columns, localizations_results_df_path, reversed_plus=0):

    file_names = []
    for image_name in os.listdir(images_path):
        if not '_plus' in image_name:
            print(image_name)
            file_names.append(image_name)

            localization_idx = int(image_name[-8:-4])

            if reversed_plus:
                image_z0_plus_path = os.path.join(images_path, image_name)
                image_z0_path = os.path.join(images_path, image_name[:-4] + '_plus.tif')
            else:
                image_z0_path = os.path.join(images_path, image_name)
                image_z0_plus_path = os.path.join(images_path, image_name[:-4] + '_plus.tif')

            z, corr_score = z_dfd(bead_tif_path, delta_z, bead_center_slice, crop_bead, num_couple_slices,
                                  image_z0_path, image_z0_plus_path,
                                  num_iterations,
                                  delta_from_each_image=delta_from_each_image)

            localization = np.hstack((localization_idx, z, corr_score))
            localization_df = pd.DataFrame([localization], columns=columns)

            localization_df.to_csv(localizations_results_df_path, mode='a', header=False, index=False)

    return

