import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

def diameter_estimation_brightfield(image, spheroid_diameter_path=0, image_idx=0, kernel_size=7, num_filter_iter=25):

    grey_image = cv2.normalize(src=image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                               dtype=cv2.CV_8UC1)
    grey_image_laplacian = cv2.Laplacian(grey_image, ddepth=-1)
    kernel_mean_averaging = np.ones((kernel_size,kernel_size))/(kernel_size**2)

    filtered_thresh = grey_image_laplacian
    for i in range(num_filter_iter):
        ret, thresh = cv2.threshold(filtered_thresh, 0, 255, cv2.THRESH_OTSU)
        filtered_thresh = cv2.filter2D(thresh, -1, kernel_mean_averaging)

    distance = cv2.distanceTransform(filtered_thresh, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    _, max_val, _, centre = cv2.minMaxLoc(distance)
    circle = cv2.circle(grey_image, centre, int(max_val), (255, 0, 0), 2)
    if spheroid_diameter_path:
        fig = plt.figure()
        plt.imshow(circle)
        fig.savefig(os.path.join(spheroid_diameter_path, str(image_idx) + '.png'))
    diameter = max_val * 2 # pixels

    return diameter, centre

def crop_spheroid_TP(image_path, cropped_spheroid_wh, spheroid_start_x, spheroid_start_y):
    img = cv2.imread(image_path, -1)
    img = img[spheroid_start_x:spheroid_start_x + cropped_spheroid_wh,
          spheroid_start_y:spheroid_start_y + cropped_spheroid_wh]
    cropped_image = np.reshape(img, (1, cropped_spheroid_wh, cropped_spheroid_wh))
    return cropped_image


def crop_cluster_TP(all_locations, image, crop_cluster_cell_wh, cropped_images_path, label=''):
    print('cropping TP clusters...')

    for location_idx in range(len(all_locations)):
        frame_idx = all_locations['frame_idx'][location_idx]
        x_location = all_locations['x_TP_location'][location_idx]
        y_location = all_locations['y_TP_location'][location_idx]
        location_idx_from_df = all_locations['localization_index'][location_idx]

        image_reshaped = image[int(frame_idx), :, :]
        image_H = np.shape(image_reshaped)[1]

        crop_image = image_reshaped[int(x_location - crop_cluster_cell_wh/2): min(image_H-1,int(x_location+ crop_cluster_cell_wh/2 + 1)),
                     max(0, int(y_location - crop_cluster_cell_wh/2)):int(y_location + crop_cluster_cell_wh/2 + 1)]
        try:
            print('cluster idx: ' + str(location_idx))
            cv2.imwrite(os.path.join(cropped_images_path, 'frame_' + str(int(frame_idx)).zfill(4) + '_location_' +
                                     str(location_idx_from_df).zfill(4) + label + '.tif'),
                    crop_image)
        except:
            print('i ' + str(location_idx) + ' failed')
