import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import peak_local_max
import os
from IPython.display import display

def TP_locations(image, frame_idx, window_size: int, min_distance: int, percentile,
                 plot, save_plot, save_plot_path='', gauss_filter=False, crop_cluster_cell_wh=False):

    if window_size%2:
        image_gauss = cv2.GaussianBlur(image, ksize=[window_size, window_size], sigmaX=20)
    else:
        image_gauss = cv2.GaussianBlur(image, ksize=[window_size+1, window_size+1], sigmaX=20)

    if gauss_filter:
        coordinates = peak_local_max(image_gauss, min_distance=min_distance)
        image_value_coordinates = image_gauss[coordinates[:, 0], coordinates[:, 1]]

        filtered_coordinate_idx = image_value_coordinates > np.percentile(image_gauss, percentile)
        filtered_coordinate = coordinates[filtered_coordinate_idx]
    else:
        coordinates = peak_local_max(image, min_distance=min_distance)
        image_value_coordinates = image[coordinates[:, 0], coordinates[:, 1]]

        filtered_coordinate_idx = image_value_coordinates > np.percentile(image, percentile)
        filtered_coordinate = coordinates[filtered_coordinate_idx]

    TP_locations_matrix = filtered_coordinate

    if plot:
        fig3 = plt.figure()
        plt.imshow(image, cmap='gray')
        plt.scatter(filtered_coordinate[:, 1], filtered_coordinate[:, 0], c='r')
        if crop_cluster_cell_wh:
            half_size = crop_cluster_cell_wh // 2
            for coord in filtered_coordinate:
                y, x = coord
                rect = plt.Rectangle((x - half_size, y - half_size), crop_cluster_cell_wh,
                                     crop_cluster_cell_wh, linewidth=1.5, edgecolor='lightblue', facecolor='none')
                plt.gca().add_patch(rect)
        plt.title('TP image with marked 2D localizations')
        display(fig3)

    if save_plot:
        fig2 = plt.figure()
        plt.imshow(image, cmap='gray')
        plt.scatter(filtered_coordinate[:, 1], filtered_coordinate[:, 0], c='r')
        if crop_cluster_cell_wh:
            half_size = crop_cluster_cell_wh // 2
            for coord in filtered_coordinate:
                y, x = coord
                rect = plt.Rectangle((x - half_size, y - half_size), crop_cluster_cell_wh,
                                     crop_cluster_cell_wh, linewidth=1.5, edgecolor='lightblue', facecolor='none')
                plt.gca().add_patch(rect)
        plt.title('TP image with marked 2D localizations')
        fig2.savefig(os.path.join(save_plot_path, str(frame_idx) + '.jpg'))

    return TP_locations_matrix

