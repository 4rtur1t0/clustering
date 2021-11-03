"""
For each of the given paths, compute descriptors for cam0 (front camera) and cam1 (back camera).
Save descriptors in a json format.

Author: Arturo Gil
Insitution: Universidad Miguel Hernandez de Elche
Date: July 2021

"""
import cv2
import numpy as np
import pandas as pd
import json
from config import EXPERIMENT_CONFIG

# load the images to be computed to extract descriptors from filepath in config/experiments.yml
# compute descriptors for all paths
paths = EXPERIMENT_CONFIG.configurations.get('all_paths')
VIEW_IMAGES = True
# Initial scaling of images
IMAGES_SCALING = 50
# number of ORB features to extract
NFEATURES = 2000


def resize_image(img, scale_percent=50):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def convert_to_list(keypoint_list):
    ret_list = []
    for kp in keypoint_list:
        ret_list.append({'x': kp.pt[0],
                         'y': kp.pt[1],
                         'octave': kp.octave,
                         'angle': kp.angle})
    return ret_list


def compute_descriptors(path, camera):
    """
    Read images at data.csv and save descriptors in a json file.
    Currently computing ORB descriptors. Feel free to change the detector.
    :param path: The path to the images. Camera can be cam0 or cam1
    :return:
    """
    descriptors_list = []
    keypoints_list = []
    csv_path = path + '/mav0/' + camera + '/data.csv'
    images_path = path + '/mav0/' + camera + '/data/'
    images = pd.read_csv(csv_path)
    print("Computing ORB on camera: ", camera)
    orb = cv2.ORB_create(nfeatures=NFEATURES)

    cv2.namedWindow("Detector")
    i = 0
    for image in images['filenames']:
        i += 1
        # print("Computing ORB on camera: ", camera)
        print('Processing image: ', images_path + str(image))
        print('Processed percent: ', 100*i/len(images), '%')
        input_image = cv2.imread(images_path + str(image))
        input_image = resize_image(input_image, scale_percent=IMAGES_SCALING)
        kp, des = orb.detectAndCompute(input_image, None)
        print("Number of keypoints found: ", len(kp))
        if VIEW_IMAGES:
            responses = []
            for p in kp:
                responses.append(p.response)
            mean_value = np.mean(responses)
            kpp = []
            for p in kp:
                if p.response < mean_value:
                    kpp.append(p)
            out_image = cv2.drawKeypoints(input_image, kpp, None, color=(255, 0, 0))
            cv2.imshow("Output1", out_image)
            cv2.waitKey(0)
        descriptors_list.append(des.tolist())
        keypoints_list.append(convert_to_list(kp))
    return descriptors_list, keypoints_list


# def save_to_json(path, descriptors_front, descriptors_back, keypoints_front, keypoints_back):
#     descriptors_out_path = path + '/mav0/descriptors.json'
#     print('Saving: ', descriptors_out_path)
#     data = {'descriptors_front': descriptors_front,
#             'descriptors_back': descriptors_back,
#             'keypoints_front': keypoints_front,
#             'keypoints_back': keypoints_back,
#             'feature_detector': 'ORB'}
#     with open(descriptors_out_path, 'w') as outfile:
#         json.dump(data, outfile)

def save_to_json(path, descriptors, keypoints):
    descriptors_out_path = path + 'descriptors.json'
    print('Saving: ', descriptors_out_path)
    data = {'descriptors': descriptors,
            'keypoints': keypoints,
            'path': descriptors_out_path,
            'feature_detector': 'ORB'}
    with open(descriptors_out_path, 'w') as outfile:
        json.dump(data, outfile)


if __name__ == '__main__':
    for path in paths:
        [descriptors, keypoints] = compute_descriptors(path, 'cam0')
        save_to_json(path + '/mav0/cam0/', descriptors, keypoints)
        [descriptors_back, keypoints_back] = compute_descriptors(path, 'cam1')
        save_to_json(path + '/mav0/cam1/', descriptors, keypoints)

