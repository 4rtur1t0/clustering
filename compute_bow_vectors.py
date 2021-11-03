"""
Load the vocabulary
Load the descriptors of each image.
Project each descriptor to a word.

compute bag of words vectors.

SAVE A .JSON file with all descriptors
"""
import json
import pickle
from config.experiment_config import EXPERIMENT_CONFIG
from cluster.load_data import load_ORB_data

process_path = 'all_paths'
vocabulary_file = 'voc_kw2lw4.pkl'
vocabulary_path = 'voc/' + vocabulary_file
cameras = ['cam0', 'cam1']
paths = EXPERIMENT_CONFIG.configurations.get(process_path)


def load_vocabulary():
    filehandler = open(vocabulary_path, 'rb')
    vocabulary = pickle.load(filehandler)
    print(vocabulary)
    return vocabulary


def save_bowvector_to_file(path, cam, tf_idf, name):
    bowvector_file = path + '/mav0/' + cam + '/' + name
    #tf_idf = tf_idf.to_list()
    json_serializable = []
    for tf_idf_i in tf_idf:
        json_serializable.append(tf_idf_i.tolist())
    with open(bowvector_file, 'w') as outfile:
        # json.dump({'bow_vectors': json_serializable}, outfile)
        json.dump(json_serializable, outfile)


if __name__ == '__main__':
    vocabulary = load_vocabulary()
    # idf is already precomputed for the vocabulary database.
    idf = vocabulary.idf

    print('COMPUTING BOW vectors')
    print('Loading descriptors from: ', process_path)
    for path in paths:
        for cam in cameras:
            print('PROCESSING PATH: ', path)
            print('LOADING DESCRIPTORS FOR: ', cam)

            # load ORB descriptors or compute them from images
            # here, a list with a single path is passed
            X_im = load_ORB_data(paths=[path], mode='single', camera=cam)
            print('FOUND: ', len(X_im), ' images')
            print('COMPUTING TF-IDF FOR EACH IMAGE: ')
            # compute tf
            tf = vocabulary.compute_tf(X_im)
            tf_idf = vocabulary.compute_tf_idf(tf, idf)
            # save tf_idf
            save_bowvector_to_file(path, cam, tf_idf, 'tf_idf_bowvectors' + vocabulary_file + '.json')
            # save only tf
            save_bowvector_to_file(path, cam, tf, 'tf_bowvectors' + vocabulary_file + '.json')








        # X_im = load_ORB_data(paths=[path], mode='single', camera='cam1')
        # print('FOUND: ', len(X_im), ' images')
        # print('COMPUTING TF-IDF FOR EACH IMAGE: ')
        #
        # # compute tf
        # tf = vocabulary.compute_tf(X_im)
        # tf_idf = vocabulary.compute_tf_idf(tf, idf)
        # #save_to_file(cam0)
        #
        # # TODO: save tf and tf_idf for current path
        # # TODO: normalize tf_idf to one



    # # Optional, y using brute-force project to vocabulary
    # # vocabulary.build_leaf_word_list()
    # # vocabulary.build_idf()
    # # two options, either process image, get descriptors, project to BOW and plot WORDS on image
    #
    # # or load jsons from the descriptors files
    # # in this case, consider that we have an experiment with 78 images: experiment descriptors will be a list of 78
    # # elements, each element contains a list of 2000 descriptors, which correspond to the number of descriptors
    # # extracted at each image.
    # for path in paths:
    #     # TODO: at this point, experiment descriptors of the front image and back image should be fused together
    #     experiment_descriptors = []
    #     # by extending experiment
    #     [experiment_descriptors_front, experiment_keypoints_front] = load_descriptors(path + '/mav0/cam0/')
    #     [experiment_descriptors_back, experiment_keypoints_back] = load_descriptors(path + '/mav0/cam1/')
    #
    #     # for image_descriptors in experiment_descriptors:
    #     # loop over all images captured
    #     for i in range(0, len(experiment_descriptors_front)):
    #         # append both descriptors for both cameras (front and back)  now concat descriptors from front and back
    #         image_descriptors = []
    #         image_descriptors.extend(experiment_descriptors_front[i])
    #         image_descriptors.extend(experiment_descriptors_back[i])
    #         image_descriptors = np.array(image_descriptors, np.uint8)
    #         image_descriptors = np.unpackbits(image_descriptors, axis=1)
    #         # return the words and number of words at each image
    #         bow = vocabulary.project_to_vocabulary(image_descriptors)
    #
    #         # transform each bow to histogram
    #         # - 1 histogram of numbers.Number
    #         # - 2 histogram of     (probability
    #         # - 3 tf- idf
    #         # - 4
    #
    #
    #     # vocabulary.compute_bow()
    #     # [descriptors, keypoints] = load_descriptors(path + '/mav0/cam1/')

