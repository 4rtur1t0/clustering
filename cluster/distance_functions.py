"""
functions to compute distances between datapoints

"""
import numpy as np

# _nbits[k] is the number of 1s in the binary representation of k for 0 <= k < 256.
_nbits = np.array(
      [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3,
       4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4,
       4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2,
       3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5,
       4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4,
       5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3,
       3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2,
       3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6,
       4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5,
       6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5,
       5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6,
       7, 7, 8], dtype=np.uint8)


def hamming(a, b):
    """
    Compute a hamming distance between descriptors a and b
    Use a bitwise or on the whole array of bytes.
    Next, _nbits is a precomputed array that stores the number of bits on each 1-byte result.

    Example:
        The number 13 is represented by 00001101.
        Likewise, 17 is represented by 00010001.
        The bit-wise XOR of 13 and 17 is therefore 00011100, or 28:

        >>> np.bitwise_xor(13, 17)
            28
        >>> np.binary_repr(28)
            '11100'

    Using bitwise_xor to compoute the operation. For example, given two arrays
    >>> np.bitwise_xor([31, 3, 67, 78], [5, 6, 90, 255])
    array([ 26,   5,  25, 177])

    :param a: a binary descriptor.
    :param b:
    :return:
    """
    # a = np.uint8(a)
    # b = np.uint8(b)
    c = np.bitwise_xor(a, b)
    n = _nbits[c].sum()
    return n


def find_closest(data, centroid_k, distance_function='euclidean'):
    """
        Find the closest sample of data to the centroid
        :param data:
        :return:
    """
    # distances of all datapoints to centroid k
    d = compute_distance_function(data, centroid_k, distance_function=distance_function)
    k_i = np.where(d == np.amin(d))
    index = k_i[0][0]
    return data[index, :]


def compute_distance_function(data, centroid, distance_function='euclidean'):
    """
        Compute distance between all data and a given centroid.
        In the euclidean distance, np.linalg.norm allows to compute the distance of all data to a centroid by computing
        a difference and then computing the L2-norm.
        In the case of the Hamming distance, the np.bitwise_xor has to be called for each descriptor independently, thus
        slowing down
    """
    if distance_function == 'euclidean':
        # returns an array of 1 x N, where N is the number of datapoints in data
        d = np.linalg.norm(data - centroid, axis=1)
        return d
    elif distance_function == 'hamming':
        # iterate along samples to compute the hamming distance
        d = compute_hamming_distances(data, centroid)
        return d
    else:
        print('PLEASE INDICATE EITHER euclidean or hamming DISTANCE FUNCTIONS')
        raise Exception


def compute_hamming_distances(data, centroid):
    distances = -1 * np.ones(len(data), dtype=np.uint8)
    try:
        assert (data.dtype == np.uint8)
        assert (centroid.dtype == np.uint8)
    except AssertionError:
        print("Please use binary data (numpy.uint8) if Hamming distance is used.")
        print("Alternatively, use mean-round as averaging method is used.")
        print("Exiting")
        exit()
    # compute hamming distance for all descriptors in the dataset to the given centroid
    for i in range(len(data)):
        sample = data[i]
        distances[i] = hamming(sample, centroid)
    return distances





