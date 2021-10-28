import sys
import math
import numpy as np
from PIL import Image
import imageio
from pprint import pprint
from skimage.draw import line_aa
import scipy.misc



"""
Input: 2D numpy array, std value for gaussian distribution
Output: new 2D numpy array which has been convoluted using a gaussian distribution
"""
def gaussian(image, sigma=1):
    filtersize = (2*3*sigma) + 1
    kernel = np.zeros((filtersize, filtersize))
    
    for x in range(kernel.shape[0]):
        for y in range(kernel.shape[1]):
            kernel[x,y] = 1/(2*math.pi*(sigma**2))*math.exp(-1*((x-(filtersize-1)/2)**2 + (y-(filtersize-1)/2)**2)/(2*(sigma**2)))

    # scaling guas kernel so sum = 1
    sum_gaus = np.sum(kernel)
    scale_factor = 1/sum_gaus
    kernel = kernel * scale_factor

    new_arr = np.empty_like(image)
    image = np.pad(image, ((3*sigma,3*sigma),(3*sigma,3*sigma)), 'edge')
    
    for i in range(new_arr.shape[0]):
        for j in range(new_arr.shape[1]):
            new_arr[i][j] = np.sum(image[i:i+filtersize, j:j+filtersize]*kernel)

    return new_arr

"""
Input: 2D numpy array and an upper threshhold
Output: 2 new 2D numpy arrays which have been convoluted using the vertical and horizontal sobel filters
"""
def harris(image, threshold):

    # Filters
    horizontal = np.array([[1,2,1], [0,0,0], [-1,-2,-1]])
    vertical = np.array([[1,0,-1], [2,0,-2], [1,0,-1]])

    # Gaus with sigma = 1 -> 7x7 filter
    filtersize = (2*3*1) + 1
    kernel = np.zeros((filtersize, filtersize))
    
    for x in range(kernel.shape[0]):
        for y in range(kernel.shape[1]):
            kernel[x,y] = 1/(2*math.pi*(sigma**2))*math.exp(-1*((x-(filtersize-1)/2)**2 + (y-(filtersize-1)/2)**2)/(2*(sigma**2)))

    # scaling guas kernel so sum = 1
    sum_gaus = np.sum(kernel)
    scale_factor = 1/sum_gaus
    kernel = kernel * scale_factor

    n = image.shape[0]
    m = image.shape[1]

    dx = np.zeros((n, m))
    dy = np.zeros((n, m))
    r_matrix =  np.zeros((n, m)) 

    image = np.pad(image, ((1,1),(1,1)), 'constant')

    # nxm matrix where each [i][j] is a 2x2 matrix containing the values of the harris matrix for that point
    for i in range(n):
        for j in range(m):
            dx[i][j] = np.sum(np.multiply(image[i:i+3, j:j+3], horizontal))
            dy[i][j] = np.sum(np.multiply(image[i:i+3, j:j+3], vertical))

    dx = np.pad(dx, ((3,3),(3,3)), 'constant')
    dy = np.pad(dy, ((3,3),(3,3)), 'constant')

    for i in range(n):
        for j in range(m):
            dx_window = dx[i:i+7, j:j+7] * kernel
            dy_window = dy[i:i+7, j:j+7] * kernel

            # 2x2 matrix: [[dx2, dxdy], [dxdy, dy2]]
            local_harris = np.array([[np.sum(dx_window*dx_window), np.sum(dx_window*dy_window)], [np.sum(dx_window*dy_window), np.sum(dy_window*dy_window)]])
            r_matrix[i][j] = np.linalg.det(local_harris) - .05*(np.trace(local_harris))**2

    r_matrix[r_matrix < threshold] = 0
    return r_matrix

"""
Input: np arr of determanant hessian matrix
Output: new np array NMS
"""
def NonMaxSupression(det_hess):
    n,m = det_hess.shape

    new_arr = np.zeros((n, m))

    det_hess = np.pad(det_hess, ((1,1),(1,1)), 'constant')

    # Finding local maximums
    for i in range(n):
        for j in range(m):
            if(det_hess[i+1][j+1] >= np.max(det_hess[i:i+3, j:j+3])):
                new_arr[i][j] = det_hess[i+1][j+1]

    return new_arr
"""
Input: np array, int n
Output: Array containing values that are greater than the nth largest value of the array
"""
def topNPoints(arr, n):
    nthVal = np.partition(arr.flatten(), -n)[-n]
    arr[arr < nthVal] = 0
    return arr

"""
Input: Two arrays
Output: list of [(coord1), (coord2), score)] representing best SSD matches between the two arrays 
"""
def SSD(arr1, arr2):
    print(arr1.shape)
    print(arr2.shape)

    # non zero elements in arr1
    non_zero1 = np.nonzero(arr1)
    non_zero2 = np.nonzero(arr2)

    # works at 1
    arr1 = np.pad(arr1, ((2,2),(2,2)), 'constant')
    arr2 = np.pad(arr2, ((2,2),(2,2)), 'constant')


    # List of tuples that represent coordinates of non zero elements
    tuples1 = list(zip(non_zero1[0], non_zero1[1]))
    tuples2 = list(zip(non_zero2[0], non_zero2[1]))

    # list of [(coord1), (coord2), score)]
    best_matches= []

    # works with + 2
    # iterate over each feature in arr1, and compare it to all the features in arr2
    for tuple1 in tuples1:
        # 5x5 window around feature point
        window1 = arr1[tuple1[0]:tuple1[0]+5, tuple1[1]: tuple1[1]+5]

        lowest_score = math.inf
        best_point = (0,0)

        for tuple2 in tuples2:
            window2 = arr2[tuple2[0]:tuple2[0]+5, tuple2[1]: tuple2[1]+5]

            if np.sum(np.square(window1-window2)) < lowest_score:
                lowest_score = np.sum(np.square(window1-window2))
                best_point = tuple2
        
        best_matches.append([tuple1, best_point, lowest_score])
    
    return best_matches

"""
Input: list of [(coord1), (coord2), score)]
arr1 (nxm)
arr2 (pxk)
Output: Imag
"""
def graph_matches(matches, arr1, arr2):
    # sort matches to find top 20
    # pprint(matches)
    matches.sort(key = lambda x: x[2])
    matches = matches[:20]

    concat_arr = np.concatenate((arr1, arr2), axis=1)
    # concat_arr = np.pad(concat_arr, ((1,1),(1,1)), 'edge')

    print(concat_arr.shape)

    for match in matches:
        p1 = match[0]
        p2 = match[1]

        rr, cc, val = line_aa(p1[0], p1[1], p2[0], p2[1] + arr1.shape[1])
        concat_arr[rr, cc] = val * 255

    imageio.imwrite('concat.pgm', concat_arr)



"""
Input: Filename/path in current directory
Output: np array of image
Action: Converts image to greyscale np array
"""
def openImage(filename):
    image = Image.open(filename).convert('L')
    np_image = np.asarray(image)
    return np_image

def Q1(image1, image2):
    F1_guausian = gaussian(image1, sigma)
    F2_gaussian = gaussian(image2, sigma) 

    imageio.imwrite('Gaus_F1.pgm', F1_guausian)
    imageio.imwrite('Gaus_F2.pgm', F2_gaussian)

    F1_harris = harris(F1_guausian, threshold)
    F2_harris = harris(F2_gaussian, threshold)

    imageio.imwrite('Harris_F1.pgm', F1_harris)
    imageio.imwrite('Harris_F2.pgm', F2_harris)

    F1_topN = topNPoints(F1_harris, 1000)
    F2_topN = topNPoints(F2_harris, 1000)

    imageio.imwrite('Topn_F1.pgm', F1_topN)
    imageio.imwrite('Topn_F2.pgm', F2_topN)

    F1_nms = NonMaxSupression(F1_topN)
    F2_nms = NonMaxSupression(F2_topN)

    imageio.imwrite('nms_F1.pgm', F1_nms)
    imageio.imwrite('nms_F2.pgm', F2_nms)

    matches = SSD(F1_nms, F2_nms)
    graph_matches(matches, image1, image2)
    


"""
Arg1: hw3.py
Arg2: Filename1
Arg3: Filename2
Arg4: Sigma for gaussian
Arg5: threshold for harris dector
"""
if __name__ == "__main__":
    filename1 = sys.argv[1]
    filename2 = sys.argv[2]
    sigma = int(sys.argv[3])
    threshold = int(sys.argv[4])

    image1 = openImage(filename1)
    image2 = openImage(filename2)

    # Q1(image1, image2)
    Q1(image1, image2.rotate(45))