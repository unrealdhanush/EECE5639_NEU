import numpy as np
from matplotlib import pyplot as plt
import matplotlib.path as mplPath
import math
import cv2
import os
import gc
import time
import colorsys


def harris(img, win_size=3, sobel_size=3, k=0.04, step_size=1):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = img.shape[0], img.shape[1]

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_size)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_size)

    Ix2 = sobelx ** 2
    Iy2 = sobely ** 2
    Ixy = sobelx * sobely

    R = np.zeros_like(img, dtype=np.float64)

    offset = np.floor(win_size / 2).astype(np.int8)
    for y in range(offset, h-offset, step_size):
        for x in range(offset, w-offset, step_size):
            Sx2 = np.sum(Ix2[y-offset:y+1+offset, x-offset:x+1+offset])
            Sy2 = np.sum(Iy2[y-offset:y+1+offset, x-offset:x+1+offset])
            Sxy = np.sum(Ixy[y-offset:y+1+offset, x-offset:x+1+offset])
            r = (Sx2*Sy2)-(Sxy**2) - k*(Sx2+Sy2)**2
            if r < 0:
                r = 0
            R[y][x] = r

    R /= np.max(R)
    R *= 255.0
    return R


def non_max_suppression(img, window_size=5, thresh=2.5):
    h, w = img.shape[0], img.shape[1]

    offset = np.floor(window_size / 2).astype(np.int8)

    temp = np.zeros_like(img)
    temp[offset:h-offset, offset:w-offset] = img[offset:h-offset, offset:w-offset]
    img = temp
    del temp

    index = np.zeros((h, w, 3), dtype=np.int16)

    index[:, :, 0] = img

    for y in range(h):
        for x in range(w):
            index[y][x][1] = x
            index[y][x][2] = y

    corners = []

    global NMS_index
    NMS_index = np.zeros((h, w))

    for y in range(offset, h - offset, offset):
        for x in range(offset, w - offset, offset):
            ret = mean_shift_conv(index, (x, y), window_size, thresh)
            if ret:
                corners.append(ret)

    corners = list(set(corners))
    corners.sort()
    return corners


def mean_shift_conv(index, point, window_size=5, thresh=2.5):
    x, y = point
    offset = np.floor(window_size / 2).astype(np.int8)

    if NMS_index[y][x] == 1:
        return None

    window = index[y - offset:y + 1 + offset, x - offset:x + 1 + offset]
    if np.max(window[:, :, 0]) > thresh:
        window = np.reshape(window, (window_size ** 2, 3))
        window[::-1] = window[window[:, 0].argsort()]
        if window[0][1] == x and window[0][2] == y:
            NMS_index[y][x] = 1
            return x, y
        else:
            if mean_shift_conv(index, (window[0][1], window[0][2]), window_size, thresh):
                NMS_index[y][x] = 1
                return window[0][1], window[0][2]
            else:
                return None
    else:
        return None


def NCC(img1, img2, p1, p2, window_size=3):
    if len(img1.shape) > 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    if len(img2.shape) > 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    h, w = img1.shape[0], img1.shape[1]

    offset = np.floor(window_size / 2).astype(np.int8)

    (x1, y1) = p1
    (x2, y2) = p2

    if x1 <= offset or x1 >= (w-offset) or y1 <= offset or y1 >= (h-offset) or\
            x2 <= offset or x2 >= (w - offset) or y2 <= offset or y2 >= (h - offset):
        return 0

    window1 = img1[y1-offset:y1+1+offset, x1-offset:x1+1+offset]
    window1 = window1.astype(np.float64)

    window2 = img2[y2-offset:y2+1+offset, x2-offset:x2+1+offset]
    window2 = window2.astype(np.float64)

    mean1 = np.mean(window1)
    std1 = np.std(window1)

    mean2 = np.mean(window2)
    std2 = np.std(window2)

    s = 0
    for i in range(0, window_size):
        for j in range(0, window_size):
            s += (window1[i][j] - mean1) / std1 * (window2[i][j] - mean2) / std2
    s /= (window_size ** 2)

    return s


def remvaluefromlist(the_list, val):
    return [value for value in the_list if value[1] != val]


def NCCfilt(NCClist, len_threshold=0, score_threshold=0):
    NCClist.sort(key=lambda x: x[0], reverse=True)

    if score_threshold > 0:
        NCClist = [NCClist[i] for i in range(len(NCClist)) if NCClist[i][0] > score_threshold]

    if 0 < len_threshold <= len(NCClist):
        NCClist = NCClist[:len_threshold]

    return NCClist


def est_homography(points1, points2, thresh=3, N_max=10000):
    p = 0.9
    e = 0.3
    s = len(points1)
    N = np.min((np.log(1 - p) / (np.log(1 - (1 - e) ** s)), N_max))
    N = np.int32(N)

    inliers = []
    inlier_max = 0
    print("Corner size:", s, "")
    for x in range(0, N):
        four_points1 = []
        four_points2 = []
        c = 0
        while c < 4:

            r = np.random.randint(low=0, high=s)
            if points1[r] not in four_points1 and points2[r] not in four_points2:
                four_points1.append(points1[r])
                four_points2.append(points2[r])
                c += 1

        four_points1 = np.array(four_points1, ndmin=2)
        four_points2 = np.array(four_points2, ndmin=2)

        h, status = cv2.findHomography(four_points2, four_points1)

        if status[0][0] != 0:
            counter = 0
            temp = []
            for i in range(len(points1)):
                x1, y1 = homographypoints(points2[i], h)
                if x1 is None or y1 is None:
                    break

                if points1[i][0]-thresh <= x1 <= points1[i][0]+thresh and points1[i][1]-thresh <= y1 <= points1[i][1]+thresh:
                    counter += 1
                    temp.append((points1[i], points2[i]))

            if counter > inlier_max:
                inliers = temp
                inlier_max = counter

    p1 = np.array([(k[0][0], k[0][1]) for k in inliers])
    p2 = np.array([(k[1][0], k[1][1]) for k in inliers])
    h, _ = cv2.findHomography(p2, p1)
    return h, inliers


def homographypoints(p, H):
    den = H[2][0]*p[0] + H[2][1]*p[1] + 1
    if den == 0.0 or math.isnan(den):
        return None, None
    x = (H[0][0]*p[0] + H[0][1]*p[1] + H[0][2]) / den
    y = (H[1][0]*p[0] + H[1][1]*p[1] + H[1][2]) / den

    if np.fabs(x) > 16000 or np.fabs(y) > 16000:
        return None, None

    return np.array([x, y], dtype=np.int16)


def displaycornerpairs(img1, img2, lines, y1_shift=0, y2_shift=0):
    w1, h1 = img1.shape[1], img1.shape[0]
    w2, h2 = img2.shape[1], img2.shape[0]

    w_max, h_max = np.max((w1, w2)), np.max((h1, h2))

    out1 = np.zeros((h_max, w_max), dtype=np.uint8)
    out1[y1_shift:h1+y1_shift, 0:w1] = img1
    out2 = np.zeros((h_max, w_max), dtype=np.uint8)
    out2[y2_shift:h2+y2_shift, 0:w2] = img2
    combine = np.concatenate((out1, out2), axis=1)

    for p in lines:
        cv2.circle(combine, (p[0][0], p[0][1] + y1_shift), 2, (0, 0, 255), 1)
        cv2.circle(combine, (p[1][0] + w1, p[1][1] + y2_shift), 2, (0, 0, 255), 1)

        hsv = np.array((np.random.rand(), 1, 1))
        rgb = list((np.array(colorsys.hsv_to_rgb(hsv[0], hsv[1], hsv[2])) * 255.0).astype(np.uint16))
        rgb = [int(rgb[0]), int(rgb[1]), int(rgb[2])]
        cv2.line(combine, (p[0][0], p[0][1]+y1_shift), (p[1][0]+w1, p[1][1]+y2_shift), rgb, 1)

    plt.imshow(combine)
    plt.axis('off')
    plt.show()


def main():
    path = "/Users/dhanush/Northeastern/2023spring/eece5639/project3"
    #path = "/Users/dhanush/Northeastern/2023spring/eece5639/project2/DanaOffice"

    img1=cv2.imread('image1.jpeg', 0)
    img2=cv2.imread('image2.jpeg', 0)
    imgset = [img1,img2]

    w, h = imgset[0].shape[1], imgset[0].shape[0]

    imagecurrent = imgset[0]
    x_shift, y_shift = 0, 0

    for img_no in range(1, 2):
        w_cur, h_cur = imagecurrent.shape[1], imagecurrent.shape[0]

        corners = []

        a = time.time()                     # Timing Statements
        ret = harris(imagecurrent, 3, 3, 0.04, step_size=2)
        print("Harris: ", time.time() - a)  # Timing Statements
        a = time.time()                     # Timing Statements
        centroids = non_max_suppression(ret, 15)
        print("NMS: ", time.time() - a)  # Timing Statements
        a = time.time()                     # Timing Statements
        corners.append(centroids)

        ret = harris(imgset[img_no], 3, 3, 0.04, step_size=2)
        print("Harris: ", time.time() - a)  # Timing Statements
        a = time.time()                     # Timing Statements
        centroids = non_max_suppression(ret, 15)
        print("NMS: ", time.time() - a)
        a = time.time()
        corners.append(centroids)

        NCCscore = []
        for i in range(len(corners[0])):
            for j in range(len(corners[1])):
                score = NCC(imgset[0], imgset[1], corners[0][i], corners[1][j], 25)
                NCCscore.append((score, corners[0][i], corners[1][j]))

        NCCscore = NCCfilt(NCCscore, len_threshold=20)
        print("Harris: ", time.time() - a)  # Timing Statements
        a = time.time()                     # Timing Statements

        p1 = [p[1] for p in NCCscore]
        p2 = [p[2] for p in NCCscore]
        H, inliers = est_homography(p1, p2)
        H_I = np.linalg.inv(H)
        print(H)
        print(H_I)

        print("Homography: ", time.time() - a)
        a = time.time()

        p1 = [(p[1], p[2]) for p in NCCscore]
        displaycornerpairs(imagecurrent, imgset[img_no], p1)
        displaycornerpairs(imagecurrent, imgset[img_no], inliers)

        c1 = homographypoints((0, 0), H)
        c2 = homographypoints((w-1, 0), H)
        c3 = homographypoints((0, h-1), H)
        c4 = homographypoints((w-1, h-1), H)
        center_new = homographypoints((w/2, h/2), H)
        w_new = np.max([c4[0]-c1[0], c4[0]-c3[0], c2[0]-c1[0], c2[0]-c3[0]])
        h_new = np.max([c4[1]-c1[1], c4[1]-c2[1], c3[1]-c1[1], c3[1]-c2[1]])
        x_offset = np.min([c1[0], c3[0]])
        y_offset = np.min([c1[1], c2[1]])
        if y_offset < 0:
            y_shift = -y_offset
        else:
            y_shift = 0

        if x_offset < 0:
            x_shift = -x_offset
        else:
            x_shift = 0

        O = np.array([[1, 0, x_shift],
                        [0, 1, y_shift],
                        [0, 0, 1]])

        output = cv2.warpPerspective(imgset[img_no], O @ H, (max(w_new+x_offset, w_cur), max(h_new, h_cur)))

        output[y_shift: h_cur+y_shift, x_shift: w_cur+x_shift] = imagecurrent

        for i in range(h):
            for j in range(w):
                p2 = homographypoints((j, i), H_I)
                if 0 <= p2[0] < w and 0 <= p2[1] < h:
                    output[p2[1]+y_shift][p2[0]+x_shift] = (255, 255, 255)


        imagecurrent = output

        plt.figure()
        plt.imshow(output)
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    main()