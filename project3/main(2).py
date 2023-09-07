import cv2 
import numpy as np
import matplotlib as plt

MAX_FEATURES=500
GOOD_MATCH_PERCENT=0.15
#Input images
img1=cv2.imread('image1.jpeg',cv2.IMREAD_GRAYSCALE)
img2=cv2.imread('image2.jpeg',cv2.IMREAD_GRAYSCALE)

sift=cv2.SIFT_create(MAX_FEATURES)
kp1,des1=sift.detectAndCompute(img1, None)
kp2,des2=sift.detectAndCompute(img2, None)

matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE)
matches = list(matcher.match(des1, des2, None))

#sorting the matches based on match score 
matches.sort(key=lambda x: x.distance, reverse=False)
 
# Remove not so good matches
numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
matches = matches[:numGoodMatches]

# Draw top matches
imgMatches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None)
plt.imshow(imgMatches)

# Extract location of good matches
points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    points1[i, :] = kp1[match.queryIdx].pt
    points2[i, :] = kp2[match.trainIdx].pt

# Find homography
h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

# Use homography
height, width = img2.shape
im1Reg = cv2.warpPerspective(img1, h, (width, height))

