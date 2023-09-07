import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread, imshow
from skimage.feature import match_descriptors, SIFT, plot_matches
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform

img1 = imread('image1.jpeg', as_gray=True)  # queryImage
img2 = imread('image2.jpeg', as_gray=True)  # trainImage

# Compute SIFT keypoints and descriptors
desc_extractor = SIFT()
desc_extractor.detect_and_extract(img1)
kp1 = desc_extractor.keypoints
des1 = desc_extractor.descriptors
desc_extractor.detect_and_extract(img2)
kp2 = desc_extractor.keypoints
des2 = desc_extractor.descriptors

matches=match_descriptors(img1, img2, cross_check=True)

fig, ax = plt.subplots(nrows=2, ncols=1)

#RESULTS OF PART1
plt.gray()
plot_matches(ax[0], img1, img2, kp1, kp2, matches)
ax[0].axis('off')
ax[0].set_title("Matches from SIFT")

#RANSAC
random_seed = 9
rng = np.random.default_rng(random_seed)
model, inliers = ransac((kp1[matches[:, 0]],
                         kp2[matches[:, 1]]),
                        FundamentalMatrixTransform, min_samples=8,
                        residual_threshold=1, max_trials=5000,
                        random_state=rng)

inlier_kp1 = kp1[matches[inliers, 0]]
inlier_kp2 = kp2[matches[inliers, 1]]

print(f'Number of inliers: {inliers.sum()}')
fig, ax = plt.subplots(nrows=2, ncols=1)
plt.gray()
plot_matches(ax[0], img1, img2, kp1, kp2, matches[inliers], only_matches=True)
ax[0].axis("off")
ax[0].set_title("Inlier correspondences")
plt.show()