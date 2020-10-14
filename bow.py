# author: zhangyzh
# created time: 2020/9/29
# last modifation: 2020/10/15
# version: Python 3.6.4
# -------------------------------
# this is computer version I (CS 172) homework 2
# Bag of visual words based image + SVM &
# Spatial Pyramid Matching based image + SVM
# reference: 
# 1. David G. Lowe, "Distinctive image features from scale-invariant keypoints," International Journal of Computer Vision, 60, 2 (2004), pp. 91-110
# 2. S. Lazebnik, C. Schmid, and J. Ponce, “Beyond bags of features: Spatial pyramid matching for recognizing natural scene categories,” in Computer Vision and Pattern Recognition, 2006 IEEE Computer Society Conference on, vol. 2. IEEE, 2006, pp. 2169–2178.
# 3. Y. Rubner, C. Tomasi, and L. J. Guibas, “The earth mover’s distance as a metric for image retrieval,” International Journal of Computer Vision, vol. 40, no. 2, pp. 99–121, 2000.
# 4. P. Li, J. Ma, and S. Gao, “Actions in still web images: Visualization, detec- tion and retrieval,” in Web-Age Information Management. Springer, 2011, pp. 302–313.

import os
import cv2 as cv
import numpy as np
import time

from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# get the training number for each class
train_num = int(input('Give the training number for each class (15/30/45/60):'))

# the training path
train_path = '..\\256_ObjectCategories'
train_classes = os.listdir(train_path)

pictures = []
classes = []
class_id = 0
for i in train_classes:
    # Absolute path
    abs_path = os.path.join(train_path, i)
    # get a list of picture names
    folder = [os.path.join(abs_path, f) for f in os.listdir(abs_path)]
    # save train_num of them
    pictures += folder[:train_num]
    classes += [class_id]*train_num
    class_id+=1

# apply sift
sift = cv.xfeatures2d.SIFT_create()

# get features
cur_time = time.time()
print('get features!')
features = []
for pic in pictures:
    print('read pic: {}!'.format(pic))
    img = cv.imread(pic, 0)
    # sift.detectAndCompute() return kps, feature
    # kps includes {angle, class_id, octave, pt, response, size}
    _, ft = sift.detectAndCompute(img, None)
    if ft is None:
        continue
    features.append((pic, ft))

pic_num = len(features)
# statistics and build a Histogram
print('get features!')
a = (ft for _, ft in features)
Histogram = np.vstack(a)

# K-means
print('k-means start!')
centroid = kmeans(Histogram, 100, 1)[0]

# Calculate the histogram of features
print('Calculate the histogram of features!')
img_ft = np.zeros((pic_num, 100), "float32")
for i in range(pic_num):
    words = vq(features[i][1], centroid)[0]
    print('get word of picture: {}!'.format(pictures[i]))
    for word in words:
        img_ft[i][word] += 1

print('Tf-Idf')
# Perform Tf-Idf vectorization
occurcy = np.sum((img_ft > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*pic_num+1) / (1.0*occurcy + 1)), 'float32')

# Scaling the words
print('Scaling')
scale = StandardScaler().fit(img_ft)
img_ft = scale.transform(img_ft)

# Train the Linear SVM
print('SVM')
clf = LinearSVC()
clf.fit(img_ft, np.array(classes))

# Save the SVM
print('save')
joblib.dump((clf, train_classes, scale, 100, centroid), "bow{}.pkl".format(train_num), compress=3)    

print('finished in {}'.format(int(time.time()-cur_time)))
