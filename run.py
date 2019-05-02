import cv2
import imagehash
import numpy as np
from os import listdir
from pylab import *
from PIL import Image
from scipy.cluster.vq import *
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn import tree, svm, linear_model

genuine_images_path = "data/genuine"
forged_images_path = "data/forged"

genuine_image_filenames = listdir(genuine_images_path)
forged_image_filenames = listdir(forged_images_path)

genuine_image_features = [[] for x in range(12)]
forged_image_features = [[] for x in range(12)]

for name in genuine_image_filenames:
    signature_id = int(name.split('_')[0][-3:])
    genuine_image_features[signature_id - 1].append({"name": name})

for name in forged_image_filenames:
    signature_id = int(name.split('_')[0][-3:])
    forged_image_features[signature_id - 1].append({"name": name})


def preprocess_image(image_path, display=False):
    """
    1) Convert image to gray scale
    2) Threshold image

    :param image_path: image path
    :param display: flag - if true display images
    :return: preprocessed image
    """
    raw_image = cv2.imread(image_path)
    bw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    bw_image = 255 - bw_image

    if display:
        cv2.imshow("RGB to Gray", bw_image)
        cv2.waitKey()

    _, threshold_image = cv2.threshold(bw_image, 30, 255, 0)

    if display:
        cv2.imshow("Threshold", threshold_image)
        cv2.waitKey()

    return threshold_image


def get_contour_features(preprocessed_image, display=False):
    """
    :param preprocessed_image: preprocessed image
    :param display: flag - if true display images
    :return: aspect ratio of bounding rectangle, area of bounding rectangle, contours and convex hull
    """

    rect = cv2.minAreaRect(cv2.findNonZero(preprocessed_image))
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    w = np.linalg.norm(box[0] - box[1])
    h = np.linalg.norm(box[1] - box[2])

    aspect_ratio = max(w, h) / min(w, h)
    bounding_rect_area = w * h

    if display:
        image1 = cv2.drawContours(preprocessed_image.copy(), [box], 0, (120, 120, 120), 2)
        cv2.imshow("a", cv2.resize(image1, (0, 0), fx=2.5, fy=2.5))
        cv2.waitKey()

    hull = cv2.convexHull(cv2.findNonZero(preprocessed_image))

    if display:
        convex_hull_image = cv2.drawContours(preprocessed_image.copy(), [hull], 0, (120, 120, 120), 2)
        cv2.imshow("a", cv2.resize(convex_hull_image, (0, 0), fx=2.5, fy=2.5))
        cv2.waitKey()

    _, contours, hierarchy = cv2.findContours(preprocessed_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if display:
        contour_image = cv2.drawContours(preprocessed_image.copy(), contours, -1, (120, 120, 120), 3)
        cv2.imshow("a", cv2.resize(contour_image, (0, 0), fx=2.5, fy=2.5))
        cv2.waitKey()

    contour_area = 0
    for cnt in contours:
        contour_area += cv2.contourArea(cnt)
    hull_area = cv2.contourArea(hull)

    return aspect_ratio, bounding_rect_area, hull_area, contour_area


des_list = []


def sift(preprocessed_image, image_path, display=False):
    raw_image = cv2.imread(image_path)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(preprocessed_image, None)

    if display:
        cv2.drawKeypoints(preprocessed_image, kp, raw_image)
        cv2.imshow('sift_keypoints.jpg', cv2.resize(raw_image, (0, 0), fx=3, fy=3))
        cv2.waitKey()

    return (image_path, des)


true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

im_contour_features = []

for i in range(12):
    des_list = []
    for im in genuine_image_features[i]:
        image_path = genuine_images_path + "/" + im['name']
        preprocessed_image = preprocess_image(image_path)
        hash = imagehash.phash(Image.open(image_path))

        aspect_ratio, bounding_rect_area, convex_hull_area, contours_area = \
            get_contour_features(preprocessed_image.copy(), display=False)

        hash = int(str(hash), 16)
        im['hash'] = hash
        im['aspect_ratio'] = aspect_ratio
        im['hull_area/bounding_area'] = convex_hull_area / bounding_rect_area
        im['contour_area/bounding_area'] = contours_area / bounding_rect_area

        im_contour_features.append(
            [hash, aspect_ratio, convex_hull_area / bounding_rect_area, contours_area / bounding_rect_area])

        des_list.append(sift(preprocessed_image, image_path))

    for im in forged_image_features[i]:
        image_path = forged_images_path + "/" + im['name']
        preprocessed_image = preprocess_image(image_path)
        hash = imagehash.phash(Image.open(image_path))

        aspect_ratio, bounding_rect_area, convex_hull_area, contours_area = \
            get_contour_features(preprocessed_image.copy(), display=False)

        hash = int(str(hash), 16)
        im['hash'] = hash
        im['aspect_ratio'] = aspect_ratio
        im['hull_area/bounding_area'] = convex_hull_area / bounding_rect_area
        im['contour_area/bounding_area'] = contours_area / bounding_rect_area

        im_contour_features.append(
            [hash, aspect_ratio, convex_hull_area / bounding_rect_area, contours_area / bounding_rect_area])

        des_list.append(sift(preprocessed_image, image_path))

    descriptors = des_list[0][1]
    for image_path, descriptor in des_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))
    k = 500
    voc, variance = kmeans(descriptors, k, 1)

    # Calculate the histogram of features
    im_features = np.zeros((len(genuine_image_features[i]) + len(forged_image_features[i]), k + 4), "float32")
    for i in range(len(genuine_image_features[i]) + len(forged_image_features[i])):
        words, distance = vq(des_list[i][1], voc)
        for w in words:
            im_features[i][w] += 1

        for j in range(4):
            im_features[i][k + j] = im_contour_features[i][j]

    # nbr_occurences = np.sum((im_features > 0) * 1, axis=0)
    # idf = np.array(np.log((1.0 * len(image_paths) + 1) / (1.0 * nbr_occurences + 1)), 'float32')

    # Scaling the words
    stdSlr = StandardScaler().fit(im_features)
    im_features = stdSlr.transform(im_features)

    train_genuine_features, test_genuine_features = im_features[0:3], im_features[3:5]

    train_forged_features, test_forged_features = im_features[5:8], im_features[8:10]

    # clf = linear_model.LogisticRegression(C=1e5)

    clf = LinearSVC()
    # clf = tree.DecisionTreeClassifier()
    # clf = tree.DecisionTreeRegressor()
    # clf = svm.SVC()

    clf.fit(np.concatenate((train_forged_features, train_genuine_features)),
            np.array([1 for x in range(len(train_forged_features))] + [2 for x in range(len(train_genuine_features))]))

    genuine_res = clf.predict(test_genuine_features)

    for res in genuine_res:
        if int(res) == 2:
            true_positive += 1
        else:
            false_negative += 1

    forged_res = clf.predict(test_forged_features)

    for res in forged_res:
        if int(res) == 1:
            true_negative += 1
        else:
            false_positive += 1

accuracy = float(true_positive + true_negative) / (true_positive + true_negative + false_negative + false_positive)
precision = float(true_positive) / (true_positive + false_positive)
recall = float(true_positive) / (true_positive + false_negative)
f1_score = float(2 * precision * recall) / (precision + recall)

print("Accuracy: ", round(accuracy, 2))
print("Precision: ", round(precision, 2))
print("Recall: ", round(recall, 2))
print("F1 score: ", round(f1_score, 2))
