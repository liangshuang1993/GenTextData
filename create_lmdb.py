# -*- coding: utf-8 -*-
import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np
import glob

images_folder='/home/echo/projects/text_detection/gen_dataset/datasets/train/'

path_output='/home/echo/projects/text_detection/crnn-master/lmdb_data'


os.chdir(images_folder)
all_images = glob.glob("*.jpg")
np.random.shuffle(all_images)

images_train = []
images_val = []

for i in range(int(len(all_images) * 0.9)):
    images_train.append(all_images[i])

for i in range(int(len(all_images) * 0.9), len(all_images)):
    images_val.append(all_images[i])

# left_train,labels_train,right_train = list(zip(*[os.path.splitext(x)[0].split('_')
#                                          for x in images_train]))

# print len(images_train)
# print len(images_val)

labels_train = []
labels_val = []

for image in images_train:
    labels_train.append(image.split('_')[0])

for image in images_val:
    labels_val.append(image.split('_')[0])


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(outputPath,images_train, labels_train, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert(len(images_train) == len(labels_train))
    nSamples = len(images_train)

    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = images_train[i]
        label = labels_train[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        print label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    createDataset(path_output + 'train',images_train, labels_train)
    createDataset(path_output + 'val',images_val, labels_val)
