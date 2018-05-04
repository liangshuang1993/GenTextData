# -*- coding: utf-8 -*-
import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np
import glob


path_output='/datasets/GenTextData/lmdb_data_bg/'
TRAIN_DIR1 = '/datasets/GenTextData/'
TRAIN_DIR2 = '/datasets/GenTextData.old/'

VAL_DIR = '/datasets/GenTextData/val_original'

def read_txt(image_dir, image_txt, original_images, original_labels):
    with open(image_txt, 'r') as f:
        lines = f.readlines()

    for line in lines:
        # print line.strip().decode('utf-8').split(' ')
        image, label = line.strip().decode('utf-8').split(' ')
        # print label
        original_images.append(os.path.join(image_dir, image))
        original_labels.append(label)

    return original_images, original_labels

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


def createDataset(outputPath, images_train, labels_train, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    print(len(images_train), len(labels_train))
    assert(len(images_train) == len(labels_train))
    nSamples = len(images_train)

    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    random_index = range(nSamples)
    np.random.shuffle(random_index)
    for i in random_index:
        imagePath = images_train[i]
        label = labels_train[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode('utf-8')
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 100000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    # read first train.txt
    images_train1 = []
    labels_train1 = []
    images_train1, labels_train1 = read_txt(TRAIN_DIR1, '/datasets/GenTextData/datasets_a/train.txt', [], [])
    images_train2, labels_train2 = read_txt(TRAIN_DIR1, '/datasets/GenTextData/datasets_b/train.txt', images_train1, labels_train1)
    images_train3, labels_train3 = read_txt(TRAIN_DIR1, '/datasets/GenTextData/datasets_c/train.txt', images_train2, labels_train2)
    images_train4, labels_train4 = read_txt(TRAIN_DIR1, '/datasets/GenTextData/datasets_d/train.txt', images_train3, labels_train3)
    images_train5, labels_train5 = read_txt(TRAIN_DIR1, '/datasets/GenTextData/datasets_e/train.txt', images_train4, labels_train4)
    images_train6, labels_train6 = read_txt(TRAIN_DIR1, '/datasets/GenTextData/datasets_f/train.txt', images_train5, labels_train5)
    images_train7, labels_train7 = read_txt(TRAIN_DIR1, '/datasets/GenTextData/datasets_g/train.txt', images_train6, labels_train6)
    images_train8, labels_train8 = read_txt(TRAIN_DIR1, '/datasets/GenTextData/datasets_h/train.txt', images_train7, labels_train7)
    images_train9, labels_train9 = read_txt(TRAIN_DIR1, '/datasets/GenTextData/datasets_i/train.txt', images_train8, labels_train8)
    images_train10, labels_train10 = read_txt(TRAIN_DIR1, '/datasets/GenTextData/datasets_j/train.txt', images_train9, labels_train9)
    images_train11, labels_train11 = read_txt(TRAIN_DIR1, '/datasets/GenTextData/datasets_k/train.txt', images_train10, labels_train10)
    images_train12, labels_train12 = read_txt(TRAIN_DIR1, '/datasets/GenTextData/datasets_fix/train.txt', images_train11, labels_train11)
    images_train13, labels_train13 = read_txt(TRAIN_DIR2, '/datasets/GenTextData.old/datasets/train_new.txt', images_train12, labels_train12)
    
#     print len(images_train10)

#    images_train1, labels_train1 = read_txt(TRAIN_DIR1, '/datasets/GenTextData/datasets_h/train.txt', [], [])
#    images_train2, labels_train2 = read_txt(TRAIN_DIR1, '/datasets/GenTextData/datasets_i/train.txt', images_train1, labels_train1)
#    images_train3, labels_train3 = read_txt(TRAIN_DIR1, '/datasets/GenTextData/datasets_j/train.txt', images_train2, labels_train2)
#    images_train4, labels_train4 = read_txt(TRAIN_DIR1, '/datasets/GenTextData/datasets_k/train.txt', images_train3, labels_train3)

    # read val
    images_val, labels_val = read_txt(VAL_DIR, 'val.txt', [], [])

    createDataset(path_output + 'train', images_train13, labels_train13)
    createDataset(path_output + 'val', images_val, labels_val)

