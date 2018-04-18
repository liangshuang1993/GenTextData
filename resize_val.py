import os
import cv2

SOURCE_DIR = 'datasets/val_original'
DEST_DIR = 'datasets/val'

val_images = os.listdir(SOURCE_DIR)

for image in val_images:
    img = cv2.imread(os.path.join(SOURCE_DIR, image))
    h, w, c = img.shape

    img = cv2.resize(img, (w / h * 32, 32))
    print img.shape
    cv2.imwrite(os.path.join(DEST_DIR, image), img)
