# -*- coding: utf-8 -*-
from gen import *
import cv2
import os
import numpy as np
import math
from datetime import datetime

DATASET = 'datasets'
TRAIN_DIR = DATASET + '/train'

if not os.path.exists(DATASET):
    os.mkdir(DATASET)

if not os.path.exists(TRAIN_DIR):
    os.mkdir(TRAIN_DIR)

class GenPic(object):
    def __init__(self, label_file, prefix, height=32, font_size=24, margin=(5, 4), step=3):
        self.backgounds = os.listdir('background')
        self.height = height
        self.font_size = font_size
        self.english_fonts = ['Times New Roman.ttf', '宋体_GB18030+%26+新宋体_GB18030.ttc', 'Ubuntu-Bold.ttf', 'simhei.ttf']
        self.chinese_fonts = ['原版宋体.ttf', '宋体_GB18030+%26+新宋体_GB18030.ttc']
        self.margin = margin
        self.step = step
        self.prefix = prefix
        self.count = 0
        with open(label_file, 'r') as f:
            self.labels = f.readlines()

        self.f = open('train.txt', 'w')

    def get_bg(self, length):
        bg = np.random.choice(self.backgounds)
        img = cv2.imread('background/' + bg)
        b_height, b_width, b_channel = img.shape

        # crop sutible size
        width = (self.font_size + 5)* length + self.margin[0] * 2
        if b_height < self.height:
            raise Exception('Background height less than font_height')
        if b_width < width:
            raise Exception('Background width less than font_with')

        pos_h = np.random.randint(b_height - self.height)
        pos_w = np.random.randint(b_width - width)

        return img[pos_h: pos_h + self.height, pos_w: pos_w + width]

    def get_color(self, thes=50):
        # color_ = (0, 0, 0) # black
        return (np.random.randint(thes), np.random.randint(thes), np.random.randint(thes))

    def generate_pics(self, image_dir):

        for label in self.labels:
            # print label
            # print len(label.decode('utf-8'))
            label = label.strip()

            print label

            length = len(label.decode('utf-8'))

            step = np.random.randint(2, self.step + 1)
            # for gap1 in [5]:
            for start1 in range(2, 2 * self.margin[1], step):
                for angle in range(-15, 15, 3):

                    if self.__has_chinese(label.decode('utf-8')):

                        # horizantal
                        for start0 in range(2, 2 * (self.margin[0] + length) + step - 1, step):
                            if start0 > 2 * (self.margin[0] + length):
                                start0 = 2 * (self.margin[0] + length)
                            pos = (start0, start1)
                            for font in self.chinese_fonts:
                                try:
                                    img = self.get_bg(length)
                                    color = self.get_color()
                                    self.__draw(image_dir, img, pos, label, color, font,
                                                angle)
                                except Exception as e:
                                    print e
                    else:
                        for start0 in range(2, 2 * (self.margin[0] + 3 * length) + step - 1, step):
                            if start0 > 2 * (self.margin[0] + 3 * length):
                                start0 = 2 * (self.margin[0] + 3 * length)
                            pos = (start0, start1)
                            for font in self.english_fonts:
                                try:
                                    img = self.get_bg(length)
                                    color = self.get_color()
                                    self.__draw(image_dir, img, pos, label, color, font,
                                                angle)
                                except Exception as e:
                                    print e

    def __draw(self, image_dir, img, pos, label, color_, font, angle):
        ft = put_chinese_text('fonts/' + font)
        image = ft.draw_text(img,
                             pos,
                             label,
                             np.random.randint(self.font_size - 5,
                                               self.font_size),
                             color_,
                             angle)
        # light = np.random.choice(range(8, 13))
        # image = image * light / 10.0
        image = np.uint8(np.clip((np.random.randint(10, 20) / 10.0 * image + np.random.randint(20)), 0, 255))
        image = cv2.resize(image, (int(image.shape[1] * np.random.randint(8, 12) / 10.0), 32))

        # label contains / or other char

        # image_path = os.path.join(image_dir, datetime.now().strftime('%m-%d-%H:%M:%S:%f') + '.jpg')
        image_path = os.path.join(image_dir, self.prefix + str(self.count) + '.jpg')
        self.count += 1
        cv2.imwrite(image_path, image)
        self.f.write(image_path + ' ' + label + '\n')


    def __has_chinese(self, word):
        for ch in word:
            # uc=ch.decode('utf-8')
            if u'\u4e00' <= ch <= u'\u9fff':
                return True

        return False


if __name__ == '__main__':

    # only use this script to generate train images, val images use real images
    gen = GenPic('word2.txt', 'f')
    gen.generate_pics(TRAIN_DIR)
    gen.f.close()
    print 'done'
