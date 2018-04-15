# -*- coding: utf-8 -*-
from gen import *
import cv2
import os
import numpy as np

DATASET = 'datasets'
TRAIN_DIR = DATASET + '/train'
VAL_DIR = DATASET + '/val'

if not os.path.exists(DATASET):
    os.mkdir(DATASET)

if not os.path.exists(TRAIN_DIR):
    os.mkdir(TRAIN_DIR)

# if not os.path.exists(VAL_DIR):
#     os.mkdir(VAL_DIR)

class GenPic(object):
    def __init__(self, label_file, height=32, font_size=24, gap=(3, 4)):
        self.backgounds = os.listdir('background')
        self.height = height
        self.font_size = font_size
        self.english_fonts = ['Times New Roman.ttf', '宋体_GB18030+%26+新宋体_GB18030.ttc']
        self.chinese_fonts = ['原版宋体.ttf', '宋体_GB18030+%26+新宋体_GB18030.ttc']
        self.gap = gap
        with open(label_file, 'r') as f:
            self.labels = f.readlines()

    def get_bg(self, length, gap=3):
        bg = np.random.choice(self.backgounds)
        img = cv2.imread('background/' + bg)
        b_height, b_width, b_channel = img.shape

        # crop sutible size
        width = (self.font_size + 2)* length + gap * 2
        if b_height < self.height:
            raise Exception('Background height less than font_height')
        if b_width < width:
            raise Exception('Background width less than font_with')

        pos_h = np.random.randint(b_height - self.height)
        pos_w = np.random.randint(b_width - width)

        return img[pos_h: pos_h + self.height, pos_w: pos_w + width]

    def gen_dicts(self, label_file):
        dicts = []

        for label in self.labels:
            label = label.strip().decode('utf-8')
            for char in label:
                dicts.append(char)

        dicts = list(set(dicts))

        with open('dicts.txt', 'w') as f:
            for char in dicts:
                f.write(char.encode('utf-8'))

    def generate_pics(self, image_dir, label_file):

        for label in self.labels:
            # print label
            # print len(label.decode('utf-8'))
            label = label.strip()
            color_ = (0,0,0)
            index = 0
            print label

            length = len(label.decode('utf-8'))
            # for gap1 in [5]:
            for gap1 in range(0, 2 * self.gap[1], 2):
                for angle in range(-5, 5, 2):

                    if self.__has_chinese(label.decode('utf-8')):

                        # hor
                        for gap0 in range(0, 2 * (self.gap[0] + length), 2):
                        # for gap0 in [2]:
                            pos = (gap0, gap1)
                            for font in self.chinese_fonts:
                                try:
                                    img = self.get_bg(length)
                                    self.__draw(image_dir, img, pos, label, color_, font,
                                                index, angle)
                                    index += 1
                                except Exception as e:
                                    print e
                                    continue
                    else:
                        for gap0 in range(0, 2 * (self.gap[0] + 3 * length), 2):
                        # for gap0 in [2]:
                            pos = (gap0, gap1)
                            for font in self.english_fonts:
                                try:
                                    img = self.get_bg(length)
                                    self.__draw(image_dir, img, pos, label, color_, font,
                                                index, angle)
                                    index += 1
                                except Exception as e:
                                    print e
                                    continue

    def __draw(self, image_dir, img, pos, label, color_, font, index, angle):
        ft = put_chinese_text('fonts/' + font)
        image = ft.draw_text(img, pos, label, self.font_size, color_, angle)
        # light = np.random.choice(range(8, 13))
        # image = image * light / 10.0
        image = np.uint8(np.clip((1.5 * image + np.random.randint(10)), 0, 255))
        image_path = os.path.join(image_dir, label + '_' + str(index) + '.jpg')
        cv2.imwrite(image_path, image)


    def __has_chinese(self, word):
        for ch in word:
            # uc=ch.decode('utf-8')
            if u'\u4e00' <= ch <= u'\u9fff':
                return True

        return False


if __name__ == '__main__':
    gen = GenPic('new_label.txt')
    gen.generate_pics(TRAIN_DIR)
    gen.gen_dicts('new_label.txt')
    # gen.generate_pics(VAL_DIR, 'test.txt')
    print 'done'
