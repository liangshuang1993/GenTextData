# -*- coding: utf-8 -*-
from gen import *
import cv2
import os
import numpy as np
import math

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
    def __init__(self, label_file, height=32, font_size=24, margin=(3, 4), step=3):
        self.backgounds = os.listdir('background')
        self.height = height
        self.font_size = font_size
        self.english_fonts = ['Times New Roman.ttf', '宋体_GB18030+%26+新宋体_GB18030.ttc', 'simhei.ttf']
        self.chinese_fonts = ['原版宋体.ttf', '宋体_GB18030+%26+新宋体_GB18030.ttc', 'Ubuntu-Bold.ttf', 'simhei.ttf']
        self.margin = margin
        self.step = step
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

    def get_color(self, thes=10):
        # color_ = (0, 0, 0) # black
        return (np.random.randint(thes), np.random.randint(thes), np.random.randint(thes))

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

    def generate_pics(self, image_dir):

        for label in self.labels:
            # print label
            # print len(label.decode('utf-8'))
            label = label.strip()

            index = 0
            print label

            length = len(label.decode('utf-8'))
            # for gap1 in [5]:
            for start1 in range(0, 2 * self.margin[1], np.random.randint(1, self.step + 1)):
                for angle in range(-5, 5, np.random.randint(1, self.step + 1)):

                    if self.__has_chinese(label.decode('utf-8')):

                        # hor
                        for start0 in range(0, 2 * (self.margin[0] + length), np.random.randint(1, self.step + 1)):
                        # for gap0 in [2]:
                            pos = (start0, start1)
                            for font in self.chinese_fonts:
                                try:
                                    img = self.get_bg(length)
                                    color = self.get_color()
                                    self.__draw(image_dir, img, pos, label, color, font,
                                                index, angle)
                                    index += 1
                                except Exception as e:
                                    print e
                                    continue
                    else:
                        for start0 in range(0, 2 * (self.margin[0] + 3 * length), np.random.randint(1, self.step + 1)):
                        # for gap0 in [2]:
                            pos = (start0, start1)
                            for font in self.english_fonts:
                                try:
                                    img = self.get_bg(length)
                                    color = self.get_color()
                                    self.__draw(image_dir, img, pos, label, color, font,
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
        image = np.uint8(np.clip((np.random.randint(10, 20) / 10.0 * image + np.random.randint(20)), 0, 255))
        # affine_transform
        # length = len(label.decode('utf-8'))
        # if length < 5:
        #     M = self.make_affine_transform(
        #         from_shape=image.shape,
        #         to_shape=img.shape,
        #         min_scale=0.2,
        #         max_scale=0.6,
        #         rotation_variation=0.6,
        #         scale_variation=1.2,
        #         translation_variation=0.6)
        # else:
        #     M = self.make_affine_transform(
        #         from_shape=image.shape,
        #         to_shape=img.shape,
        #         min_scale=0.5,
        #         max_scale=0.5,
        #         rotation_variation=0.1,
        #         scale_variation=0.2,
        #         translation_variation=0.2)
        #
        # ht, wt = text.shape[0], text.shape[1]
        #
        # corners_bf = numpy.matrix([[0, wt, 0, wt],
        #                            [0, 0, ht, ht]])
        # text = cv2.warpAffine(text, M, (bg.shape[1], bg.shape[0]))
        # corners_af = numpy.dot(M[:2, :2], corners_bf) + M[:2, -1]
        # tl = numpy.min(corners_af, axis=1).T
        # br = numpy.max(corners_af, axis=1).T
        # box = numpy.hstack([tl, br])
        #
        # rand = 230
        # (h, s, v) = rgb2hsv(bg_color[0], bg_color[1], bg_color[2])
        # if v > 0.85 and v <= 1.0:
        #     out = bg - text * rand
        #     out[out < 0] = 0
        # else:
        #     out = text + bg
        # out = cv2.resize(out, (bg.shape[1], bg.shape[0]))
        # out = numpy.clip(out, 0., 1.)



        image_path = os.path.join(image_dir, label + '_' + str(index) + '.jpg')
        cv2.imwrite(image_path, image)

    def __has_chinese(self, word):
        for ch in word:
            # uc=ch.decode('utf-8')
            if u'\u4e00' <= ch <= u'\u9fff':
                return True

        return False

    def __make_affine_transform(self, from_shape, to_shape,
                              min_scale, max_scale,
                              scale_variation=1.0,
                              rotation_variation=1.0,
                              translation_variation=1.0):

        out_of_bounds_scale = True
        out_of_bounds_trans = True
        from_size = np.array([[from_shape[1], from_shape[0]]]).T
        to_size = np.array([[to_shape[1], to_shape[0]]]).T

        while out_of_bounds_scale:
            scale = np.random.uniform((min_scale + max_scale) * 0.5 -
                                   (
                                   max_scale - min_scale) * 0.5 * scale_variation,
                                   (min_scale + max_scale) * 0.5 +
                                   (
                                   max_scale - min_scale) * 0.5 * scale_variation)
            if scale > max_scale or scale < min_scale:
                continue
            out_of_bounds_scale = False

        roll = np.random.uniform(-0.3, 0.3) * rotation_variation
        pitch = np.random.uniform(-0.2, 0.2) * rotation_variation
        yaw = np.random.uniform(-1.2, 1.2) * rotation_variation
        M = self.euler_to_mat(yaw, pitch, roll)[:2, :2]
        h, w = from_shape[0], from_shape[1]
        corners = np.matrix([[-w, +w, -w, +w],
                             [-h, -h, +h, +h]]) * 0.5
        skewed_size = np.array(np.max(np.dot(M, corners), axis=1) -
                                  np.min(np.dot(M, corners), axis=1))
        # Set the scale as large as possible such that the skewed and scaled shape
        # is less than or equal to the desired ratio in either dimension.
        scale *= np.min(to_size / skewed_size)
        # Set the translation such that the skewed and scaled image falls within
        # the output shape's bounds.
        while out_of_bounds_trans:
            trans = (np.random.random((2, 1)) - 0.5) * translation_variation
            trans = ((2.0 * trans) ** 5.0) / 2.0
            if np.any(trans < -0.5) or np.any(trans > 0.5):
                continue
            out_of_bounds_trans = False
        trans = (to_size - skewed_size * scale) * trans

        center_to = to_size / 2.
        center_from = from_size / 2.
        M = self.euler_to_mat(yaw, pitch, roll)[:2, :2]
        M *= scale
        T = trans + center_to - np.dot(M, center_from)
        M = np.hstack([M, T])
        return M

    def euler_to_mat(self, yaw, pitch, roll):

        # Rotate clockwise about the Y-axis
        c, s = math.cos(yaw), math.sin(yaw)
        M = np.matrix([[  c, 0.,  s],
                          [ 0., 1., 0.],
                          [ -s, 0.,  c]])

        # Rotate clockwise about the X-axis
        c, s = math.cos(pitch), math.sin(pitch)
        M = np.matrix([[ 1., 0., 0.],
                          [ 0.,  c, -s],
                          [ 0.,  s,  c]]) * M

        # Rotate clockwise about the Z-axis
        c, s = math.cos(roll), math.sin(roll)
        M = np.matrix([[  c, -s, 0.],
                          [  s,  c, 0.],
                          [ 0., 0., 1.]]) * M

        return M


if __name__ == '__main__':
    gen = GenPic('word2.txt')
    gen.generate_pics(TRAIN_DIR)
    # gen.gen_dicts('new_label.txt')
    # gen.generate_pics(VAL_DIR, 'test.txt')
    print 'done'
