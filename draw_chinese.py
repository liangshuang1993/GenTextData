# -*- coding: utf-8 -*-
import freetype
import cv2
import numpy as np
import math
import os
import copy

def draw_bbox(img, pt1, pt2, angle, color=(255, 0, 0)):
    bbox = get_bbox(pt1, pt2, angle)
    cv2.line(img, bbox[0], bbox[1], color, 1)
    cv2.line(img, bbox[1], bbox[2], color, 1)
    cv2.line(img, bbox[2], bbox[3], color, 1)
    cv2.line(img, bbox[3], bbox[0], color, 1)

def get_bbox(pt1, pt2, angle):
    bbox = []
    bbox.append(rotate_pt(pt1, (pt1[0], pt1[1]), angle))
    bbox.append(rotate_pt(pt1, (pt1[0], pt2[1]), angle))
    bbox.append(rotate_pt(pt1, (pt2[0], pt2[1]), angle))
    bbox.append(rotate_pt(pt1, (pt2[0], pt1[1]), angle))
    return bbox

def calc_bbox(points):
    min_ = [points[0][0], points[0][1]]
    max_ = [points[0][0], points[0][1]]
    for pt in points:
        if pt[0] < min_[0]:
            min_[0] = pt[0]
        if pt[1] < min_[1]:
            min_[1] = pt[1]
        if pt[0] > max_[0]:
            max_[0] = pt[0]
        if pt[1] > max_[1]:
            max_[1] = pt[1]
    return min_, max_

def rotate_pt(origin, pt, angle):
    delta = (pt[0] - origin[0], pt[1] - origin[1])
    angle_degree = angle * math.pi / 180
    sin_angle = math.sin(angle_degree)
    cos_angle = math.cos(angle_degree)
    return (int(origin[0] + cos_angle * delta[0] - sin_angle * delta[1]), int(origin[1] + sin_angle * delta[0] + cos_angle * delta[1]))

def draw_string(bg, face, string, position, angle, padding, color, ratio, gap):
    '''
    bg: pre-loaded images
    face: pre-loaded Face
    '''
    background = copy.deepcopy(bg)
    bg_height = background.shape[0]
    bg_width = background.shape[1]
    font_height = 32
    if not isinstance(string, unicode):
        string = string.decode('utf-8')
    face.set_char_size(width = int(32 * 64 * ratio), height = 32 * 64)
    rotate_matrix = cv2.getRotationMatrix2D((0, 0), angle, 1)
    matrix = freetype.Matrix(int(rotate_matrix[0, 0] * 0x10000L),
                                int(rotate_matrix[0, 1] * 0x10000L),
                                int(rotate_matrix[1, 0] * 0x10000L),
                                int(rotate_matrix[1, 1] * 0x10000L))
    face.set_transform(matrix, freetype.Vector())

    # adjust position to make sure the string always inside the background
    length = len(string)
    original_bbox_min = position
    original_bbox_max = [position[0] + (length - 1) * gap + length * ratio * font_height, position[1] + font_height]
    min_, max_ = calc_bbox(get_bbox(original_bbox_min, original_bbox_max, angle))
    min_[0] -= padding[3]
    min_[1] -= padding[0]
    max_[0] += padding[2]
    max_[1] += padding[1]
    if min_[0] < 0:
        position[0] -= min_[0]
    if min_[1] < 0:
        position[1] -= min_[1]
    if max_[0] > bg_width:
        position[0] -= max_[0] - bg_width
    if max_[1] > bg_height:
        position[1] -= max_[1] - bg_height
    original_bbox_min = position
    original_bbox_max = [position[0] + (length - 1) * gap + length * ratio * font_height, position[1] + font_height]
    min_, max_ = calc_bbox(get_bbox(original_bbox_min, original_bbox_max, angle))
    min_[0] -= padding[3]
    min_[1] -= padding[0]
    max_[0] += padding[2]
    max_[1] += padding[1]

    idx = 0
    for char in string:
        face.load_char(char)
        bitmap = face.glyph.bitmap
        cols = bitmap.width
        rows = bitmap.rows
        glyph_pixels = bitmap.buffer
        # print bitmap.buffer
        pt = [position[0] + idx * (ratio * font_height + gap), position[1]]
        new_pt = rotate_pt(position, pt, angle)
        x_pos = new_pt[0]
        y_pos = new_pt[1]

        for row in range(rows):
            for col in range(cols):
                if glyph_pixels[row*cols + col] != 0:
                    background[y_pos + row][x_pos + col][0] += color[0]
                    background[y_pos + row][x_pos + col][1] += color[1]
                    background[y_pos + row][x_pos + col][2] += color[2]
        idx += 1
    return min_, max_, background

def load_backgrounds(path):
    background_files = os.listdir(path)
    bgs = []
    for file in background_files:
        bg = cv2.imread(path + file)
        bgs.append(bg)
    return bgs

def load_faces(path, chinese = False):
    font_files = os.listdir(path)
    faces = []
    for file in font_files:
        if not chinese or has_chinese(file):
            face = freetype.Face(path + file)
            faces.append(face)
    return faces


def has_chinese(word):
    if not isinstance(word, unicode):
        word = word.decode('utf-8')
    for ch in word:
        # uc=ch.decode('utf-8')
        if u'\u4e00' <= ch <= u'\u9fff':
            return True

    return False

if __name__ == '__main__':
    with open('word2.txt', 'r') as f:
        labels = f.readlines()
    backgrounds = load_backgrounds('./background/')
    english_faces = load_faces('./fonts/', False) # chinese font supports english
    chinese_faces = load_faces('./fonts/', True)
    faces = []

    for label in labels:
        label = label.strip()
        for bg in backgrounds:
            bg_height = bg.shape[0]
            bg_width = bg.shape[1]
            if has_chinese(label):
                faces = chinese_faces
            else:
                faces = english_faces
            for face in faces:
                for positionIdx in range(2):
                    x = np.random.randint(0, bg_width)
                    y = np.random.randint(0, bg_height)
                    position = [x, y]
                    for angle in range(-6, 6, 1):
                        for paddingIdx in range(10):
                            padding = [np.random.randint(0, 5), np.random.randint(0, 8), np.random.randint(0, 5), np.random.randint(0, 8)]
                            for colorIdx in range(5):
                                color = [np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255)]
                                for ratio in range(6, 14, 1):
                                    ratio /= 10.0
                                    for gap in range(0, 6, 2):
                                        min_, max_, background = draw_string(bg, face, label, position, angle, padding, color, ratio, gap)
                                        crop_img = background[min_[1]:max_[1], min_[0]:max_[0]]
                                        cv2.imshow('background', background)
                                        cv2.imshow('trim', crop_img)
                                        cv2.waitKey(0)