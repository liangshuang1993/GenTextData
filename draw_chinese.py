# -*- coding: utf-8 -*-
import freetype
import cv2
import numpy as np
import math

def draw_bbox(img, pt1, pt2, angle):
    bbox = get_bbox(pt1, pt2, angle)
    cv2.line(img, bbox[0], bbox[1], (255, 0, 0), 1)
    cv2.line(img, bbox[1], bbox[2], (255, 0, 0), 1)
    cv2.line(img, bbox[2], bbox[3], (255, 0, 0), 1)
    cv2.line(img, bbox[3], bbox[0], (255, 0, 0), 1)

def get_bbox(pt1, pt2, angle):
    bbox = []
    bbox.append(rotate_pt(pt1, (pt1[0], pt1[1]), angle))
    bbox.append(rotate_pt(pt1, (pt1[0], pt2[1]), angle))
    bbox.append(rotate_pt(pt1, (pt2[0], pt2[1]), angle))
    bbox.append(rotate_pt(pt1, (pt2[0], pt1[1]), angle))
    return bbox

def calc_bbox(bbox):
    min = [bbox[0][0], bbox[0][1]]
    max = [bbox[0][0], bbox[0][1]]
    for pt in bbox:
        if pt[0] < min[0]:
            min[0] = pt[0]
        if pt[1] < min[1]:
            min[1] = pt[1]
        if pt[0] > max[0]:
            max[0] = pt[0]
        if pt[1] > max[1]:
            max[1] = pt[1]
    return (min, max)

def rotate_pt(origin, pt, angle):
    delta = (pt[0] - origin[0], pt[1] - origin[1])
    angle_degree = angle * math.pi / 180
    sin_angle = math.sin(angle_degree)
    cos_angle = math.cos(angle_degree)
    return (int(origin[0] + cos_angle * delta[0] - sin_angle * delta[1]), int(origin[1] + sin_angle * delta[0] + cos_angle * delta[1]))

background = cv2.imread('background/bg4.jpg')

face = freetype.Face('fonts/原版宋体.ttf')
face.set_char_size( width = 48 * 64, height = 20 * 64 )

angle = 10
rotate_matrix = cv2.getRotationMatrix2D((0, 0), angle, 1)
matrix = freetype.Matrix(int(rotate_matrix[0, 0] * 0x10000L),
                            int(rotate_matrix[0, 1] * 0x10000L),
                            int(rotate_matrix[1, 0] * 0x10000L),
                            int(rotate_matrix[1, 1] * 0x10000L))

face.set_transform(matrix, freetype.Vector())

face.load_char('测')
bitmap = face.glyph.bitmap
cols = bitmap.width
rows = bitmap.rows
glyph_pixels = bitmap.buffer
# print bitmap.buffer

x_pos = 100
y_pos = 200
color = [100, 200, 150]

for row in range(rows):
    for col in range(cols):
        if glyph_pixels[row*cols + col] != 0:
            background[y_pos + row][x_pos + col][0] += color[0]
            background[y_pos + row][x_pos + col][1] += color[1]
            background[y_pos + row][x_pos + col][2] += color[2]

draw_bbox(background, (100, 200), (100+48, 200+20), angle)
new_bbox = calc_bbox(get_bbox((100, 200), (100+48, 200+20), angle))
draw_bbox(background, new_bbox[0], new_bbox[1], 0)

cv2.imshow('s', background)
cv2.waitKey(0)