import math
import random

import cv2
import numpy as np
from tensorpack.dataflow.imgaug.geometry import RotationAndCropValid

# from tf_pose.common import CocoPart
CocoPart = None

_network_w = 368
_network_h = 368
_scale = 8


def set_network_input_wh(w, h):
    global _network_w, _network_h
    _network_w, _network_h = w, h


def set_network_scale(scale):
    global _scale
    _scale = scale

def pose_random_scale(meta):
    scalew = random.uniform(0.8, 1.2)
    scaleh = random.uniform(0.8, 1.2)
    neww = int(meta.width * scalew)
    newh = int(meta.height * scaleh)
    dst = cv2.resize(meta.img, (neww, newh), interpolation=cv2.INTER_AREA)

    # adjust meta data
    adjust_joint_list = []
    for point in meta.joint_list:
        if point[0] < -100 or point[1] < -100:
            adjust_joint_list.append((-1000, -1000))
            continue
        # if point[0] <= 0 or point[1] <= 0 or int(point[0] * scalew + 0.5) > neww or int(
        #                         point[1] * scaleh + 0.5) > newh:
        #     adjust_joint_list.append((-1, -1))
        #     continue
        adjust_joint_list.append((int(point[0] * scalew + 0.5), int(point[1] * scaleh + 0.5)))

    meta.joint_list = adjust_joint_list
    meta.width, meta.height = neww, newh
    meta.img = dst
    return meta


def pose_resize_shortestedge_fixed(meta):
    ratio_w = _network_w / meta.width
    ratio_h = _network_h / meta.height
    ratio = max(ratio_w, ratio_h)
    return pose_resize_shortestedge(meta, int(min(meta.width * ratio + 0.5, meta.height * ratio + 0.5)))


def pose_resize_shortestedge_random(meta):
    ratio_w = _network_w / meta.width
    ratio_h = _network_h / meta.height
    ratio = min(ratio_w, ratio_h)
    target_size = int(min(meta.width * ratio + 0.5, meta.height * ratio + 0.5))
    target_size = int(target_size * random.uniform(0.95, 1.6))
    # target_size = int(min(_network_w, _network_h) * random.uniform(0.7, 1.5))
    return pose_resize_shortestedge(meta, target_size)


def pose_resize_shortestedge(meta, target_size):
    global _network_w, _network_h
    img = meta.img

    # adjust image
    scale = target_size / min(meta.height, meta.width)
    if meta.height < meta.width:
        newh, neww = target_size, int(scale * meta.width + 0.5)
    else:
        newh, neww = int(scale * meta.height + 0.5), target_size

    dst = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_AREA)

    pw = ph = 0
    if neww < _network_w or newh < _network_h:
        pw = max(0, (_network_w - neww) // 2)
        ph = max(0, (_network_h - newh) // 2)
        mw = (_network_w - neww) % 2
        mh = (_network_h - newh) % 2
        color = random.randint(0, 255)
        dst = cv2.copyMakeBorder(dst, ph, ph+mh, pw, pw+mw, cv2.BORDER_CONSTANT, value=(color, 0, 0))

    # adjust meta data
    adjust_joint_list = []
    for point in meta.joint_list:
        if point[0] < -100 or point[1] < -100:
            adjust_joint_list.append((-1000, -1000))
            continue
        # if point[0] <= 0 or point[1] <= 0 or int(point[0]*scale+0.5) > neww or int(point[1]*scale+0.5) > newh:
        #     adjust_joint_list.append((-1, -1))
        #     continue
        adjust_joint_list.append((int(point[0]*scale+0.5) + pw, int(point[1]*scale+0.5) + ph))

    meta.joint_list = adjust_joint_list
    meta.width, meta.height = neww + pw * 2, newh + ph * 2
    meta.img = dst
    return meta


def pose_crop_center(meta):
    global _network_w, _network_h
    target_size = (_network_w, _network_h)
    x = (meta.width - target_size[0]) // 2 if meta.width > target_size[0] else 0
    y = (meta.height - target_size[1]) // 2 if meta.height > target_size[1] else 0

    return pose_crop(meta, x, y, target_size[0], target_size[1])


def pose_crop_random(meta):
    global _network_w, _network_h
    target_size = (_network_w, _network_h)

    x = random.randrange(0, meta.width - target_size[0]) if meta.width > target_size[0] else 0
    y = random.randrange(0, meta.height - target_size[1]) if meta.height > target_size[1] else 0
    # for _ in range(50):
    #     x = random.randrange(0, meta.width - target_size[0]) if meta.width > target_size[0] else 0
    #     y = random.randrange(0, meta.height - target_size[1]) if meta.height > target_size[1] else 0

        # check whether any face is inside the box to generate a reasonably-balanced datasets
        # for joint in meta.joint_list:
        #     if x <= joint[CocoPart.Nose.value][0] < x + target_size[0] and y <= joint[CocoPart.Nose.value][1] < y + target_size[1]:
        #         break

    return pose_crop(meta, x, y, target_size[0], target_size[1])


def pose_crop(meta, x, y, w, h):
    # adjust image
    target_size = (w, h)

    img = meta.img
    resized = img[y:y+target_size[1], x:x+target_size[0], :]

    # adjust meta data
    adjust_joint_list = []
    for point in meta.joint_list:
        if point[0] < -100 or point[1] < -100:
            adjust_joint_list.append((-1000, -1000))
            continue
        # if point[0] <= 0 or point[1] <= 0:
        #     adjust_joint_list.append((-1000, -1000))
        #     continue
        new_x, new_y = point[0] - x, point[1] - y
        # if new_x <= 0 or new_y <= 0 or new_x > target_size[0] or new_y > target_size[1]:
        #     adjust_joint_list.append((-1, -1))
        #     continue
        adjust_joint_list.append((new_x, new_y))

    meta.joint_list = adjust_joint_list
    meta.width, meta.height = target_size
    meta.img = resized
    return meta


def pose_flip(meta):
    r = random.uniform(0, 1.0)
    if r > 0.5:
        return meta

    img = meta.img
    img = cv2.flip(img, 1)

    # flip meta
    # flip_list = [CocoPart.Nose, CocoPart.Neck, CocoPart.LShoulder, CocoPart.LElbow, CocoPart.LWrist, CocoPart.RShoulder, CocoPart.RElbow, CocoPart.RWrist,
    #              CocoPart.LHip, CocoPart.LKnee, CocoPart.LAnkle, CocoPart.RHip, CocoPart.RKnee, CocoPart.RAnkle,
    #              CocoPart.LEye, CocoPart.REye, CocoPart.LEar, CocoPart.REar, CocoPart.Background]
    adjust_joint_list = []
    for point in meta.joint_list:
        if point[0] < -100 or point[1] < -100:
            adjust_joint_list.append((-1000, -1000))
            continue
        # if point[0] <= 0 or point[1] <= 0:
        #     adjust_joint_list.append((-1, -1))
        #     continue
        adjust_joint_list.append((meta.width - point[0], point[1]))

    meta.joint_list = adjust_joint_list

    meta.img = img
    return meta


def pose_rotation(meta):
    deg = random.uniform(-15.0, 15.0)
    img = meta.img

    center = (img.shape[1] * 0.5, img.shape[0] * 0.5)       # x, y
    rot_m = cv2.getRotationMatrix2D((int(center[0]), int(center[1])), deg, 1)
    ret = cv2.warpAffine(img, rot_m, img.shape[1::-1], flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)
    if img.ndim == 3 and ret.ndim == 2:
        ret = ret[:, :, np.newaxis]
    neww, newh = RotationAndCropValid.largest_rotated_rect(ret.shape[1], ret.shape[0], deg)
    neww = min(neww, ret.shape[1])
    newh = min(newh, ret.shape[0])
    newx = int(center[0] - neww * 0.5)
    newy = int(center[1] - newh * 0.5)
    # print(ret.shape, deg, newx, newy, neww, newh)
    img = ret[newy:newy + newh, newx:newx + neww]

    # adjust meta data
    adjust_joint_list = []
    for point in meta.joint_list:
        if point[0] < -100 or point[1] < -100:
            adjust_joint_list.append((-1000, -1000))
            continue
        # if point[0] <= 0 or point[1] <= 0:
        #     adjust_joint_list.append((-1, -1))
        #     continue
        x, y = _rotate_coord((meta.width, meta.height), (newx, newy), point, deg)
        adjust_joint_list.append((x, y))

    meta.joint_list = adjust_joint_list
    meta.width, meta.height = neww, newh
    meta.img = img

    return meta


def _rotate_coord(shape, newxy, point, angle):
    angle = -1 * angle / 180.0 * math.pi

    ox, oy = shape
    px, py = point

    ox /= 2
    oy /= 2

    qx = math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    new_x, new_y = newxy

    qx += ox - new_x
    qy += oy - new_y

    return int(qx + 0.5), int(qy + 0.5)

def pose_to_img(meta_l):
    global _network_w, _network_h, _scale
    return [
        meta_l[0].img.astype(np.float16),
        meta_l[0].get_heatmap(target_size=(_network_w // _scale, _network_h // _scale)),
    ]


def get_hand_roi(meta):
    if len(meta.joint_list) <= 0:
        return [0,0,0,0]

    x1 = meta.joint_list[0][0]
    y1 = meta.joint_list[0][1]
    x2 = meta.joint_list[0][0]
    y2 = meta.joint_list[0][1]
    for x, y in meta.joint_list:
        x1 = min(x1, x)
        y1 = min(y1, y)
        x2 = max(x2, x)
        y2 = max(y2, y)

    return x1, y1, x2, y2

# def crop_hand_roi(meta, roi):
#     x1, y1, x2, y2 = roi

#     x1 = max(x1, 0)
#     y1 = max(y1, 0)
#     x2 = min(x2, meta.width-1)
#     y2 = min(y2, meta.height-1)

#     meta.img = meta.img[x1:x2+1, y1:y2+1, :]
#     meta.width = x2 - x1 + 1
#     meta.height = y2 - y1 + 1

#     adjust_joint_list = []
#     for point in meta.joint_list:
#         newx = point[0] - x1
#         newy = point[1] - y1
#         if newx < -100 and newy < -100:
#             adjust_joint_list.append((-1000,-1000))
#         else:
#             adjust_joint_list.append((newx, newy))

def crop_hand_roi(meta):
    x1, y1, x2, y2 = get_hand_roi(meta)
    w = x2 - x1 + 1
    h = y2 - y1 + 1

    x1 -= w * random.random() * 1.5
    y1 -= h * random.random() * 1.5
    x2 += w * random.random() * 1.5
    y2 += h * random.random() * 1.5

    # make sure not to go over the image border
    x1 = int(max(x1, 0))
    y1 = int(max(y1, 0))
    x2 = int(min(x2, meta.width-1))
    y2 = int(min(y2, meta.height-1))

    w = x2 - x1 + 1
    h = y2 - y1 + 1

    #
    return pose_crop(meta, x1, y1 , w, h)