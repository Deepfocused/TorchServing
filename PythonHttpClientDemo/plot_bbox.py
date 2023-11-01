import random

import cv2
import numpy as np
from matplotlib import pyplot as plt

def box_resize(bbox, in_size, out_size):
    """Resize bouding boxes according to image resize operation.

    Parameters
    ----------
    bbox : numpy.ndarray
        Numpy.ndarray with shape (N, 4+) where N is the number of bounding boxes.
        The second axis represents attributes of the bounding box.
        Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
        we allow additional attributes other than coordinates, which stay intact
        during bounding box transformations.
    in_size : tuple
        Tuple of length 2: (width, height) for input.
    out_size : tuple
        Tuple of length 2: (width, height) for output.

    Returns
    -------
    numpy.ndarray
        Resized bounding boxes with original shape.
    """
    if not len(in_size) == 2:
        raise ValueError("in_size requires length 2 tuple, given {}".format(len(in_size)))
    if not len(out_size) == 2:
        raise ValueError("out_size requires length 2 tuple, given {}".format(len(out_size)))

    bbox = bbox.copy()
    x_scale = out_size[0] / in_size[0]
    y_scale = out_size[1] / in_size[1]

    bbox[:, 1] = y_scale * bbox[:, 1]
    bbox[:, 3] = y_scale * bbox[:, 3]
    bbox[:, 0] = x_scale * bbox[:, 0]
    bbox[:, 2] = x_scale * bbox[:, 2]
    return bbox

def plot_bbox(img, bboxes, scores=None, ids=None, thresh=0.5,
              class_names=None, colors=None, reverse_rgb=False, absolute_coordinates=True):

    if isinstance(ids, list):
        ids = np.array(ids)
    if isinstance(scores, list):
        scores = np.array(scores)
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes)

    if ids is not None and not len(bboxes) == len(ids):
        raise ValueError('The length of ids and bboxes mismatch, {} vs {}'
                         .format(len(ids), len(bboxes)))
    if scores is not None and not len(bboxes) == len(scores):
        raise ValueError('The length of scores and bboxes mismatch, {} vs {}'
                         .format(len(scores), len(bboxes)))

    img = img.astype(np.uint8)
    if reverse_rgb:
        img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]

    copied_img = img.copy()

    if not absolute_coordinates:
        # convert to absolute coordinates using image shape
        height = img.shape[0]
        width = img.shape[1]
        bboxes[:, (0, 2)] *= width
        bboxes[:, (1, 3)] *= height

    # use random colors if None is provided
    if colors is None:
        colors = dict()

    for i, bbox in enumerate(bboxes):
        if scores is not None and scores.ravel()[i] < thresh:  # threshold보다 작은 것 무시
            continue
        if ids is not None and ids.ravel()[i] < 0:  # 0이하 인것들 인것 무시
            continue

        cls_id = int(ids.ravel()[i]) if ids is not None else -1
        if cls_id not in colors:
            if class_names is not None and cls_id != -1:
                colors[cls_id] = plt.get_cmap('hsv')(cls_id / len(class_names))
            else:
                colors[cls_id] = (random.random(), random.random(), random.random())

        denorm_color = [x * 255 for x in colors[cls_id]]
        bbox[np.isinf(bbox)] = 0
        bbox[bbox < 0] = 0
        xmin, ymin, xmax, ymax = [int(np.rint(x)) for x in bbox]
        try:
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), denorm_color, thickness=3)
        except Exception as E:
            print(E)

        if class_names is not None and cls_id < len(class_names):
            class_name = class_names[cls_id]
        else:
            class_name = str(cls_id) if cls_id >= 0 else ''

        score = '{:.2f}'.format(scores.ravel()[i]) if scores is not None else ''

        if class_name or score:
            cv2.putText(copied_img,
                        text='{} {}'.format(class_name, score), \
                        org=(xmin + 7, ymin + 20), \
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX, \
                        fontScale=0.5, \
                        color=[0, 0, 0], \
                        thickness=1, bottomLeftOrigin=False)

    img = cv2.addWeighted(img, 0.5, copied_img, 0.5, 0)
    return img
