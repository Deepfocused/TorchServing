import argparse
import logging
import os

import cv2
import numpy as np
import requests
from tqdm import tqdm

from plot_bbox import plot_bbox

logfilepath = ""
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)

def RunHttpClientImage(input="sample.jpg", output="result.jpg", classfile="class.txt", thresh=0.5, save_image=True, show_image=True):

    try:
        class_names = np.loadtxt(classfile, dtype=str)
    except Exception:
        raise FileExistsError

    if len([class_names]) == 1:
        class_names = [class_names]

    if not os.path.exists(input):
        raise FileExistsError

    dir, filename = os.path.split(output)
    if not dir == '':
        os.makedirs(dir)

    ids = []
    scores = []
    boxes = []
    image = cv2.imread(input, flags=cv2.IMREAD_COLOR)

    # 서버  호출
    '''
    requests.post argument data: (optional) Dictionary, list of tuples, bytes, or file-like
    object to send in the body of the :class:`Request`.
    '''
    _, data = cv2.imencode('.jpg', image)
    response = requests.post(URL, data=data.tobytes(order='C')).json()

    if response:
        for rb in response:
            ids.append(0)
            boxes.append(rb['faces'])
            scores.append(rb['score'])
    else:
        print("Request failed")

    plotted_image = plot_bbox(image, boxes, scores=scores, ids=ids, thresh=thresh, class_names = class_names)
    if save_image:
        cv2.imwrite(output, plotted_image)

    if show_image:
        cv2.imshow("image", plotted_image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


def RunHttpClientVideo(input="sample.mp4", output="result.mp4", classfile="class.txt", thresh=0.5, save_video=True, show_image=True):

    try:
        class_names = np.loadtxt(classfile, dtype=str)
    except Exception:
        raise FileExistsError

    if len([class_names]) == 1:
        class_names = [class_names]

    if not os.path.exists(input):
        raise FileExistsError

    dir, filename = os.path.split(output)
    if not dir == '':
        os.makedirs(dir)

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    cap = cv2.VideoCapture(input)
    nframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    twidth  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    theight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output, fourcc, fps, (twidth, theight))

    # while(cap.isOpened()):
    for _ in tqdm(range(0, nframe)):
        ret, image = cap.read()
        if ret:
            ids = []
            scores = []
            boxes = []

            # 서버  호출
            '''
            requests.post argument data: (optional) Dictionary, list of tuples, bytes, or file-like
            object to send in the body of the :class:`Request`.
            '''
            _, data = cv2.imencode('.jpg', image)
            response = requests.post(URL, data=data.tobytes(order='C')).json()
            if response:
                for rb in response:
                    ids.append(0)
                    boxes.append(rb['faces'])
                    scores.append(rb['score'])
            else:
                print("Request failed")

            plotted_image = plot_bbox(image, boxes, scores=scores, ids=ids, thresh=thresh, class_names = class_names)
            if save_video:
                out.write(plotted_image)

            if show_image:
                cv2.imshow("image", plotted_image)
                cv2.waitKey(1)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    # https://greeksharifa.github.io/references/2019/02/12/argparse-usage/

    parser = argparse.ArgumentParser(description='python http client')
    parser.add_argument('--address', type=str, help="address:port", default="192.168.35.69:8080")
    parser.add_argument('--input', type=str, help="input image", default="sample.jpg")
    parser.add_argument('--output', type=str, help="output image or video", default="result.jpg")
    parser.add_argument('--classfile', type=str, help="class file", default="class.txt")
    parser.add_argument('--thresh', type=float, help='visual threshold', default=0.5)

    parser.add_argument('--save', action='store_true', help="image or video save?")
    parser.add_argument('--show', action='store_true', help="image show?")

    args = parser.parse_args()

    URL = 'http://' + args.address + "/predictions/facedetector" # 이미지 분석 요청

    basename = os.path.basename(args.input)
    ext = os.path.splitext(basename)[-1]

    if ext.lower() in [".jpg", ".png", ".jpeg"]:
        RunHttpClientImage(input=args.input, output=args.output, classfile = args.classfile,thresh=args.thresh, save_image=args.save, show_image=args.show)
    elif ext in [".mp4", ".avi"]:
        RunHttpClientVideo(input=args.input, output=args.output, classfile = args.classfile, thresh=args.thresh, save_video=args.save, show_image=args.show)
    else:
        raise NotImplementedError

