import argparse
import logging
import sys
import time

from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimatorRun')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--video', type=str, default='./images/p1.jpg')
    parser.add_argument('--model', type=str, default='cmu',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. '
                             'default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    parser.add_argument('--mode', type=str, default="all",
                        help='all or triangle')
    parser.add_argument('--background', type=bool, default=True,
                        help='True of False')

    args = parser.parse_args()

    w, h = model_wh(args.resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    # estimate human poses from a single image !
    # image = common.read_imgfile(args.image, None, None)
    video_file = args.video
    cap = cv2.VideoCapture(video_file)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    output_filename = str(video_file)
    if args.mode == "all":
        output_filename = str(output_filename).replace(".", "_all.")
    elif args.mode == "triangle":
        output_filename = str(output_filename).replace(".", "_triangle.")
    elif args.background :
        output_filename = str(output_filename).replace(".", "_only_skel.")
    print(output_filename)
    out = cv2.VideoWriter(output_filename, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    frame_count = 0
    while cap.isOpened():
        ret_val, image = cap.read()
        frame_count+=1
        if ret_val == False :
            break
        t = time.time()
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        elapsed = time.time() - t
        if frame_count % 10 == 0 :
            print("frame_count:\t" + str(frame_count))
            print('inference frame: %s in %.4f seconds.' % (frame_count, elapsed))
        if args.background :
            image = np.zeros(image.shape, np.uint8) + 255
        if args.mode == "all" :
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        if args.mode == "triangle":
            image = TfPoseEstimator.draw_triangle(image, humans, imgcopy=False)
        out.write(image)
    cap.release()
    out.release()
    cv2.destroyAllWindows()