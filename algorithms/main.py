import logging
import os
import uuid

import cv2
from flask import url_for

from algorithms.basic_openpose import get_points_from_image as op_from_image
from algorithms.basic_openpose import get_points_from_video as op_from_video
from algorithms.tensorflow_openpose import get_points_from_image as tf_from_image

import numpy as np
import glob

STATIC_FOLDER = './static'
logger = logging.getLogger(__name__)


def analyse_image(algorithm, image_path, opWrapper):
    try:
        logger.debug('Analysing path: {} with algorithm: {}'.format(image_path, algorithm))

        if algorithm == 'tf-openpose':
            data = tf_from_image(image_path)
        elif algorithm == 'openpose':
            data = op_from_image(image_path, opWrapper)
        else:
            raise KeyError('Specified algorithm not supported.')

        logger.debug('Algorithm finished, raw data: {}'.format(data))

        # save the result as image
        title = 'result.jpg'
        image_path = os.path.join(STATIC_FOLDER, title)

        logger.debug('Writing resulting image to: {}'.format(image_path))

        cv2.imwrite(image_path, data)
        result_path = url_for('static', filename=title)

        logger.debug('Final image path: {}'.format(result_path))

        return True, result_path
    except Exception as e:
        logger.error('An error occurred while analysing image')
        logger.error(e, exc_info=True)
        print(e)
        return False, str(e)


def analyse_video(algorithm, video_path, results_per_second, opWrapper):
    try:
        logger.debug(
            'Analysing path: {} with algorithm: {}, and RPS: {}'.format(video_path, algorithm, results_per_second))

        if algorithm == 'openpose':
            data = op_from_video(video_path, results_per_second, opWrapper)
        else:
            raise KeyError('Specified algorithm not supported.')

        logger.debug('Algorithm finished, result:')
        logger.debug(data)

        # save the result as video
        img_array = []
        for filename in glob.glob('./static/*.jpg'):
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)

        title = 'result.mp4'
        video_path = os.path.join(STATIC_FOLDER, title)

        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), results_per_second, size)
 
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

        return True, data
    except Exception as e:
        logger.error('An error occurred while analysing image')
        logger.error(e, exc_info=True)
        print(e)
        return False, str(e)
