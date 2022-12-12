import os
import time

import cv2
import numpy as np
import tensorflow as tf
from keras.applications.imagenet_utils import preprocess_input
from nets.retinaface import RetinaFace
from utils.anchors import Anchors
from utils.config import cfg_mnet, cfg_re50
from utils.utils import letterbox_image
from utils.utils_bbox import BBoxUtility, retinaface_correct_boxes



class Retinaface(object):
    _defaults = {

        "model_path"        : 'model_data/retinaface_mobilenet025.h5',
        "backbone"          : 'mobilenet',
        "confidence"        : 0.5,
        "nms_iou"           : 0.45,
        "input_shape"       : [640, 480, 3],
        "letterbox_image"   : True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        if self.backbone == "mobilenet":
            self.cfg = cfg_mnet
        else:
            self.cfg = cfg_re50

        self.bbox_util  = BBoxUtility(nms_thresh=self.nms_iou)
        self.anchors    = Anchors(self.cfg, image_size=(self.input_shape[0], self.input_shape[1])).get_anchors()
        self.generate()

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'tensorflow.keras model or weights must be a .h5 file.'
        self.retinaface = RetinaFace(self.cfg, self.backbone)
        self.retinaface.load_weights(self.model_path)
        print('{} model, anchors loaded.'.format(self.model_path))

    @tf.function
    def get_pred(self, photo):
        preds = self.retinaface(photo, training=False)
        return preds
