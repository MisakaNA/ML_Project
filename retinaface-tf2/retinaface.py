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

    #---------------------------------------------------#
    #   初始化Retinaface
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            
        #---------------------------------------------------#
        #   不同主干网络的config信息
        #---------------------------------------------------#
        if self.backbone == "mobilenet":
            self.cfg = cfg_mnet
        else:
            self.cfg = cfg_re50

        #---------------------------------------------------#
        #   工具箱和先验框的生成
        #---------------------------------------------------#
        self.bbox_util  = BBoxUtility(nms_thresh=self.nms_iou)
        self.anchors    = Anchors(self.cfg, image_size=(self.input_shape[0], self.input_shape[1])).get_anchors()
        self.generate()

    #---------------------------------------------------#
    #   载入模型
    #---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'tensorflow.keras model or weights must be a .h5 file.'

        #-------------------------------#
        #   载入模型与权值
        #-------------------------------#
        self.retinaface = RetinaFace(self.cfg, self.backbone)
        self.retinaface.load_weights(self.model_path)
        print('{} model, anchors loaded.'.format(self.model_path))

    @tf.function
    def get_pred(self, photo):
        preds = self.retinaface(photo, training=False)
        return preds

    def get_FPS(self, image, test_interval):
        # ---------------------------------------------------#
        #   把图像转换成numpy的形式
        # ---------------------------------------------------#
        image = np.array(image, np.float32)
        # ---------------------------------------------------#
        #   计算输入图片的高和宽
        # ---------------------------------------------------#
        im_height, im_width, _ = np.shape(image)
        # ---------------------------------------------------------#
        #   letterbox_image可以给图像增加灰条，实现不失真的resize
        # ---------------------------------------------------------#
        if self.letterbox_image:
            image = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        else:
            self.anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()
        # ---------------------------------------------------------#
        #   图片预处理，归一化。
        # ---------------------------------------------------------#
        photo = np.expand_dims(preprocess_input(image), 0)

        t1 = time.time()
        for _ in range(test_interval):
            # ---------------------------------------------------------#
            #   传入网络进行预测
            # ---------------------------------------------------------#
            preds = self.get_pred(photo)
            preds = [pred.numpy() for pred in preds]
            # ---------------------------------------------------------#
            #   将预测结果进行解码
            # ---------------------------------------------------------#
            results = self.bbox_util.detection_out(preds, self.anchors, confidence_threshold=self.confidence)

        t2 = time.time()
        tact_time = (t2 - t1)
        return tact_time
