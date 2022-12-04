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
#ADHSA


class Retinaface(object):
    _defaults = {

        "model_path"        : 'model_data/retinaface_resnet50.h5',
        "backbone"          : 'resnet50',
        "confidence"        : 0.5,
        "nms_iou"           : 0.45,
        "input_shape"       : [1280, 1280, 3],
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

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        #---------------------------------------------------#
        #   对输入图像进行一个备份，后面用于绘图
        #---------------------------------------------------#
        old_image   = image.copy()
        #---------------------------------------------------#
        #   把图像转换成numpy的形式
        #---------------------------------------------------#
        image       = np.array(image, np.float32)
        #---------------------------------------------------#
        #   计算输入图片的高和宽
        #---------------------------------------------------#
        im_height, im_width, _ = np.shape(image)
        #---------------------------------------------------#
        #   计算scale，用于将获得的预测框转换成原图的高宽
        #---------------------------------------------------#
        scale = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
        ]
        scale_for_landmarks = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0]
        ]

        #---------------------------------------------------------#
        #   letterbox_image可以给图像增加灰条，实现不失真的resize
        #---------------------------------------------------------#
        if self.letterbox_image:
            image = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        else:
            self.anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()
        #---------------------------------------------------------#
        #   图片预处理，归一化。
        #---------------------------------------------------------#
        photo   = np.expand_dims(preprocess_input(image), 0)
        
        #---------------------------------------------------------#
        #   传入网络进行预测
        #---------------------------------------------------------#
        preds = self.get_pred(photo)
        preds = [pred.numpy() for pred in preds]
        #-----------------------------------------------------------#
        #   将预测结果进行解码
        #-----------------------------------------------------------#
        results = self.bbox_util.detection_out(preds, self.anchors, confidence_threshold=self.confidence)

        #---------------------------------------------------------#
        #   如果没有检测到物体，则返回原图
        #---------------------------------------------------------#
        if len(results) <= 0:
            return old_image

        results = np.array(results)
        #---------------------------------------------------------#
        #   如果使用了letterbox_image的话，要把灰条的部分去除掉。
        #---------------------------------------------------------#
        if self.letterbox_image:
            results = retinaface_correct_boxes(results, np.array([self.input_shape[0], self.input_shape[1]]), np.array([im_height, im_width]))
        
        results[:, :4] = results[:, :4] * scale
        results[:, 5:] = results[:, 5:] * scale_for_landmarks

        for b in results:
            text    = "{:.4f}".format(b[4])
            b       = list(map(int, b))
            #---------------------------------------------------#
            #   b[0]-b[3]为人脸框的坐标，b[4]为得分
            #---------------------------------------------------#
            cv2.rectangle(old_image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(old_image, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            print(b[0], b[1], b[2], b[3], b[4])
            #---------------------------------------------------#
            #   b[5]-b[14]为人脸关键点的坐标
            #---------------------------------------------------#
            cv2.circle(old_image, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(old_image, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(old_image, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(old_image, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(old_image, (b[13], b[14]), 1, (255, 0, 0), 4)
        return old_image
        
    def get_FPS(self, image, test_interval):
        #---------------------------------------------------#
        #   把图像转换成numpy的形式
        #---------------------------------------------------#
        image       = np.array(image, np.float32)
        #---------------------------------------------------#
        #   计算输入图片的高和宽
        #---------------------------------------------------#
        im_height, im_width, _ = np.shape(image)
        #---------------------------------------------------------#
        #   letterbox_image可以给图像增加灰条，实现不失真的resize
        #---------------------------------------------------------#
        if self.letterbox_image:
            image = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        else:
            self.anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()
        #---------------------------------------------------------#
        #   图片预处理，归一化。
        #---------------------------------------------------------#
        photo   = np.expand_dims(preprocess_input(image), 0)
        
        #---------------------------------------------------------#
        #   传入网络进行预测
        #---------------------------------------------------------#
        preds   = self.get_pred(photo)
        preds   = [pred.numpy() for pred in preds]
        #---------------------------------------------------------#
        #   将预测结果进行解码
        #---------------------------------------------------------#
        results = self.bbox_util.detection_out(preds, self.anchors, confidence_threshold=self.confidence)

        t1 = time.time()
        for _ in range(test_interval):
            #---------------------------------------------------------#
            #   传入网络进行预测
            #---------------------------------------------------------#
            preds   = self.get_pred(photo)
            preds   = [pred.numpy() for pred in preds]
            #---------------------------------------------------------#
            #   将预测结果进行解码
            #---------------------------------------------------------#
            results = self.bbox_util.detection_out(preds, self.anchors, confidence_threshold=self.confidence)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def get_map_txt(self, image):
        #---------------------------------------------------#
        #   把图像转换成numpy的形式
        #---------------------------------------------------#
        image       = np.array(image, np.float32)
        #---------------------------------------------------#
        #   计算输入图片的高和宽
        #---------------------------------------------------#
        im_height, im_width, _ = np.shape(image)
        #---------------------------------------------------#
        #   计算scale，用于将获得的预测框转换成原图的高宽
        #---------------------------------------------------#
        scale = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
        ]
        scale_for_landmarks = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0]
        ]

        #---------------------------------------------------------#
        #   letterbox_image可以给图像增加灰条，实现不失真的resize
        #---------------------------------------------------------#
        if self.letterbox_image:
            image = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        else:
            self.anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()
            
        #---------------------------------------------------------#
        #   图片预处理，归一化。
        #---------------------------------------------------------#
        photo   = np.expand_dims(preprocess_input(image), 0)
        
        #---------------------------------------------------------#
        #   传入网络进行预测
        #---------------------------------------------------------#
        preds   = self.get_pred(photo)
        preds   = [pred.numpy() for pred in preds]

        #---------------------------------------------------------#
        #   将预测结果进行解码
        #---------------------------------------------------------#
        results = self.bbox_util.detection_out(preds, self.anchors, confidence_threshold = self.confidence)

        #--------------------------------------#
        #   如果没有检测到物体，则返回原图
        #--------------------------------------#
        if len(results)<=0:
            return np.array([])

        results = np.array(results)
        #---------------------------------------------------------#
        #   如果使用了letterbox_image的话，要把灰条的部分去除掉。
        #---------------------------------------------------------#
        if self.letterbox_image:
            results = retinaface_correct_boxes(results, np.array([self.input_shape[0], self.input_shape[1]]), np.array([im_height, im_width]))
        
        results[:, :4] = results[:, :4] * scale
        results[:, 5:] = results[:, 5:] * scale_for_landmarks

        return results