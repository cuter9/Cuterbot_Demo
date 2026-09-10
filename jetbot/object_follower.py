#!/usr/bin/env python
# coding: utf-8

# # Object Following - Live Demo
# 
# In this notebook we'll show how you can follow an object with JetBot!  We'll use a pre-trained neural network
# that was trained on the [COCO dataset](http://cocodataset.org) to detect 90 different common objects.  These include
# 
# * Person (index 0)
# * Cup (index 47)
# 
# and many others (you can check [this file](https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_complete_label_map.pbtxt) for a full list of class indices).  The model is sourced from the [TensorFlow object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection),
# which provides utilities for training object detectors for custom tasks also!  Once the model is trained, we optimize it using NVIDIA TensorRT on the Jetson Nano.
# 
# This makes the network very fast, capable of real-time execution on Jetson Nano!  We won't run through all of the training and optimization steps in this notebook though.
# 
# Anyways, let's get started.  First, we'll want to import the ``ObjectDetector`` class which takes our pre-trained SSD engine.

# ### Compute detections on single camera image

# In[ ]:

# from queue import Empty
import torch
import torch.nn.functional as F
import torchvision

import cv2
import numpy as np
import traitlets

from jetbot import Camera
from jetbot import Robot
from jetbot import bgr8_to_jpeg
from jetbot.utils import get_cls_dict_yolo, get_cls_dict_ssd
import time
from jetbot import ObjectDetector


# model = ObjectDetector('ssd_mobilenet_v2_coco_onnx.engine')
# model = ObjectDetector_YOLO('yolov4-288.engine')


# class ObjectFollower(traitlets.HasTraits):
def norm(vec):
    """Computes the length of the 2D vector"""
    return np.sqrt(vec[0] ** 2 + vec[1] ** 2)


def object_center_detection(det):
    """Computes the center x, y coordinates of the object"""
    # print(self.matching_detections)
    bbox = det['bbox']
    # print(f"object _detection box: {bbox}")
    center_x = (bbox[0] + bbox[2]) / 2.0 - 0.5
    center_y = (bbox[1] + bbox[3]) / 2.0 - 0.5
    object_center = (center_x, center_y)
    return object_center


class ObjectFollower(ObjectDetector):
    follower_model = traitlets.Unicode(default_value='').tag(config=True)
    type_follower_model = traitlets.Unicode(default_value='').tag(config=True)
    conf_th = traitlets.Float(default_value=0.5).tag(config=True)
    cap_image = traitlets.Any()
    label = traitlets.Integer(default_value=1).tag(config=True)
    label_text = traitlets.Unicode(default_value='').tag(config=True)
    speed_of = traitlets.Float(default_value=0).tag(config=True)
    speed_gain_of = traitlets.Float(default_value=0.15).tag(config=True)
    speed_dev_of = traitlets.Float(default_value=0.5).tag(config=True)
    turn_gain_of = traitlets.Float(default_value=0.3).tag(config=True)
    steering_bias_of = traitlets.Float(default_value=0.0).tag(config=True)
    target_view_of = traitlets.Float(default_value=0.6).tag(config=True)
    mean_view_of = traitlets.Float(default_value=0).tag(config=True)
    e_view_of = traitlets.Float(default_value=0).tag(config=True)
    blocked = traitlets.Float(default_value=0).tag(config=True)
    is_detecting = traitlets.Bool(default_value=True).tag(config=True)

    def __init__(self, init_sensor_of=False):

        super().__init__()
        self.detections = None
        self.matching_detections = None
        self.object_center = None
        self.closest_object = None
        self.speed_of = self.speed_gain_of
        # self.is_detecting = True

        self.robot = None
        self.capturer = None
        if init_sensor_of:
            self.robot = Robot.instance()
            # Camera instance would be better to put after all models instantiation
            self.capturer = Camera()
            self.img_width = self.capturer.width
            self.img_height = self.capturer.height
            self.width_display = self.capturer.width_display
            self.height_display = self.capturer.height_display
            self.cap_image = np.empty(shape=(self.height_display, self.width_display, 3), dtype=np.uint8).tobytes()
            self.current_image = np.empty((self.img_height, self.img_width, 3))

        self.ob_detect_duration_max = 10
        self.ob_detect_count = 0

        self.target_view_of = 0.5
        self.mean_view_of = 0
        self.mean_view_prev_of = 0
        self.e_view_of = 0
        self.e_view_prev_of = 0

        self.execution_time_of = []
        self.enable_of_exec = True
        # self.fps = []

    def load_object_detector(self):

        # self.object_detector = None
        self.engine_path = self.follower_model  # set the engine_path in object detection module
        self.type_model_od = self.type_follower_model  # set the type_model in object detection module

        # avoider_model='../collision_avoidance/best_model.pth'
        # self.obstacle_detector = Avoider(model_params=self.avoider_model)
        print('path of object detector model: %s' % self.follower_model)
        self.load_od_engine()  # load object detection engine function in object detection module

    def run_objects_detection(self):
        # self.image = self.capturer.value
        # print(self.image[1][1], np.shape(self.image))
        self.detections = self.execute_od(self.current_image)  # function in object detection for executing trt engine
        self.matching_detections = [d for d in self.detections[0] if d['label'] == int(self.label)]
        if int(self.label) >= 0:
            if self.type_follower_model == "SSD" or self.type_follower_model == "SSD_FPN":
                self.label_text = get_cls_dict_ssd('coco')[int(self.label)]
            elif self.type_follower_model == "YOLO" or self.type_follower_model == "YOLO_v7":
                self.label_text = get_cls_dict_yolo('coco')[int(self.label)]
        else:
            self.label_text = " Not defined !"

        # print(int(self.label), "\n", self.matching_detections)

    def closest_object_detection(self):
        """Finds the detection closest to the image center"""
        self.closest_object = None
        if len(self.matching_detections) != 0:
            for det in self.matching_detections:
                # if (norm(object_center_detection(det)) < norm(object_center_detection(self.closest_object)) or self.closest_object) is None:
                #    self.closest_object = det
                if self.closest_object is None:
                     self.closest_object = det
                elif norm(object_center_detection(self.closest_object)) < norm(
                        object_center_detection(self.closest_object)):
                    self.closest_object = det
            print(self.closest_object['bbox'])

    def start_of(self):
        self.capturer.unobserve_all()
        self.load_object_detector()

        print("start running!")
        self.capturer.observe(self.execute_of, names='value')

    def execute_of(self, change):
        if not self.enable_of_exec:
            return
        # print("start execution !")
        start_time = time.time()

        self.current_image = change['new']
        # width = self.img_width
        # height = self.img_height

        # print(image)
        # ** execute collision model to determine if blocked
        # turn left if blocked
        if self.blocked > 0.5:
            #      # robot.left(0.3)
            self.robot.left(0.05)
            # self.cap_image = bgr8_to_jpeg(self.current_image)
            self.cap_image = bgr8_to_jpeg(cv2.resize(self.current_image,
                                                     (self.width_display, self.height_display),
                                                     interpolation=cv2.INTER_LINEAR))

            # self.cap_image = bgr8_to_jpeg(self.capturer.image_display)
            return

        # compute all detected objects
        self.run_objects_detection()
        self.closest_object_detection()
        # detections = self.object_detector(image)
        # print(self.detections)

        # self.speed_of = self.speed_gain_of

        # draw all detections on image
        for det in self.detections[0]:
            bbox = det['bbox']
            cv2.rectangle(self.current_image, (int(self.img_width * bbox[0]), int(self.img_height * bbox[1])),
                          (int(self.img_width * bbox[2]), int(self.img_height * bbox[3])), (255, 0, 0), 2)

        # select detections that match selected class label and
        # get detection closest to center of field of view and draw it
        # cls_obj = self.closest_object
        if self.closest_object is not None:
            bbox = self.closest_object['bbox']
            cv2.rectangle(self.current_image, (int(self.img_width * bbox[0]), int(self.img_height * bbox[1])),
                           (int(self.img_width * bbox[2]), int(self.img_height * bbox[3])), (0, 255, 0), 5)
            self.ob_detect_count = self.ob_detect_duration_max

            self.mean_view_of = 0.4 * (bbox[2] - bbox[0]) + 0.6 * self.mean_view_prev_of
            self.e_view_of = self.target_view_of - self.mean_view_of
            if np.abs(self.e_view_of / self.target_view_of) > 0.1:
                self.speed_of = self.speed_of + self.speed_gain_of * self.e_view_of + self.speed_dev_of * (
                        self.e_view_of - self.e_view_prev_of)
            self.mean_view_prev_of = self.mean_view_of
            self.e_view_prev_of = self.e_view_of

        # otherwise go forward or turn arround if no target detected
        # if self.closest_object is None:
        else:
            if self.ob_detect_count <= 0:
                # self.robot.forward(float(self.speed_gain_of))
                self.robot.left(0.2 * float(self.speed_gain_of))
            else:
                self.ob_detect_count -= 1
        # otherwise steer towards target

        # move robot forward and steer proportional target's x-distance from center
        center = object_center_detection(self.closest_object)
        self.robot.set_motors(
            max(min(float(self.speed_of + self.turn_gain_of * center[0] + self.steering_bias_of), 1.0), -1.0),
            max(min(float(self.speed_of - self.turn_gain_of * center[0] + self.steering_bias_of), 1.0), -1.0)
        )

        end_time = time.time()
        # self.execution_time.append(end_time - start_time + self.capturer.cap_time)
        self.execution_time_of.append(end_time - start_time)
        # self.fps.append(1/(end_time - start_time))

        # update image widget
        # image_widget.value = bgr8_to_jpeg(image)
        self.cap_image = bgr8_to_jpeg(cv2.resize(self.current_image,
                                                 (self.width_display, self.height_display),
                                                 interpolation=cv2.INTER_LINEAR))
        # print("ok!")
        # return self.cap_image

    def stop_of(self, change):
        from jetbot.utils import plot_exec_time
        print("stop running!")
        self.capturer.unobserve_all()
        time.sleep(1.0)
        self.robot.stop()
        self.capturer.stop()

        # plot execution time of object follower model processing
        model_name = "object follower model"
        model_name_str = self.follower_model.split('/')[-1].split('.')[0]
        plot_exec_time(self.execution_time_of[1:], model_name, model_name_str)
        # plot_exec_time(self.execution_time[1:], self.fps[1:], model_name, self.follower_model.split('.')[0])
        # plt.show()


class Avoider(object):

    def __init__(self, model_params='../collision_avoidance/best_model.pth'):
        self.model_params = model_params
        self.collision_model = torchvision.models.alexnet(pretrained=False)
        self.collision_model.classifier[6] = torch.nn.Linear(self.collision_model.classifier[6].in_features, 2)
        self.collision_model.load_state_dict(torch.load(self.model_params))
        # collision_model.load_state_dict(torch.load('../collision_avoidance/best_model.pth'))
        self.device = torch.device('cuda')
        self.collision_model = self.collision_model.to(self.device)
        self.prob_blocked = 0

    def detect(self, image):
        collision_output = self.collision_model(self.preprocess(image)).detach().cpu()
        self.prob_blocked = float(F.softmax(collision_output.flatten(), dim=0)[0])
        # blocked_widget.value = prob_blocked

    def preprocess(self, camera_value):
        # global device
        mean = 255.0 * np.array([0.485, 0.456, 0.406])
        stdev = 255.0 * np.array([0.229, 0.224, 0.225])
        normalize = torchvision.transforms.Normalize(mean, stdev)
        x = camera_value
        x = cv2.resize(x, (224, 224))
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = x.transpose((2, 0, 1))
        x = torch.from_numpy(x).float()
        x = normalize(x)
        x = x.to(self.device)
        x = x[None, ...]
        return x
