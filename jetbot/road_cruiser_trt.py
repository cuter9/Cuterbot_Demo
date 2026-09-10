import time

import torch

import PIL.Image
import cv2

import numpy as np

import traitlets
from traitlets import HasTraits, Unicode, Float
from torch2trt import TRTModule

from jetbot import Camera, bgr8_to_jpeg
from jetbot import Robot

from jetbot.utils.model_selection import ClassifierPreprocessV1


class RoadCruiserTRT(HasTraits):
    cruiser_model = Unicode(default_value='').tag(config=True)
    type_cruiser_model = Unicode(default_value='').tag(config=True)
    cruiser_model_preprocess = Unicode(default_value='').tag(config=True)
    speed_rc = Float(default_value=0).tag(config=True)
    speed_gain_rc = Float(default_value=0.15).tag(config=True)
    steering_gain_rc = Float(default_value=0.08).tag(config=True)
    steering_dgain_rc = Float(default_value=1.5).tag(config=True)
    steering_bias_rc = Float(default_value=0.0).tag(config=True)
    steering_rc = Float(default_value=0.0).tag(config=True)
    x_slider = Float(default_value=0).tag(config=True)
    y_slider = Float(default_value=0).tag(config=True)
    cap_image = traitlets.Any()

    def __init__(self, init_sensor_rc=False):
        super().__init__()

        self.trt_model_rc = TRTModule()
        self.preprocess = None

        self.robot = None
        self.capturer = None

        if init_sensor_rc:
            self.capturer = Camera()
            self.robot = Robot.instance()
            self.width_display = self.capturer.width_display
            self.height_display = self.capturer.height_display
            self.cap_image = np.empty(shape=(self.height_display, self.width_display, 3), dtype=np.uint8).tobytes()

        self.angle = 0.0
        self.angle_last = 0.0
        self.execution_time = []
        # self.fps = []
        self.x_slider = 0
        self.y_slider = 0
        self.speed_rc = self.speed_gain_rc

        self.device = torch.device('cuda')
        self.execution_time_rc = []
        self.enable_rc_exec = True

    # ---- Creating the Pre-Processing Function
    # 1. Convert from HWC layout to CHW layout
    # 2. Normalize using same parameters as we did during training (our camera provides values in [0, 255] range and training loaded images in [0, 1] range so we need to scale by 255.0
    # 3. Transfer the data from CPU memory to GPU memory
    # 4. Add a batch dimension
    def load_road_cruiser(self):

        print('path of cruiser model: %s' % self.cruiser_model)

        if "workspace" in self.cruiser_model:
            self.trt_model_rc.load_state_dict(torch.load(self.cruiser_model))
            # load preprocess for loaded cruiser model
            # self.preprocess = tv_classifier_preprocesss()
            # use weights_only=True, ref: https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models
            # self.preprocess.load_state_dict(torch.load(self.cruiser_model_preprocess))
            model_config = torch.load(self.cruiser_model_preprocess)
            self.preprocess = ClassifierPreprocessV1(model_config)
            # self.preprocess.to(self.device).eval().half()

        else:
            self.trt_model_rc.load_state_dict(torch.load('best_steering_model_xy_trt_' + self.cruiser_model + '.pth'))

        print("engine is built from pytorch model!")

    def preprocess_rc(self, image):
        image = PIL.Image.fromarray(image)
        image = self.preprocess(image, is_training=False).to(self.device).half()
        return image[None, ...]

    def execute_rc(self, change):
        if not self.enable_rc_exec:
            return

        start_time = time.time()
        image = change['new']
        self.cap_image = bgr8_to_jpeg(cv2.resize(image,
                                                 (self.width_display, self.height_display),
                                                 interpolation=cv2.INTER_LINEAR))

        xy = self.trt_model_rc(self.preprocess_rc(image)).detach().float().cpu().numpy().flatten()
        x = xy[0]           #  the range of x: -1(left) ~ +1(right)
        # y = (0.5 - xy[1]) / 2.0   # This is suitable for the image window without referring to central line
        y = (1 - xy[1])     # the range of y: 2(up) ~ 0(down) , This is suitable for the y data around 0, i.e. the central line is at the middle of image

        self.x_slider = x.item()
        self.y_slider = y.item()

        self.speed_rc = self.speed_gain_rc

        # angle = np.sqrt(xy)*np.arctan2(x, y)
        angle_1 = np.arctan2(x, y)                          # -0.5*pi ~ 0.5*pi
        self.angle = 0.5 * np.pi * np.tanh(0.5 * angle_1)   # -0.5*pi*0.22 ~ 0.5*pi*0.22
        pid = self.angle * self.steering_gain_rc + (self.angle - self.angle_last) * self.steering_dgain_rc
        self.angle_last = self.angle

        self.steering_rc = pid + self.steering_bias_rc

        self.robot.left_motor.value = max(min(self.speed_gain_rc + self.steering_rc, 1.0), 0.0)
        self.robot.right_motor.value = max(min(self.speed_gain_rc - self.steering_rc, 1.0), 0.0)

        end_time = time.time()
        # self.execution_time.append(end_time - start_time + self.camera.cap_time)
        self.execution_time_rc.append(end_time - start_time)
        # self.fps.append(1/(end_time - start_time))


    # We accomplish that with the observe function.
    def start_rc(self, change):
        # self.capturer.unobserve_all()
        # self.execute({'new': self.camera.value})
        self.load_road_cruiser()
        print("start running!")
        self.capturer.observe(self.execute_rc, names='value')

    def stop_rc(self, change):
        from jetbot.utils import plot_exec_time
        self.capturer.unobserve_all()
        print("start stopping!")
        time.sleep(1.0)
        self.robot.stop()
        self.capturer.stop()

        # plot execution time of road cruiser model processing
        model_name = 'road cruiser model'
        cruiser_model_name = self.cruiser_model.split("/")[-1].split('.')[0]
        plot_exec_time(self.execution_time_rc[1:], model_name, cruiser_model_name)
        # plt.show()
