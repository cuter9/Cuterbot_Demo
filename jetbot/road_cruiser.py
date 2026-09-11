import time

import torch

import PIL.Image
import cv2

import numpy as np
import torchvision

from traitlets import HasTraits, Float, Unicode
import traitlets

from jetbot import Camera, bgr8_to_jpeg
from jetbot import Robot
from jetbot.utils.model_selection import load_model, ClassifierPreprocessV1

class RoadCruiser(HasTraits):
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
    use_gpu = Unicode(default_value='gpu').tag(config=True)
    cap_image = traitlets.Any()

    def __init__(self, init_sensor_rc=False):
        super().__init__()

        self.cruiser_model_type_pth = None
        self.cruiser_model_pth = None
        self.preprocess = None
        self.cruiser_model_preprocess_pth = None

        if init_sensor_rc:
            self.capturer = Camera()
            self.robot = Robot.instance()
            self.width_display = self.capturer.width_display
            self.height_display = self.capturer.height_display
            self.cap_image = np.empty(shape=(self.height_display, self.width_display, 3), dtype=np.uint8).tobytes()

        # self.robot = Robot()
        self.angle = 0.0
        self.angle_last = 0.0
        # self.fps = []
        self.x_slider = 0
        self.y_slider = 0

        self.enable_rc_exec = True
        self.execution_time_rc = []
        self.observe(self.select_gpu, names=['use_gpu'])
        self.device = None
        self.is_loaded = True

    def load_road_cruiser(self):
        pth_model_name = self.cruiser_model.split('/')[-1].split('.')[0].split('_', 4)[-1].split('-')[0]
        print('pytorch model name: %s' % pth_model_name)
        self.cruiser_model_pth, self.cruiser_model_type_pth, self.cruiser_model_preprocess_pth = load_model(
            pth_model_name=pth_model_name,
            pretrained=False)
        if self.cruiser_model_pth is None:
            self.is_loaded = False
            print(
                f"{pth_model_name} is not available in the current torchvision version {torchvision.__version__}")
            return

        print('path of cruiser model: %s' % self.cruiser_model)
        print('use %s for inference.' % self.use_gpu)
        # self.cruiser_model.load_state_dict(torch.load('best_steering_model_xy_' + cruiser_model + '.pth'))
        self.cruiser_model_pth.load_state_dict(torch.load(self.cruiser_model))
        self.cruiser_model_preprocess_pth = torch.load(self.cruiser_model_preprocess)

        model_config = self.cruiser_model_preprocess_pth
        self.preprocess = ClassifierPreprocessV1(model_config)

        if self.use_gpu == 'gpu':
            print("torch cuda version : ", torch.version.cuda)
            print("cuda is available for pytorch: ", torch.cuda.is_available())
            self.device = torch.device('cuda')
            self.cruiser_model_pth.to(self.device)
            self.cruiser_model_pth.eval().half()

        elif self.use_gpu == 'cpu':
            self.device = torch.device('cpu')
            self.cruiser_model_pth.to(self.device)
            self.cruiser_model_pth.eval()

    def select_gpu(self, change):
        self.use_gpu = change['new']

    # ---- Creating the Pre-Processing Function
    # 1. Convert from HWC layout to CHW layout
    # 2. Normalize using same parameters as we did during training (our camera provides values in [0, 255] range and training loaded images in [0, 1] range so we need to scale by 255.0
    # 3. Transfer the data from CPU memory to GPU memory
    # 4. Add a batch dimension

    def preprocess_rc(self, image):
        # tv = int(torchvision.__version__.split(".")[1])  # torchvision version
        image = PIL.Image.fromarray(image)

        # "v1" for torchvision transform v1
        if self.use_gpu == 'gpu':
            image = self.preprocess(image, is_training=False).to(self.device).half()
        elif self.use_gpu == 'cpu':
            image = self.preprocess(image, is_training=False).to(self.device)

        return image[None, ...]

    def execute_rc(self, change):
        if not self.enable_rc_exec:
            return

        start_time = time.time()
        # global angle, angle_last
        image = change['new']
        self.cap_image = bgr8_to_jpeg(cv2.resize(image,
                                                 (self.width_display, self.height_display),
                                                 interpolation=cv2.INTER_LINEAR))

        xy = self.cruiser_model_pth(self.preprocess_rc(image)).detach().float().cpu().numpy().flatten()
        x = xy[0]           #  the range of x: -1(left) ~ +1(right)
        # y = (0.5 - xy[1]) / 2.0  # This is suitable for the image window without referring to the central line
        y = (1 - xy[1])     # range of y : 2(up) ~ 0(down). This is suitable for the y data around 0, i.e. the central line is at the middle of image

        self.x_slider = x.item()
        self.y_slider = y.item()

        self.speed_rc = self.speed_gain_rc

        # angle = np.sqrt(xy)*np.arctan2(x, y)
        angle_1 = np.arctan2(x, y)                              # -0.5*pi ~ 0.5*pi
        self.angle = 0.5 * np.pi * np.tanh(0.5 * angle_1)       # -0.5*pi*0.22 ~ 0.5*pi*0.22
        pid = self.angle * self.steering_gain_rc + (self.angle - self.angle_last) * self.steering_dgain_rc
        self.angle_last = self.angle

        self.steering_rc = pid + self.steering_bias_rc

        # the speed limit is set by alpha value of Motor Class in robot.py to 0.8
        self.robot.left_motor.value = max(min(self.speed_gain_rc + self.steering_rc, 1.0), 0.0)
        self.robot.right_motor.value = max(min(self.speed_gain_rc - self.steering_rc, 1.0), 0.0)

        end_time = time.time()
        self.execution_time_rc.append(end_time - start_time)
        # self.fps.append(1/(end_time - start_time))

    # We accomplish that with the observe function.
    def start_rc(self):
        # self.execute({'new': self.camera.value})
        self.load_road_cruiser()
        if self.is_loaded:
            print("start running!")
            self.capturer.observe(self.execute_rc, names='value')
        else:
            print("The model can not be loaded, start stopping!")
            self.stop_rc()

    def stop_rc(self):
        from jetbot.utils import plot_exec_time
        print("start stopping!")
        # self.camera.unobserve(self.execute, names='value')
        self.capturer.unobserve_all()
        time.sleep(1.0)
        self.robot.stop()
        self.capturer.stop()

        # plot execution time of road cruiser model processing
        model_name = "road cruiser model"
        cruiser_model_str = self.cruiser_model.split("/")[-1].split('.')[0]
        plot_exec_time(self.execution_time_rc[1:], model_name, cruiser_model_str)
        # plt.show()
