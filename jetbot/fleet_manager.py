from jetbot import Camera
from jetbot import Robot
from jetbot import bgr8_to_jpeg
from jetbot import RoadCruiser
from jetbot import ObjectFollower
# from jetbot.utils import get_cls_dict_yolo, get_cls_dict_ssd
from traitlets import Float, Bool, Any

import cv2
import numpy as np
# import traitlets

import time

def object_center_detection(det):
    """Computes the center x, y coordinates of the object"""
    # print(self.matching_detections)
    bbox = det['bbox']
    center_x = (bbox[0] + bbox[2]) / 2.0 - 0.5
    center_y = (bbox[1] + bbox[3]) / 2.0 - 0.5
    object_center = (center_x, center_y)
    return object_center


def norm(vec):
    """Computes the length of the 2D vector"""
    return np.sqrt(vec[0] ** 2 + vec[1] ** 2)


class Fleeter(ObjectFollower, RoadCruiser):
    cap_image = Any()
    conf_th = Float(default_value=0.5).tag(config=True)
    # label = Integer(default_value=1).tag(config=True)
    # label_text = Unicode(default_value='').tag(config=True)
    speed_fm = Float(default_value=0.10).tag(config=True)
    speed_gain_fm = Float(default_value=0.01).tag(config=True)
    speed_dev_fm = Float(default_value=0.5).tag(config=True)
    turn_gain_fm = Float(default_value=0.3).tag(config=True)
    steering_bias_fm = Float(default_value=0.0).tag(config=True)
    # blocked = Float(default_value=0).tag(config=True)
    target_view = Float(default_value=0.6).tag(config=True)
    mean_view = Float(default_value=0).tag(config=True)
    e_view = Float(default_value=0).tag(config=True)
    # is_detecting = Bool(default_value=True).tag(config=True)
    is_detected = Bool(default_value=False).tag(config=True)

    def __init__(self, init_sensor_fm=False):

        # causing the  re-instantiation error of camera and robot motor,
        # which should be avoided when design the parent classes
        ObjectFollower.__init__(self, init_sensor_of=False)
        RoadCruiser.__init__(self, init_sensor_rc=False)

        self.detections = None
        self.matching_detections = None
        self.object_center = None
        self.closest_object = None
        self.is_detecting = True
        self.is_detected = False
        self.is_loaded = False

        self.robot = None
        self.capturer = None
        if init_sensor_fm:
            self.robot = Robot.instance()
            self.capturer = Camera()
            self.img_width = self.capturer.width
            self.img_height = self.capturer.height
            self.width_display = self.capturer.width_display
            self.height_display = self.capturer.height_display
            self.cap_image = np.empty(shape=(self.height_display, self.width_display, 3), dtype=np.uint8).tobytes()
            self.current_image = np.empty((self.img_height, self.img_width, 3))

        self.default_speed = self.speed_fm
        self.detect_duration_max = 10
        self.no_detect = 0
        self.target_view = 0.5
        self.mean_view = 0
        self.mean_view_prev = 0
        self.e_view = 0
        self.e_view_prev = 0

        self.enable_fm_exec = True
        self.execution_time_fm = []
        # self.fps = []

    def execute_fm(self, change):
        if not self.enable_fm_exec:
            return
        # do object following
        start_time = time.time()
        self.execute(change)
        end_time = time.time()
        # self.execution_time.append(end_time - start_time + self.capturer.cap_time)
        self.execution_time_fm.append(end_time - start_time)
        # self.fps.append(1/(end_time - start_time))

        if not self.is_detected:
            self.speed_fm = self.speed_rc  # set fleet mge speed to road cruising speed (self.speed)
            self.enable_rc_exec = True
        else:
            self.enable_rc_exec =False

        # if closest object is not detected and followed, do road cruising
        # if not self.is_detected:
        #    self.execute_rc(change)
        #    self.speed_fm = self.speed_rc

    def start_fm(self):
        self.load_object_detector()  # load object detector function in object follower module
        self.load_road_cruiser()  # load_road_cruiser function in road_cruiser_trt module
        self.capturer.unobserve_all()

        print("start running")
        self.capturer.observe(self.execute_fm, names='value')
        self.capturer.observe(self.execute_rc, names='value')

    def execute(self, change):
        # print("start execution !")
        self.current_image = change['new']

        # compute all detected objects
        self.run_objects_detection()
        self.closest_object_detection()
        # detections = self.object_detector(image)
        # print(self.detections)

        # draw all detections on image
        for det in self.detections[0]:
            bbox = det['bbox']
            cv2.rectangle(self.current_image, (int(self.img_width * bbox[0]), int(self.img_height * bbox[1])),
                          (int(self.img_width * bbox[2]), int(self.img_height * bbox[3])), (255, 0, 0), 2)

        # select detections that match selected class label
        # get detection closest to center of field of view and draw it
        cls_obj = self.closest_object
        if cls_obj is not None:
            self.is_detected = True
            self.no_detect = self.detect_duration_max  # set max detection no to prevent temperary loss of object detection
            bbox = cls_obj['bbox']
            cv2.rectangle(self.current_image, (int(self.img_width * bbox[0]), int(self.img_height * bbox[1])),
                          (int(self.img_width * bbox[2]), int(self.img_height * bbox[3])), (0, 255, 0), 5)

            self.mean_view = 0.4 * (bbox[3] - bbox[1]) + 0.6 * self.mean_view_prev
            self.e_view = self.target_view - self.mean_view

            if np.abs(self.e_view / self.target_view) > 0.1:
                self.speed_fm = self.speed_fm + self.speed_gain_fm * self.e_view + self.speed_dev_fm * (
                        self.e_view - self.e_view_prev)
            # self.speed_rc = self.speed_fm

            self.mean_view_prev = self.mean_view
            self.e_view_prev = self.e_view

        # otherwise go forward if no target detected
        if cls_obj is None:
            if self.no_detect <= 0:  # if object is not detected for a duration, road cruising
                self.mean_view = 0.0
                self.mean_view_prev = 0.0
                self.is_detected = False
                self.cap_image = bgr8_to_jpeg(cv2.resize(self.current_image,
                                                         (self.width_display, self.height_display),
                                                         interpolation=cv2.INTER_LINEAR))
                # self.cap_image = bgr8_to_jpeg(self.current_image)
                return

            else:
                self.no_detect -= 1  # observe for a duration for the miss of object detection
            # self.robot.forward(float(self.speed))

        # otherwise steer towards target
        else:
            # move robot forward and steer proportional target's x-distance from center
            center = object_center_detection(cls_obj)
            self.robot.set_motors(
                float(self.speed_fm + self.turn_gain_fm * center[0] + self.steering_bias_fm),
                float(self.speed_fm - self.turn_gain_fm * center[0] + self.steering_bias_fm)
            )

        # update image widget
        self.cap_image = bgr8_to_jpeg(cv2.resize(self.current_image,
                                                 (self.width_display, self.height_display),
                                                 interpolation=cv2.INTER_LINEAR))
        # self.cap_image = bgr8_to_jpeg(self.current_image)

        # print("ok!")
        # return self.cap_image

    def stop_fm(self):
        from jetbot.utils import plot_exec_time
        print("start stopping!")

        self.capturer.unobserve_all()
        time.sleep(1.0)
        self.robot.stop()
        self.capturer.stop()

        # self.road_cruiser.stop_cruising(change)
        # plot execution time of road cruiser model processing
        cruiser_model_name = "road cruiser model"
        cruiser_model_str = self.cruiser_model.split("/")[-1].split('.')[0]
        plot_exec_time(self.execution_time_rc[1:], cruiser_model_name, cruiser_model_str)

        # plot execution time of fleet controller model processing
        follower_model_name = "fleet controller model"
        follower_model_str = self.follower_model.split("/")[-1].split(".")[0]
        plot_exec_time(self.execution_time_fm[1:], follower_model_name, follower_model_str)
        # plot_exec_time(self.execution_time[1:], self.fps[1:], model_name, self.follower_model.split(".")[0])
        # plt.show()
