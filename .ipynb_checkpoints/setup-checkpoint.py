import glob
import subprocess
from setuptools import setup, find_packages, Extension


def build_libs():
    subprocess.call(['cmake', '.'])
    subprocess.call(['make'])
    

build_libs()


setup(
    name='jetbot',
    version='1.0.0',
    description='An open-source robot based on NVIDIA Jetson Nano',
    packages=find_packages(),
    install_requires=[
        'Adafruit_MotorHat',
        'Adafruit-SSD1306',
    ],
    package_data={'jetbot_v1': ['ssd_tensorrt/*.so', 'yolo_tensorrt/*.so']},
)
#        'sparkfun-qwiic'