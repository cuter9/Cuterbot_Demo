import pandas as pd
import os
from traitlets import HasTraits, Unicode, List, Bool
# import numpy as np

import torch
import torchvision.models as pth_models

HEAD_LIST = ['model_function', 'model_type', 'model_path']
MODEL_REPO_DIR = os.path.join(os.environ["HOME"], "model_repo")
MODEL_REPO_DIR_DOCKER = os.path.join("/workspace", "model_repo")
os.environ['MODEL_REPO_DIR_DOCKER'] = MODEL_REPO_DIR_DOCKER
os.environ['MODEL_REPO_DIR'] = MODEL_REPO_DIR


def load_tune_pth_model(pth_model_name="resnet18", pretrained=True):
    model_type = None
    # ----- modify the last layer for classification, and the model used in notebook should be modified too.
    model = getattr(pth_models, pth_model_name)(pretrained=pretrained)  # for fine-tuning

    if 'mobilenet_v3' in pth_model_name: # 'mobilenet_v3_large' or  'mobilenet_v3_small'
        model_type = "MobileNet"

    elif pth_model_name == 'mobilenet_v2':
        model_type = "MobileNet"

    elif pth_model_name == 'vgg11':  # VGGNet
        model_type = "VggNet"

    elif 'resnet' in pth_model_name:  # ResNet
        model_type = "ResNet"

    elif 'efficientnet' in pth_model_name:  # ResNet
        model_type = "EfficientNet"

    elif pth_model_name == 'inception_v3':  # Inception_v3
        model_type = "InceptionNet"

    elif pth_model_name == 'googlenet':  # Inception_v3
        model_type = "GoogleNet"

    elif "densenet" in pth_model_name:  # densenet121, densenet161, densenet169, densenet201
        model_type = "DenseNet"

    elif "shufflenet_v2" in pth_model_name:  # shufflenet_v2_x1_0 or shufflenet_v2_x0_5
        model_type = "ShuffleNet"

    elif "mnasnet" in pth_model_name:  # mnasnet1_0 or mnasnet0_5
        model_type = "MnasNet"

    else:
        assert (model is not None and model_type is not None), "Check if the model name set is compatible with torchvision."

    return model, model_type


class model_selection(HasTraits):
    model_function = Unicode(default_value='object detection').tag(config=True)
    model_function_list = List(default_value=[]).tag(config=True)
    model_type = Unicode(default_value='SSD').tag(config=True)
    model_type_list = List(default_value=[]).tag(config=True)
    model_path = Unicode(default_value='').tag(config=True)
    model_path_list = List(default_value=[]).tag(config=True)
    selected_model_path = Unicode(default_value='').tag(config=True)
    is_selected = Bool(default_value=False).tag(config=True)

    def __init__(self, core_library='TensorRT', dir_model_repo=MODEL_REPO_DIR_DOCKER):
        super().__init__()

        self.core_library = core_library
        if self.core_library == 'TensorRT':
            self.df = pd.read_csv(os.path.join(dir_model_repo, "trt_model_tbl.csv"),
                                  header=None, names=HEAD_LIST)
        elif self.core_library == 'Pytorch':
            self.df = pd.read_csv(os.path.join(dir_model_repo, "torch_model_tbl.csv"),
                                  header=None, names=HEAD_LIST)

        for p in self.df.values:
            p[2] = os.path.join(dir_model_repo, p[2].split("/", 1)[1])

        self.model_function_list = list(self.df["model_function"].astype("category").cat.categories)
        self.update_model_type_list()
        # d_mf = self.df[self.df.model_function == self.model_function]   # data frame of given function
        # self.model_type_list = list(d_mf["model_type"].astype("category").cat.categories)
        self.update_model_list()
        # mpl = d_mf[d_mf.model_type == self.model_type].loc[:, ['model_path']].values.tolist()
        # self.model_path_list = np.squeeze(mpl).tolist()
        self.observe(self.update_model, names=['model_function', 'model_type', 'model_path'])
        # self.is_selected = False
        # self.observe(self.selected, names=['is_selected'])

    def update_model_type_list(self):
        mf = self.df[self.df.model_function == self.model_function]  # select the models based on given model function
        # mt = mf[mf.model_type == self.model_type]
        self.model_type_list = list(
            mf["model_type"].astype("category").cat.categories)  # the model types of the given model function
        return self.model_type_list

    def update_model_list(self):
        mf = self.df[self.df.model_function == self.model_function]  # select the models based on given model function
        mt = mf[mf.model_type == self.model_type]  # select the models from the given model type
        mpl = mt.loc[:, ['model_path']].values
        # self.model_path_list = np.squeeze(mpl).tolist()
        self.model_path_list = mpl[:, 0].tolist()
        return self.model_path_list

    def update_model(self, change):
        # print(change)
        if change['name'] == 'model_function':
            self.model_function = change['new']
            self.update_model_type_list()
        if change['name'] == 'model_type':
            self.model_type = change['new']
            self.update_model_list()
        if change['name'] == 'model_path':
            self.model_path = change['new']
            # self.selected_model_path = os.path.join(MODEL_REPO_DIR_DOCKER, self.model_path.split("/", 1)[1])
        # print(self.selected_model_path)

    # def selected(self, change):
    #     self.is_selected = change['new']


'''
ms = trt_model_selection()
# ms.model_function = 'object detection'
# ms.model_type = 'SSD_FPN'
# model_type_list = ms.update_model_type_list()
model_path_list = ms.update_model_list()
print(ms.model_function_list, ms.model_type_list)
print(model_path_list)
'''