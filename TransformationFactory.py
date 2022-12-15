import warnings

import torchvision
import torchvision.transforms as transforms
import torch

class TransformationFactory(object):
    def __init__(self):
        self.CIFAR10_channel_means =  (0.4914, 0.4822, 0.4465)
        self.CIFAR10_channel_std = (0.2023, 0.1994, 0.2010)
        self.CIFAR10_randomCropSettings = {
            "size": [32,32],
            "padding": 4
        }

    def get_standard_training_CIFAR10_compostion(self):
        transformation_composition=transforms.Compose([
            transforms.ToTensor(),
            torchvision.transforms.RandomCrop(self.CIFAR10_randomCropSettings["size"], padding=self.CIFAR10_randomCropSettings["padding"]),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=self.CIFAR10_channel_means , std=self.CIFAR10_channel_std)
        ])

        return transformation_composition

    def get_standard_test_validation_test_CIFAR10_compostion(self):
        transformation_composition=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.CIFAR10_channel_means , std=self.CIFAR10_channel_std)
        ])
        return transformation_composition

    def get_standard_CIFAR10_composition(self, train_test_validation):
        if train_test_validation == "train":
            return self.get_standard_training_CIFAR10_compostion()

        return self.get_standard_test_validation_test_CIFAR10_compostion()

    def get_non_standard_training_CIFAR10_compostion(self, t_form, p):
        transformation_composition=transforms.Compose([
            transforms.ToTensor(),
            torchvision.transforms.RandomCrop(self.CIFAR10_randomCropSettings["size"], padding=self.CIFAR10_randomCropSettings["padding"]),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomApply(transforms = t_form, p=p),
            transforms.Normalize(mean=self.CIFAR10_channel_means , std=self.CIFAR10_channel_std)
        ])

        return transformation_composition

    def get_non_standard_test_validation_test_CIFAR10_compostion(self, t_form, p):
        transformation_composition=transforms.Compose([
            transforms.ToTensor(),
            torchvision.transforms.RandomApply(transforms = t_form, p=p),
            transforms.Normalize(mean=self.CIFAR10_channel_means , std=self.CIFAR10_channel_std)
        ])
        return transformation_composition

    def get_gaussian_blur_CIFAR10_composition(self, intensity, p:float=0.5, train_test_validation: str="train"):

        #intensity==1
        kernel_size=3
        sigma=1
        if intensity==2:
            kernel_size=5
            sigma=1.25
        elif intensity==3:
            kernel_size=7
            sigma=1.5
        elif intensity==4:
            kernel_size=9
            sigma=1.75
        elif intensity==5:
            kernel_size=11
            sigma=2

        t_form = torch.nn.ModuleList([torchvision.transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)])

        if train_test_validation == "train":
            return self.get_non_standard_training_CIFAR10_compostion(t_form, p)

        #else if test or validation
        return self.get_non_standard_test_validation_test_CIFAR10_compostion(t_form, p)


    def get_random_perspective_CIFAR10_composition(self, intensity, p:float=0.5, train_test_validation: str="train"):

        #intensity==1
        distortion_scale = 0.1
        if intensity==2:
            distortion_scale = 0.25
        elif intensity==3:
            distortion_scale = 0.5
        elif intensity==4:
            distortion_scale = 0.75
        elif intensity==5:
            distortion_scale = 0.99

        t_form = torch.nn.ModuleList([torchvision.transforms.RandomPerspective(distortion_scale=distortion_scale, p=1)])

        if train_test_validation == "train":
            return self.get_non_standard_training_CIFAR10_compostion(t_form, p)

        #else if test or validation
        return self.get_non_standard_test_validation_test_CIFAR10_compostion(t_form, p)


    def get_random_resize_and_crop_CIFAR10_composition(self, intensity, p:float=0.5, train_test_validation: str="train"):

        #intensity==1
        scale=(0.85, 0.85)
        if intensity==2:
            scale=(0.7, 0.7)
        elif intensity==3:
            scale=(0.55, 0.55)
        elif intensity==4:
            scale=(0.40, 0.40)
        elif intensity==5:
            scale=(0.25, 0.25)

        t_form = torch.nn.ModuleList([torchvision.transforms.RandomResizedCrop(size=(32,32), scale=scale, ratio=(0.75, 1.3333333333333333))])

        if train_test_validation == "train":
            return self.get_non_standard_training_CIFAR10_compostion(t_form, p)

        #else if test or validation
        return self.get_non_standard_test_validation_test_CIFAR10_compostion(t_form, p)

    def get_transformation_composition_by_name(self, transformation_set: str = "", intensity: int = 0, p:float=0.5, train_test_validation: str = "train"):

        if not ((train_test_validation == "train") | (train_test_validation == "test") | (train_test_validation == "val") | (train_test_validation == "validation")):
            raise ValueError(f'TransformationFactory argument value for train_test_validation is not acceptable, must be train, test, val, or validation: and got {train_test_validation}')

        if transformation_set == "gaussian-blur":
            return self.get_gaussian_blur_CIFAR10_composition(intensity,p, train_test_validation)
        elif transformation_set == "random-perspective":
            return self.get_random_perspective_CIFAR10_composition(intensity,p, train_test_validation)
        elif transformation_set == "random-resize-and-crop":
            return self.get_random_resize_and_crop_CIFAR10_composition(intensity,p, train_test_validation)
        if (transformation_set != "standard") | (transformation_set is not None):
            warnings.warn("TransformationsFactory not supplied with an expected transformation set name, so standard set supplied.")

        return self.get_standard_CIFAR10_composition(train_test_validation)