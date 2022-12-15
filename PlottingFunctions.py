import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

import torch
from dataset import CIFAR10
from torch.utils.data import DataLoader

def example_img_plotter(img, transformation=None, imgaug=None):
    x=1

    plt.imshow(  img.permute(1, 2, 0)  )

    x=1
#     plt.figure(figsize=(15,5))
#     for jj in range(6):
#         plt.subplot(2,6,jj+1);plt.imshow(data[inds[jj],0].cpu().numpy(),cmap='gray');plt.axis("off");
#         plt.title("clean. pred={}".format(classes[clean_preds[inds[jj]]]))
#     for jj in range(6):
#         plt.subplot(2,6,6+jj+1);plt.imshow(adv_data[inds[jj],0].cpu().numpy(),cmap='gray');plt.axis("off");
#         plt.title("adv. pred={}".format(classes[adv_preds[inds[jj]]]))
#     plt.tight_layout()
#     plt.show()

def set_device():
    # specify the device for computation
    #############################################
    # your code here
    device = 'cpu'
    if device =='cuda':
        print("Run on GPU...")
    else:
        print("Run on CPU...")

    return device


def main():
    #load data
    #############################################
    # your code here
    # specify preprocessing function

    channel_means =  (0.4914, 0.4822, 0.4465)
    channel_std = (0.2023, 0.1994, 0.2010)

    #
    # 1) First we conver the PIL imags to a Tensor, so that our PyTorch functions can process the data.
    # 2) We Normalize the data to make it easeir for the neural network layers to learn
    # 3) Additionally, only for the training set we augment to provide more data for the network to learn
    # from, and to prevent the network from memorizing the data by regularizing it. The augmented diversity
    # helps prevent the network from seeing the exact same set of images repeately during training.
    #
    transform_train=transforms.Compose([
        transforms.ToTensor(),
        torchvision.transforms.RandomCrop([32,32], padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        # torchvision.transforms.RandomErasing(),
        # torchvision.transforms.RandomAdjustSharpness(sharpness_factor=2),
        # torchvision.transforms.RandomAffine((30,70)), # reduced performance
        # torchvision.transforms.RandomAutocontrast(), # did not improve performance
        # torchvision.transforms.RandomGrayscale(), # did not improve performance
        # torchvision.transforms.RandomInvert(),    # reduced performance
        # torchvision.transforms.GaussianBlur(3),   # did not improve performance
        transforms.Normalize(mean=channel_means, std=channel_std)
    ])

    transform_val =transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=channel_means, std=channel_std)
    ])
    #############################################



    # a few arguments, do NOT change these
    DATA_ROOT = "./data"
    TRAIN_BATCH_SIZE = 128
    VAL_BATCH_SIZE = 100

    #############################################
    # your code here
    # construct dataset
    train_set = CIFAR10(
        root=DATA_ROOT,
        mode='train',
        download=True,
        transform=transform_train    # your code
    )

    # construct dataloader
    train_loader = DataLoader(
        train_set,
        batch_size=TRAIN_BATCH_SIZE ,  # your code
        shuffle=True,     # your code
        num_workers=4
    )

    device = set_device()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        ####################################
        for i in range(0, 128):
            example_img_plotter(inputs[i, :,:,:], transformation=None, imgaug=None)
        x=1
        inputs = inputs.to(device)
        targets = targets.to(device)


if __name__ == "__main__":
    main()
