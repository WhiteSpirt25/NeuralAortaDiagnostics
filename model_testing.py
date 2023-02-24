import argparse
from importlib import import_module
import json
import os
import pickle
import random
from types import SimpleNamespace


import albumentations as album
import numpy as np
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.nn.modules.loss import CrossEntropyLoss

from aortic_dataset import SparseDatasetNpz


def read_config():
    """Simple function to read file config.json to dict."""
    with open("config.json") as jsonfile:
        # `json.loads` parses a string in json format
        return json.load(jsonfile)


def seed_everything(seed: int):
    """Function for seeding random value generators in python, numpy and torch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def save_model_with_stats(model, train_logs, validation_logs, model_name):
    """Saves model(.pth) and it's learning stats(pickle'd lists).

    TODO: Upgrade to some learning tracking system(MLflow, WandB etc.).
    Or at-least save learning data as .csv or other format readable otside of
    python.
    """
    model_dir = os.path.join(config.SAVE_DIR, model_name)
    torch.save(model, os.path.join(model_dir, f"{model_name}.pth"))

    with open(os.path.join(model_dir, "train_logs.pickle"), "wb") as file:
        pickle.dump(train_logs, file)

    with open(os.path.join(model_dir, "validation_logs.pickle"), "wb") as file:
        pickle.dump(validation_logs, file)


def to_tensor(x, **kwargs):
    return TF.to_tensor(x)


def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """
    if preprocessing_fn:
        _transform = [
            album.Lambda(image=preprocessing_fn),
            album.Lambda(image=to_tensor, mask=to_tensor),
        ]
    else:
        _transform = [
            album.Lambda(image=to_tensor, mask=to_tensor),
        ]
    return album.Compose(_transform)


# Loss from TransUnet repository
# Removed one-hot-encoding and softmax to suit dataset
class DiceLoss(torch.nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            # inputs = torch.softmax(inputs, dim=1)
            inputs = torch.sigmoid(inputs)  # since labels not softmax
        # target = self._one_hot_encoder(target) # dataset already one-hot enc.
        if weight is None:
            weight = [1] * self.n_classes
        assert (
            inputs.size() == target.size()
        ), "predict {} & target {} shape do not match".format(
            inputs.size(), target.size()
        )
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class DiceBCELoss(torch.nn.Module):
    def __init__(self, n_classes, weight=None, size_average=True, normalizing=True):
        super(DiceBCELoss, self).__init__()
        self.__name__ = "DiceBCELoss"
        self.dl = DiceLoss(n_classes)
        self.ce = CrossEntropyLoss()
        self.normalizing = normalizing

    def forward(self, inputs, targets, smooth=1e-5):

        # CrossEntropyLoss
        CE = self.ce(inputs, targets)

        dice_loss = self.dl(inputs, targets, softmax=self.normalizing)

        Dice_BCE = 0.5 * CE + 0.5 * dice_loss

        return Dice_BCE


def run_test(model, datasets: dict):
    """Learn function with train-val loop.

    Args:
        model -- any PyTorch segmentation model

        datasets -- dictionary, with 2 PyTorch data.Dataset classes used for training.
            Required keys in dictionary are 'train' and 'validation'.

    Return:
        model -- trained model

        train_logs_list -- list of metric values recorded during training epochs

        valid_logs_list -- list of metric values recorded during validation epochs

        best_loss -- best observed validation loss
    """
    seed_everything(config.SEED)

    if config.SAVE_MODEL:
        if not os.path.exists(os.path.join(config.SAVE_DIR, config.SAVE_NAME)):
            os.mkdir(os.path.join(config.SAVE_DIR, config.SAVE_NAME))

    train_loader = torch.utils.data.DataLoader(
        datasets["train"], batch_size=config.BATCH_SIZE, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        datasets["validation"], batch_size=config.BATCH_SIZE, shuffle=True
    )

    # define loss function
    if config.MODEL_NAME == "TransUnet":
        loss = DiceBCELoss(config.NUMBER_OF_CLASSES)
    else:
        loss = DiceBCELoss(config.NUMBER_OF_CLASSES, normalizing=False)

    # define metrics
    metrics = [
        utils.metrics.IoU(threshold=0.5),
        utils.metrics.Recall(),
        utils.metrics.Fscore(),
    ]

    # define optimizer
    optimizer = torch.optim.Adam(
        [
            dict(params=model.parameters(), lr=config.BASE_LR),
        ]
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_epoch = utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=config.DEVICE,
        verbose=True,
    )

    valid_epoch = utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=config.DEVICE,
        verbose=True,
    )

    early_stop_counter = 0

    best_loss = 999999999.0
    loss_name = loss.__name__
    train_logs_list, valid_logs_list = [], []

    for i in range(0, config.MAX_EPOCHS):

        # Perform training & validation
        print("\nEpoch: {}".format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)

        lr_scheduler.step()

        # Save model if a better val IoU score is obtained
        if best_loss > valid_logs[loss_name]:
            best_loss = valid_logs[loss_name]

            if config.EARLY_STOPPING:
                print("trigger times: 0")
                early_stop_counter = 0

            if config.SAVE_MODEL:
                save_model_with_stats(
                    model, train_logs_list, valid_logs_list, MODEL_NAME, config.SAVE_DIR
                )
                # torch.save(model, os.path.join(save_dir,model_save_name))
                print("------------\nModel saved!\n------------")
            else:
                print("------------\nBest iou score!\n------------")

        elif config.EARLY_STOPPING:
            early_stop_counter += 1
            print(f"Early stop triggered {early_stop_counter} times:")

            if early_stop_counter > config.PATIENCE:
                print("Early stopping!\nStart to test process.")
                break

    return model, train_logs_list, valid_logs_list, best_loss


if __name__ == "__main__":

    # Getting model and config names from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Name of model to be tested")
    cli_args = parser.parse_args()

    MODEL_NAME = cli_args.model_name

    # Getting config info for pipeline
    config = read_config()
    # Mergin MAIN dict of config with model specific(model overwrites MAIN)
    config = config["MAIN"] | config["MODELS"][cli_args.model_name]
    # Making class-like interface for code readability
    config = SimpleNamespace(**config)

    # Adding model name to config for usability
    config.MODEL_NAME = cli_args.model_name

    # Replacing device name with torch.device
    if config.DEVICE == "cuda":
        config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        config.DEVICE = torch.device("cpu")

    print(f"Running on {config.DEVICE}")

    universal_transform = album.Compose(
        [
            album.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
            # album.RandomCrop(width=256, height=256),
            album.Rotate(p=0.6),
            album.HorizontalFlip(p=0.5),
            album.VerticalFlip(p=0.5),
            # album.RandomBrightnessContrast(p=0.35),
            album.RandomResizedCrop(config.IMAGE_SIZE, config.IMAGE_SIZE, p=0.3),
        ]
    )

    val_transform = album.Compose(
        [
            album.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        ]
    )

    image_transform = album.Compose(
        [
            album.RandomBrightnessContrast(p=0.35)
            # album.RandomResizedCrop(IMAGE_SIZE,IMAGE_SIZE,p=0.3)
        ]
    )

    # Importing model specific code
    model_file = import_module(f"models.{config.MODEL_NAME}")
    model_wrapper = model_file.create_wrapper()
    model = model_wrapper.model(config.NUMBER_OF_CLASSES)

    # Creating datasets
    dataset = SparseDatasetNpz(
        config.DATASET_DIR,
        augmentations=universal_transform,
        image_only_aug=image_transform,
        mask_only_aug=None,
        preprocessing=get_preprocessing(model_wrapper.preprocessing),
    )
    print(f"Training dataset size:{len(dataset)}")

    valid_dataset = SparseDatasetNpz(
        os.path.join(config.DATASET_DIR, "validation"),
        augmentations=val_transform,
        image_only_aug=None,
        mask_only_aug=None,
        preprocessing=get_preprocessing(model_wrapper.preprocessing),
    )
    print(f"Validation dataset size:{len(valid_dataset)}")

    datasets = {"train": dataset, "validation": valid_dataset}

    # Starting train-val loop
    model, train_logs, valid_log, best_loss = run_test(model, datasets)

    print(f"Best observed loss: {best_loss}")
