import numpy as np

from .TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from .TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

from .model_wrapper import ModelWrapper


def create_model(number_of_classes):
    """Creates TransUnet model using ViT_seg config same as in original repo. """
    IMAGE_SIZE = 224
    vit_name = "R50-ViT-B_16"
    vit_patches_size = 16
    pretrained_path = r"C:\Users\alexa\Downloads\imagenet21k_R50+ViT-B_16.npz"

    config_vit = CONFIGS_ViT_seg[vit_name]
    config_vit.n_classes = number_of_classes
    config_vit.n_skip = 3
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (
            int(IMAGE_SIZE / vit_patches_size),\
            int(IMAGE_SIZE / vit_patches_size)
            )
        
    model = ViT_seg(
        config_vit,\
        img_size=IMAGE_SIZE,\
        num_classes=config_vit.n_classes\
                )
    model.load_from(weights=np.load(pretrained_path))
    return model


def image_range_preprocessing(image, **kwargs):
    """Normalizes images(not masks) to use in TransUnet."""
    image = image.astype("float") / 255
    return image


def create_wrapper():
    return ModelWrapper(model=create_model, preprocessing=image_range_preprocessing)
