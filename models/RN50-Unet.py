import segmentation_models_pytorch as smp

from .model_wrapper import ModelWrapper

ENCODER = "resnet50"
ENCODER_WEIGHTS = "imagenet"
ACTIVATION = "sigmoid"


def create_model(number_of_classes):
    """Creates ResNet50 + Unet pretrained model."""

    # create segmentation model with pretrained encoder
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=number_of_classes,
        activation=ACTIVATION,
    )

    return model


def create_wrapper():
    return ModelWrapper(
        model=create_model,
        preprocessing=smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS),
    )
