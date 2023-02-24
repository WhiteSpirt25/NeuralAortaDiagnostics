import segmentation_models_pytorch as smp

from .model_wrapper import ModelWrapper


def create_model(number_of_classes):
    """Creates DeepLabV3Plus + xception71 pretrained model."""
    ENCODER = "tu-xception71"
    ENCODER_WEIGHTS = "imagenet"
    ACTIVATION = (
        "sigmoid"
    )

    # create segmentation model with pretrained encoder
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=number_of_classes,
        activation=ACTIVATION,
    )

    return model

def create_wrapper():
    """Function to be called in pipeline. Must return instance of ModelWrapper"""
    return ModelWrapper(model=create_model)
