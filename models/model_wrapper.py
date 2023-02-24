from dataclasses import dataclass
from typing import Callable

@dataclass
class ModelWrapper:
    """Simple dataclass adapter between ML models and learning pipeline.
    
    model -- pytorch's model suitable to used in 
        segmentations models pytorch(smp) train and valid epoch.
    
    preprocessing -- function to be applied after augmenations to the image.
    """
    model: Callable
    preprocessing: Callable = None
