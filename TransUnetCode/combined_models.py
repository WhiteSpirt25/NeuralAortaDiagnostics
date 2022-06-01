import segmentation_models_pytorch as smp
import pickle
import os
from torch.utils import data
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import sparse
import pickle
import albumentations as A
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import torch.nn.functional as nnf
from TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg

def save_model_with_stats(model,train_logs,validation_logs,model_name,save_path):
    model_dir = os.path.join(save_dir,model_name)
    torch.save(model, os.path.join(model_dir,f'{model_name}.pth'))

    with open(os.path.join(model_dir,'train_logs.pickle'),'wb') as file:
        pickle.dump(train_logs, file)

    with open(os.path.join(model_dir,'validation_logs.pickle'),'wb') as file:
        pickle.dump(validation_logs, file)

def load_sparce_npz(path:str):
    '''
    Loads npy array as sparce pickled scipy matrix.
    '''
    with open(path,'rb') as file:
        s = pickle.load(file)
    
    # convert to numpy array
    s = s.todense()
    if len(s.shape) == 2:
      return s
    return np.transpose(s,[1,2,0])
    


class sparseDatasetNpz(data.Dataset):
    
    # initialise function of class
    def __init__(self, root, augmentations = None,image_only_aug = None,mask_only_aug = None, preprocessing = None):
        # the data directory 
        self.root = root
        # the list of filename
        self.filenames = os.listdir(os.path.join(root,'images'))
        #self.target_transform = target_transform
        self.augmentation = augmentations
        self.image_only_aug = image_only_aug
        self.mask_only_aug = mask_only_aug
        self.preprocessing = preprocessing

    # obtain the sample with the given index
    def __getitem__(self, index):
        # obtain filenames from list
        image_filename = self.filenames[index]
        # Load data and label
        image = cv2.imread(os.path.join(self.root,'images', image_filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        pre, ext = os.path.splitext(image_filename)
        mask_filename =  pre + '.pickle'

        #print(os.path.join(self.root,'masks', mask_filename))
        mask = load_sparce_npz(os.path.join(self.root,'masks', mask_filename))
                   
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        
        if self.image_only_aug:
            sample = self.image_only_aug(image=image)
            image = sample['image']
            
        if self.mask_only_aug:
            sample = self.mask_only_aug(mask=mask)
            mask = sample['mask']
            
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        image=image.float()
        # output of Dataset must be tensor so tensor in transforms
        return image, mask
    
    # the total number of samples (optional)
    def __len__(self):
        return len(self.filenames)


class sparseDatasetNpzInMem(data.Dataset):
    
    # initialise function of class
    def __init__(self, root, augmentations = None,image_only_aug = None, preprocessing = None):
        # the data directory 
        self.root = root
        # the list of filename
        self.filenames = os.listdir(os.path.join(root,'images'))
        #self.target_transform = target_transform
        self.augmentation = augmentations
        self.image_only_aug = image_only_aug
        self.preprocessing = preprocessing
        
        self.images = []
        self.masks = []
        for image_filename in self.filenames:
            img = cv2.imread(os.path.join(self.root,'images', image_filename))
            self.images.append(img)
            
            pre, ext = os.path.splitext(image_filename)
            mask_filename =  pre + '.pickle'
            
            mask = load_sparce_npz(os.path.join(self.root,'masks', mask_filename))
            self.masks.append(mask)
        
        self.images = np.array(self.images,dtype=object)
        self.masks = np.array(self.masks,dtype=object)

    # obtain the sample with the given index
    def __getitem__(self, index):
        # obtain filenames from list
        image = self.images[index]
        mask = self.masks[index]
                   
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        
        if self.image_only_aug:
            sample = self.image_only_aug(image=image)
            image = sample['image']
            
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        image=image.float()
        '''
        # universal preprocessing
        image = TF.to_tensor(tr['image'])
        image = TF.normalize(image,(0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))
        mask = tr['mask']
        '''
        # output of Dataset must be tensor so tensor in transforms
        return image, mask
    
    # the total number of samples (optional)
    def __len__(self):
        return len(self.filenames)



def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def to_tensor(x, **kwargs):
    return TF.to_tensor(x)
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn = None):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    if preprocessing_fn:
      _transform = [
          A.Lambda(image=preprocessing_fn),
          A.Lambda(image=to_tensor, mask=to_tensor),
      ]
    else:
      _transform = [
          A.Lambda(image=to_tensor, mask=to_tensor),
      ]
    return A.Compose(_transform)

import torch.nn.functional as F
class DiceBCELoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.__name__ = 'DiceBCELoss'
    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets.float(), reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

class MyDoubleNet(torch.nn.Module): 
    def __init__(self, first_model_path,second_model_path,device,preprocessing = None):
        super(MyDoubleNet, self).__init__()
        if os.path.exists(first_model_path):
            self.first_model = torch.load(first_model_path, map_location=device)
            self.first_model.eval()
        else:
            print(f'Error finding model [{first_model_path}]')
            raise 
            
        if os.path.exists(second_model_path):
            self.second_model = torch.load(second_model_path, map_location=device)
            self.second_model.eval()
        else:
            print(f'Error finding model [{second_model_path}]')
            raise 
        
        self.preprocessing = preprocessing
            

    def forward(self, x):
        first_model_output = self.first_model.forward(x)
        
        if self.preprocessing:
            sample = self.preprocessing(image=first_model_output)
            first_model_output = sample['image']
        
        
        newx = torch.clone(x)
        newx[:,2,:,:] = torch.sigmoid(first_model_output[:,0,:,:])
        
        output = self.second_model(newx)
        
        return output


#from self_attention_cv.transunet import TransUnet
if __name__ == '__main__':
    
    seed_everything(42)
    
    IMAGE_SIZE = 224

    val_transform = A.Compose([
        A.Resize(IMAGE_SIZE,IMAGE_SIZE),
    ])

    dataset_dir = r'C:\diploma\datasets\imageTBAD_Full'

    val_dataset = sparseDatasetNpz(os.path.join(dataset_dir,'validation'), 
                                   augmentations=val_transform,
                                   image_only_aug=None,
                                   mask_only_aug=None,
                                   preprocessing=get_preprocessing(None))


    BATCH_SIZE = 12

    valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE,shuffle=True)

    # Set device: `cuda` or `cpu`
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = MyDoubleNet(first_model_path=r"E:\University\ScienceWork\Medicine\AortaStuff\ML\models\aorta\TransUnet_norrc\TransUnet_norrc.pth",
                        second_model_path=r"E:\University\ScienceWork\Medicine\AortaStuff\ML\models\flows\Xception_DLV3P_dbce_30e_224\Xception_DLV3P_dbce_30e_224.pth",device=DEVICE)
    
    model.eval()
    # define loss function
    
    loss = smp.utils.losses.DiceLoss()
    
    # define metrics
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Recall(),
        smp.utils.metrics.Fscore()    
    ]

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )
    valid_logs = valid_epoch.run(valid_loader)
    print(valid_logs)
