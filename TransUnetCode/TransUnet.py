from TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import segmentation_models_pytorch as smp
import os

from torch.utils import data
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import sparse
import pickle

import albumentations as A
import cv2
import numpy as np
import matplotlib.pyplot as plt

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
        # remap pixels to [0,1]
        image = image.astype('float') / 255

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



def to_tensor(x, **kwargs):
    return TF.to_tensor(x)

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

from torch.nn.modules.loss import CrossEntropyLoss
import torch.nn.functional as F
class DiceBCELoss(torch.nn.Module):
    def __init__(self, n_classes, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.__name__ = 'DiceBCELoss'
        self.dl = DiceLoss(n_classes)
        self.ce = CrossEntropyLoss()

    def forward(self, inputs, targets, smooth=1e-5):
        
        #CrossEntropyLoss
        CE = self.ce(inputs, targets)

        dice_loss = self.dl(inputs,targets,softmax=True)

        Dice_BCE = 0.5*CE + 0.5*dice_loss
        
        return Dice_BCE
# ## Model
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
            #inputs = torch.softmax(inputs, dim=1)
            inputs = torch.sigmoid(inputs) # my stuff
        #target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


# Arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, help='Dataset directory')
parser.add_argument('--pretrained_path', type=str, help='Path to pretrained vit model file')
parser.add_argument('--num_classes', type=int,
                    default=3, help='output channel of network')
parser.add_argument('--max_epochs', type=int,
                    default=30, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=8, help='batch_size')
parser.add_argument('--find_lr', type=bool,  default=False,
                    help='Perform lr search')                   
parser.add_argument('--base_lr', type=float,  default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=42, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--save_name', type=str,
                    default=None, help='select model save name')
parser.add_argument('--save_path', type=str,
                    default=None, help='select model save path')
args = parser.parse_args()








if __name__ == '__main__':

    
    seed_everything(args.seed)
    
    IMAGE_SIZE = args.img_size

    universal_transform = A.Compose([
        A.Resize(IMAGE_SIZE,IMAGE_SIZE),
        #A.RandomCrop(width=256, height=256),
        A.Rotate(p=0.6),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        #A.RandomBrightnessContrast(p=0.35),
        #A.RandomResizedCrop(IMAGE_SIZE,IMAGE_SIZE,p=0.3)
    ])



    image_transform = A.Compose([
        A.RandomBrightnessContrast(p=0.35)
        #A.RandomResizedCrop(IMAGE_SIZE,IMAGE_SIZE,p=0.3)
    ])
    
    val_transform = A.Compose([
        A.Resize(IMAGE_SIZE,IMAGE_SIZE),
    ])
    

    vit_name = args.vit_name
    vit_patches_size = args.vit_patches_size
    pretrained_path = args.pretrained_path

    config_vit = CONFIGS_ViT_seg[vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(IMAGE_SIZE / vit_patches_size), int(IMAGE_SIZE / vit_patches_size))
        
    model = ViT_seg(config_vit, img_size=IMAGE_SIZE, num_classes=config_vit.n_classes)
    model.load_from(weights=np.load(pretrained_path))

    
    
    dataset_dir = args.dataset_dir
    dataset = sparseDatasetNpz(dataset_dir,
                               augmentations=universal_transform,
                               image_only_aug=image_transform,
                               mask_only_aug=None,
                               preprocessing=get_preprocessing())
    print(len(dataset))


    val_dataset = sparseDatasetNpz(os.path.join(dataset_dir,'validation'), 
                                   augmentations=val_transform,
                                   image_only_aug=None,
                                   mask_only_aug=None,
                                   preprocessing=get_preprocessing())
    print(len(val_dataset))

    

    model_name = args.save_name

    SAVE_MODEL = False if model_name is None else True
    save_dir = args.save_path

    if SAVE_MODEL:
      if not os.path.exists(os.path.join(save_dir, model_name)):
        os.mkdir(os.path.join(save_dir, model_name))
    

    BATCH_SIZE = args.batch_size

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,shuffle=True)
    valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE,shuffle=True)


    # Set num of epochs
    EPOCHS = args.max_epochs

    # Set device: `cuda` or `cpu`
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define loss function config_vit.n_classes
    #loss = smp.utils.losses.DiceLoss(activation=torch.sigmoid)
    loss = DiceBCELoss(config_vit.n_classes)

    # define metrics
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Recall(),
        smp.utils.metrics.Fscore()    
    ]

    # define optimizer
    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=args.base_lr),
    ])

    if args.find_lr:
        from torch_lr_finder import LRFinder

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-2)
        lr_finder = LRFinder(model, optimizer, loss, device="cuda")
        lr_finder.range_test(train_loader, end_lr=0.1, num_iter=300)
        lr_finder.plot() # to inspect the loss-learning rate graph
        lr_finder.reset() # to reset the model and optimizer to their initial state

        exit()

    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor = 1,end_factor = 0.01,total_iters=EPOCHS)
    

    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )


    best_score = 999999999.0
    loss_name = loss.__name__
    train_logs_list, valid_logs_list = [], []

    for i in range(0, EPOCHS):

        # Perform training & validation
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)

        lr_scheduler.step()

        # Save model if a better val IoU score is obtained
        if best_score > valid_logs[loss_name]:
            best_score = valid_logs[loss_name]
            if SAVE_MODEL:
                save_model_with_stats(model,train_logs_list,valid_logs_list,model_name,save_dir)
                #torch.save(model, os.path.join(save_dir,model_save_name))
                print('------------\nModel saved!\n------------')
            else:
                print('------------\nBest iou score!\n------------')

    print(best_score)
