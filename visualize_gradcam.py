from __future__ import print_function
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torchvision import models, transforms
from torchvision.models import resnet
import argparse
import cv2,os,glob
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from pytorch_grad_cam import GradCAM, \
                             ScoreCAM, \
                             GradCAMPlusPlus, \
                             AblationCAM, \
                             XGradCAM, \
                             EigenCAM, \
                             EigenGradCAM, \
                             LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image,show_cam_on_image_nonmask
import torch.nn as nn
import warnings
warnings.filterwarnings(action='ignore')
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--imgpath', type=str, default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth', action='store_true',
                        help='Reduce noise by taking the first principle componenet'
                        'of cam_weights*activations')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args

def get_img_files(imgpath,position='front',ab_label='Tuberculosis'):
    imgs = glob.glob(imgpath + '/'+ab_label+'/*'+position+'.png')
    return imgs

def wide_resnet50(n_classes=2):
    net = resnet.wide_resnet50_2(pretrained=True)
    um_ftrs = net.fc.in_features
    net.fc = nn.Linear(um_ftrs, n_classes)
    return nn.DataParallel(net)
test_transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
def pre_processing(img):
    img = cv2.equalizeHist(img)
    img = Image.fromarray(img, 'L')
    raw_image = test_transform(img)
    raw_image = torch.cat((raw_image, raw_image, raw_image), axis=0)
    return raw_image

if __name__ == '__main__':

    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    imgpath='./Multiview_datasets/'
    label_targets= ['Tuberculosis', 'Pneumonia']
    positionsed = ['front','side60R','side60L','side30R','side30L']
    propose=True
    for label_target in label_targets:

        output_dir = './Result_visualize/' + label_target + '/'
        os.makedirs(output_dir, exist_ok=True)


        checkpointdir = './Classification/Multi_checkpoint/'
        for position in positionsed:
            model=wide_resnet50(n_classes=2)
            checkpoint = torch.load(
                checkpointdir + label_target+'_'+position + '.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()

            target_category = 1 #Abnormal #AI predict label = None

            image_paths = get_img_files(imgpath,position,label_target)

            target_layer = 'module.layer4'
            for name, module in model.named_modules():
                if name == target_layer:
                    target_layers=module
                    break

            cam_algorithm = GradCAMPlusPlus
            with cam_algorithm(model=model,
                               target_layers=target_layers,
                               use_cuda=args.use_cuda) as cam:
                for img_name in tqdm(image_paths):

                    img = cv2.imread(img_name, 0)
                    if propose ==True:
                        maskpath = img_name.replace('.png','mask.jpg')
                        mask = cv2.imread(maskpath, 0)
                        mask[mask > 0] = 1
                    else:
                        mask = np.ones((512, 512))

                    raw_image=pre_processing(img)
                    rgb_img = cv2.imread(img_name)
                    rgb_img=cv2.resize(rgb_img,(512,512))
                    mask = cv2.resize(mask, (512, 512))
                    input_tensor = torch.unsqueeze(raw_image,dim=0)

                    cam.batch_size = 1

                    grayscale_cam ,label= cam(input_tensor=input_tensor,
                                        target_category=target_category,
                                        aug_smooth=args.aug_smooth,
                                        eigen_smooth=args.eigen_smooth
                                        ,masks=mask)
                    grayscale_cam = grayscale_cam[0, :]
                    if propose == True:
                        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                    else:
                        cam_image = show_cam_on_image_nonmask(rgb_img, grayscale_cam, use_rgb=True)
                    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

                    cv2.imwrite(output_dir+img_name.split('/')[-1].replace('.png', '')+label+'_original.jpg', rgb_img )

                    cv2.imwrite(output_dir+img_name.split('/')[-1].replace('.png','')+label+'_cam.jpg', cam_image)
