# -*- coding: utf-8 -*-


import torch
import numpy as np

# dataset
import mydataset_v1 as mydataset
from torch.utils.data import DataLoader
import os
import cv2
from network import AttU_Net


num_batch = 8
num_mask = 2

def main():
    # Semantic segmentation (inference)

    if torch.cuda.is_available():
        device = torch.device("cuda") 
        num_worker = 0
    else:
        device = torch.device("cpu") 
        num_worker = 0
    save_dir = './savemask/'
    image_paths = '../Multiview_datasets/'
    # Model initialization
    net=AttU_Net(img_ch=1,output_ch=2)

    # Load model
    views = ['front','side30L','side60L','side30R','side60R']
    for position in views:
        model_dir='./checkpoint/model_'+position+'.pth'
        if os.path.isfile(model_dir):
            print('\n>> Load model - %s' % (model_dir))
            checkpoint = torch.load(model_dir)
            net.load_state_dict(checkpoint['model_state_dict'])
        else:
            print('[Err] Model does not exist in %s' % (model_dir))
            exit()

        net.to(device)
        testset = mydataset.MyInferenceClass(image_path=image_paths,position=position,save_dir=save_dir)
        testloader = DataLoader(testset, batch_size=num_batch, shuffle=False, num_workers=num_worker, pin_memory=True)
        print("  >>> Total # of test sampler : %d" % (len(testset)))

        # inference
        print('\n\n>> Evaluate Network')

        with torch.no_grad():

            # initialize
            net.eval()
            for i, data in enumerate(testloader, 0):
                outputs = net(data['input'].to(device))
                outputs = torch.argmax(outputs.detach(), dim=1)

                # each case
                for k in range(len(data['input'])):
                    # get size and case id
                    # post processing
                    outputs_max = [mydataset.one_hot(outputs[k], num_mask) for k in range(len(data['input']))]
                    cv2.imwrite(data['name'][k].replace('.png','') + 'mask.jpg', torch.argmax(outputs_max[k], dim=0).numpy() * 255)


        print(position , ' segmentation done! ' )


if __name__=='__main__':

    main()

