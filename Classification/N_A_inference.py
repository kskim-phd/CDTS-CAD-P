import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.models import resnet
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import numpy as np
from sklearn.metrics import classification_report
import glob,cv2
from efficientnet_pytorch import EfficientNet
import warnings
from sklearn.metrics import confusion_matrix
from PIL import Image
from utils import plot_confusion_matrix
warnings.filterwarnings(action='ignore')
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


num_workers = 1
batch_size = 8
img_wsize = 512
img_hsize = 512
gpu = 'T'
n_classes = 2
# label_target = 'Tuberculosis'
label_target = 'Pneumonia'  # select
answer=2 #A

pretrained_modeldir= './Multi_checkpoint/'+label_target
def get_train_validation_files(ab_label='Tuberculosis'):

    validation_files = []
    print("Get train/validation files.")


    directory_0 = '../Multiview_datasets/Normal/'

    directory_1 = '../Multiview_datasets/' + str(ab_label) + '/'


    name_0 = glob.glob(directory_0 + '*front.png')
    name_0.sort()
    name_1 = glob.glob(directory_1 + '*front.png')
    name_1.sort()

    for file in name_0:
        validation_file = (file, [1, 0])
        validation_files.append(validation_file)
    for file in name_1:
        validation_file = (file, [0, 1])
        validation_files.append(validation_file)
    print('Testset : ', len(validation_files))

    return validation_files


class IntracranialDataset(Dataset):
    def __init__(self, df, transform=None):
        self.data = df
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data[idx][0]

        img = cv2.imread(img_name, 0)
        img1 = cv2.imread(img_name.replace('front', 'side30L'), 0)
        img2 = cv2.imread(img_name.replace('front', 'side60L'), 0)
        img3 = cv2.imread(img_name.replace('front', 'side30R'), 0)
        img4 = cv2.imread(img_name.replace('front', 'side60R'), 0)
        datas=[]
        for imgs,compose in zip([img,img1,img2,img3,img4],['front','side30L', 'side60L', 'side30R', 'side60R']):
            imgs=cv2.equalizeHist(imgs)
            datas.append(Image.fromarray(imgs, 'L'))
        img  = datas[0]
        img1 = datas[1]
        img2 = datas[2]
        img3 = datas[3]
        img4 = datas[4]

        if self.transform:
            img = self.transform(img)
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
            img4 = self.transform(img4)
        img = torch.cat((img, img, img), axis=0)
        img1 = torch.cat((img1, img1, img1), axis=0)
        img2 = torch.cat((img2, img2, img2), axis=0)
        img3 = torch.cat((img3, img3, img3), axis=0)
        img4 = torch.cat((img4, img4, img4), axis=0)
        labels = torch.tensor(self.data[idx][1])
        return {'image': [img,img1,img2,img3,img4], 'labels': labels}


def epochVal(modelfront,modelsideR30,modelsideL30,modelsideR60,modelsideL60, dataLoader):

    valprediction = []
    valy = []
    outputfnt_list=[]
    output60L_list=[]
    output60R_list=[]
    output30L_list=[]
    output30R_list=[]
    for i, batch in enumerate(dataLoader):
        front = batch["image"][0].float().cuda()
        side30L = batch["image"][1].float().cuda()
        side60L = batch["image"][2].float().cuda()
        side30R = batch["image"][3].float().cuda()
        side60R = batch["image"][4].float().cuda()

        target = batch["labels"].long().cuda()

        outputfnt = modelfront(front)
        output30L = modelsideL30(side30L)
        output60L = modelsideL60(side60L)
        output30R = modelsideR30(side30R)
        output60R = modelsideR60(side60R)

        outputfnt_list.append(torch.argmax(F.softmax(outputfnt, dim=1), axis=1).cpu().detach().numpy())
        output60L_list.append(torch.argmax(F.softmax(output60L, dim=1), axis=1).cpu().detach().numpy())
        output60R_list.append(torch.argmax(F.softmax(output60R, dim=1), axis=1).cpu().detach().numpy())
        output30L_list.append(torch.argmax(F.softmax(output30L, dim=1), axis=1).cpu().detach().numpy())
        output30R_list.append(torch.argmax(F.softmax(output30R, dim=1), axis=1).cpu().detach().numpy())

        valy += target.tolist()

        del front,side30L,side60L, side30R,side60R,target

    output=np.array([np.concatenate(outputfnt_list),
                     np.concatenate(output60L_list),
                     np.concatenate(output60R_list),
                     np.concatenate(output30L_list),
                     np.concatenate(output30R_list)])
    output=np.sum(output,axis=0)
    for idx in range(len(output)):
        output[idx]= 1 if output[idx]>=3 else 0

    print('='*20,'Baseline','='*20)
    print(classification_report( np.argmax(valy, axis=1),np.concatenate(outputfnt_list),digits=3,
                                target_names=['Normal', label_target]))
    cm = confusion_matrix(np.argmax(valy, axis=1), np.concatenate(outputfnt_list))
    plt.figure(figsize=(8, 8))
    plt.grid(b=False)
    plot_confusion_matrix(cm, classes=['Normal', label_target], normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues)
    plt.show()
    print('='*48)
    print('='*20,'Propose','='*20)
    print(classification_report( np.argmax(valy, axis=1),output,digits=3,
                                target_names=['Normal', label_target]))

    cm = confusion_matrix( np.argmax(valy, axis=1),output)
    plt.figure(figsize=(8, 8))
    plt.grid(b=False)
    plot_confusion_matrix(cm, classes=['Normal', label_target], normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues)
    plt.show()

    print('='*48)

transform_test = transforms.Compose([
    transforms.Resize((img_hsize, img_wsize)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])



def mymodel(n_classes):
    net = resnet.wide_resnet50_2(pretrained=True)
    um_ftrs = net.fc.in_features
    net.fc = nn.Linear(um_ftrs, n_classes)
    return nn.DataParallel(net)

def main():
    netfront = mymodel(n_classes).to(device)
    netside30R = mymodel(n_classes).to(device)
    netside30L = mymodel(n_classes).to(device)
    netside60R = mymodel(n_classes).to(device)
    netside60L = mymodel(n_classes).to(device)


    checkpoint = torch.load(pretrained_modeldir + '_front.pth')
    netfront.load_state_dict(checkpoint['model_state_dict'])
    checkpoint = torch.load(pretrained_modeldir + '_side30R.pth')
    netside30R.load_state_dict(checkpoint['model_state_dict'])
    checkpoint = torch.load(pretrained_modeldir + '_side30L.pth')
    netside30L.load_state_dict(checkpoint['model_state_dict'])
    checkpoint = torch.load(pretrained_modeldir + '_side60L.pth')
    netside60L.load_state_dict(checkpoint['model_state_dict'])
    checkpoint = torch.load(pretrained_modeldir + '_side60R.pth')
    netside60R.load_state_dict(checkpoint['model_state_dict'])

    testdata = get_train_validation_files(ab_label=label_target)
    valdataset = IntracranialDataset(testdata, transform=transform_test)
    valloader = DataLoader(valdataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    netfront.eval()
    netside30R.eval()
    netside30L.eval()
    netside60R.eval()
    netside60L.eval()

    epochVal(netfront ,netside30R ,netside30L ,netside60R ,netside60L, valloader)



if __name__=='__main__':

    main()
