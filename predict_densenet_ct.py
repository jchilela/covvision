"""
IRB - Instituto Respira Brasil

@script: predict_densenet.py

Objetivos:
    Predizer (APENAS PREDICT) via Rede Neural de Imagens de Tomografia 
    Computadorizada para teste de Covid-19

Script oriiginal de github.com/iliasprc/COVIDNET

Essa vesão foi adaptado de 
   https://drive.google.com/drive/folders/11YmPgWuUjLAFPlpJpMIwkdgAfSOQYRiP

Versão 1 Adaptada por Julio Chilela e Lirio Ramalheira em 09 de junho 2020
"""
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
"""

Versão 2 Adaptda por Sam Faraday em 1 de julho de 2020
       Ajustados os paths para Linux
       Criadas pasta Iimagens_processadas e as subpastas CT_COVID e CT_NonCOVID
       Removidos 
                Imports and Variables not used   
                Old Comments 
Último Backup do modeloe treinado em
 em https://drive.google.com/drive/folders/1ZvjxXx8-b7pTgRDwG8r6oDyJUwdfxegQ               
"""
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import os
from PIL import Image
from torch.optim.lr_scheduler import StepLR
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
# In[1]:

__name__ = '__main__'

torch.cuda.empty_cache()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop((224),scale=(0.5,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

val_transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])


# In[2]:

batchsize=10
def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data

class CovidCTDataset(Dataset):
    def __init__(self, root_dir, txt_COVID, txt_NonCOVID, transform=None):
        """
        Args:
            txt_path (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        File structure:
        - root_dir
            - CT_COVID
                - img1.png
                - img2.png
                - ......
            - CT_NonCOVID
                - img1.png
                - img2.png
                - ......
        """
        self.root_dir = root_dir
        self.txt_path = [txt_COVID,txt_NonCOVID]
        self.classes = ['CT_COVID', 'CT_NonCOVID']
        self.num_cls = len(self.classes)
        self.img_list = []
        for c in range(self.num_cls):
            cls_list = [[os.path.join(self.root_dir,self.classes[c],item), c] for item in read_txt(self.txt_path[c])]
            self.img_list += cls_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_list[idx][0]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        sample = {'img': image,
                  'label': int(self.img_list[idx][1])}
        return sample
    
if __name__ == '__main__':
    testset = CovidCTDataset(root_dir      ='/home/charles/workspace/CovidCTNew/baseline methods/DenseNet169/imagens_processadas',#pastas da imagem a ser processada
                              txt_COVID    ='/home/charles/workspace/CovidCTNew/baseline methods/DenseNet169/imagens_processadas/CT_COVID/testCT_COVID.txt',# limpar e  acrescentar o nome da imagem no arquivo txt
                              txt_NonCOVID ='/home/charles/workspace/CovidCTNew/baseline methods/DenseNet169/imagens_processadas/CT_NonCOVID/testCT_NonCOVID.txt', #limpar
                              transform= val_transformer)
    
    print(testset.__len__())

    test_loader = DataLoader(testset, batch_size=batchsize, drop_last=False, shuffle=False)
    from efficientnet_pytorch import EfficientNet
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
    model = model.cuda()
    checkpoint = torch.load('/home/charles/workspace/CovidCTNew/baseline methods/DenseNet169/modelo_treinado/efficientNet-b0.pt') #modelo treinamento
    model.load_state_dict(checkpoint)
    modelname = 'efficientNet-b0' #modelo treinado
    alpha = None
    device = 'cuda'
    votenum = 10
    #test(1)


def test(epoch):
    
    model.eval()
    test_loss = 0
    correct = 0
  
    criteria = nn.CrossEntropyLoss()
    # Don't update model
    with torch.no_grad():
        predlist=[]
        scorelist=[]
        targetlist=[]
        # Predict
        for batch_index, batch_samples in enumerate(test_loader):
            data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)

            output = model(data)
            
            test_loss += criteria(output, target.long())
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.long().view_as(pred)).sum().item()
            targetcpu=target.long().cpu().numpy()
            predlist=np.append(predlist, pred.cpu().numpy())
            scorelist=np.append(scorelist, score.cpu().numpy()[:,1])
            targetlist=np.append(targetlist,targetcpu)
           
    return targetlist, scorelist, predlist    

# In[3]:


    
# test
#bs = 10
import warnings
warnings.filterwarnings('ignore')

#r_list = []
#p_list = []
#acc_list = []
#AUC_list = []

vote_pred = np.zeros(testset.__len__())
vote_score = np.zeros(testset.__len__())

#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum = 0.9)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
scheduler = StepLR(optimizer, step_size=1)

total_epoch = 1
for epoch in range(1, total_epoch+1):
    
    targetlist, scorelist, predlist = test(epoch)
    vote_pred = vote_pred + predlist 
    vote_score = vote_score + scorelist 
    
    TP = ((predlist == 1) & (targetlist == 1)).sum()
    TN = ((predlist == 0) & (targetlist == 0)).sum()
    FN = ((predlist == 0) & (targetlist == 1)).sum()
    FP = ((predlist == 1) & (targetlist == 0)).sum()
    """
    print('TP=',TP,'TN=',TN,'FN=',FN,'FP=',FP)
    print('TP+FP',TP+FP)
    p = TP / (TP + FP)
    print('precision',p)
    p = TP / (TP + FP)
    r = TP / (TP + FN)
    print('recall',r)
    F1 = 2 * r * p / (r + p)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('F1',F1)
    print('acc',acc)
    #AUC = roc_auc_score(targetlist, vote_score)
    #print('AUC', AUC)
    """
    if epoch == 1:
        
        # major vote
        vote_pred[vote_pred <= (votenum/2)] = 0
        vote_pred[vote_pred > (votenum/2)] = 1
        
        TP = ((vote_pred == 1) & (targetlist == 1)).sum()
        TN = ((vote_pred == 0) & (targetlist == 0)).sum()
        FN = ((vote_pred == 0) & (targetlist == 1)).sum()
        FP = ((vote_pred == 1) & (targetlist == 0)).sum()
        """ 
        print('TP=',TP,'TN=',TN,'FN=',FN,'FP=',FP)
        print('TP+FP',TP+FP)
        p = TP / (TP + FP)
        print('precision',p)
        p = TP / (TP + FP)
        r = TP / (TP + FN)
        print('recall',r)
        F1 = 2 * r * p / (r + p)
        acc = (TP + TN) / (TP + TN + FP + FN)
        print('F1',F1)
        print('acc',acc)
        #AUC = roc_auc_score(targetlist, vote_score)
        #print('AUC', AUC)
        """
        vote_pred = np.zeros((1,testset.__len__()))
        vote_score = np.zeros(testset.__len__())
        print('vote_pred',vote_pred) #aarray com 1 ou zero
        #print('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}'.format(epoch, r, p, F1, acc, AUC))

       # f = open(f'model_result/test_{modelname}.txt', 'a+')
        #f.write('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}'.format(epoch, r, p, F1, acc, AUC))
        #f.close()