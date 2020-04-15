import torch
from torch import nn
from torchvision import models

original_model = models.resnet34(pretrained=True)

class ImprovedModel(torch.nn.Module):
    
    def __init__(self, cfg):

        super(ImprovedModel, self).__init__()
        print(original_model)
        
        self.layer1 = nn.Sequential(
            *list(original_model.children())[0:6]
        )
        #print("Layer 1: ", self.layer1)

        self.layer2 = nn.Sequential(
            *list(original_model.children())[6:7]
        )
        #print("Layer 2: ", self.layer2)
        
        self.layer3 = nn.Sequential(
            *list(original_model.children())[7:8]
        )
        #print("Layer 3 ", self.layer3)
        
           
        self.layer4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3,3), stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()  
        )
        #print("Layer 4: ", self.layer4)
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=(1,2), stride=1, padding=1), #From [5,4] to [3,3] 
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()  
        )
        
        #print("Layer 5: ", self.layer5)
        
        
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3,3), stride=1, padding=1), #From [5,4] to [3,3] 
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3,3), stride=1, padding=1)
        )
        
        #self.layer6 = nn.Sequential(
        #    nn.AdaptiveAvgPool2d(output_size=(1, 1))
        #)
        
        
        #print("Layer 6: ", self.layer6)
        
       
        
        


    def forward(self, x):
        
        #print("Input: ", x.size())
        l1 = self.layer1(x)
        #print("L1: ", l1.size())
        l2 = self.layer2(l1)
        #print("L2: ", l2.size())
        l3 = self.layer3(l2)
        #print("L3: ", l3.size())
        l4 = self.layer4(l3)
        #print("L4: ", l4.size())
        l5 = self.layer5(l4)
        #print("L5: ", l5.size())
        l6 = self.layer6(l5)
        #print("L6: ", l6.size())

        out_features = [l1, l2, l3, l4, l5, l6]

        #for idx, feature in enumerate(out_features):
            #print(feature.shape[1:])

        return tuple(out_features)