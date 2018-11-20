import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pretrainedmodels

nclasses = 20 


# class Net_vgg(nn.Module):
#    def __init__(self,num_classes=20):
#        super(Net_vgg,self).__init__()
#        self.vgg16 = models.vgg16(pretrained=True)
#        for param in self.vgg16.features.parameters():
#            param.requires_grad = False
#        for param in list(self.vgg16.classifier.parameters())[:-1] : 
#            param.requires_grad = False
#        num_features = self.vgg16.classifier[-1].in_features
#        self.vgg16.classifier = self.vgg16.classifier[:-1]
     
#        lin1 = nn.Linear(num_features,512)
#        relu = nn.ReLU()
#        dp = nn.Dropout(0.3)
#        lin2 = nn.Linear(512,20)
     
#        self.fc = nn.Sequential(lin1,relu,dp,lin2)
      
     
#    def forward(self, input):
#        output = self.vgg16(input)
#        return self.fc(output)

    
# class Net(nn.Module):
#     def __init__(self,num_classes=20):
#         super(Net,self).__init__()
#         self.inc = models.inception_v3(pretrained=True)
       
#         for param in self.inc.parameters():
#             param.requires_grad = False
       
#          num_fea = self.inc.AuxLogits.fc.in_features
#          self.inc.AuxLogits.fc = nn.Linear(num_fea,20)
       
#          num_features = self.inc.fc.in_features
#          self.inc.fc = nn.Linear(num_features,20)
            
#     def forward(self, input):
#         x = self.inc(input)
#         return x


# class Net(nn.Module):
#     def __init__(self,num_classes=20):
#         super(Net,self).__init__()
#         self.res = models.resnet152(pretrained=True)

#         for param in self.res.conv1.parameters():
#             param.requires_grad = False
#         for param in self.res.bn1.parameters():
#             param.requires_grad = False
#         for param in self.res.layer1.parameters():
#             param.requires_grad = False
#         for param in self.res.layer2.parameters():
#             param.requires_grad = False
#         for param in self.res.layer3.parameters():
#             param.requires_grad = False    
        
        
        
#         self.res.avgpool = nn.Sequential()
#         num_features = self.res.fc.in_features
        
        
        
        
#         lin1 = nn.Linear(num_features,num_classes)
#         #relu = nn.ReLU()
#         #lin2 = nn.Linear(1024,256)
#         #lin3 = nn.Linear(512,20)
      
#         self.res.fc = lin1 #nn.Sequential(lin1,relu,lin3)
#         self.res.fc = nn.Sequential()
      
#     def forward(self, input):
#         x = self.res(input)
#         print(x.size())
#         return x
    
    
    
    
# class Net(nn.Module):
#     def __init__(self,num_classes=20):
#         super(Net,self).__init__()
        
#         self.res2 = models.resnet101(pretrained=True)
#         self.res = models.resnet152(pretrained=True)
        
#         for param in self.res2.conv1.parameters():
#             param.requires_grad = False
#         for param in self.res2.bn1.parameters():
#             param.requires_grad = False
#         for param in self.res2.layer1.parameters():
#             param.requires_grad = False
#         for param in self.res2.layer2.parameters():
#             param.requires_grad = False
#         for param in self.res2.layer3.parameters():
#             param.requires_grad = False 
            
#         for param in self.res.conv1.parameters():
#             param.requires_grad = False
#         for param in self.res.bn1.parameters():
#             param.requires_grad = False
#         for param in self.res.layer1.parameters():
#             param.requires_grad = False
#         for param in self.res.layer2.parameters():
#             param.requires_grad = False
#         for param in self.res.layer3.parameters():
#             param.requires_grad = False    
        
#         num_features = self.res.fc.in_features
#         num_features2 = self.res2.fc.in_features
        
#         self.res2.fc = nn.Sequential()
#         self.res.fc = nn.Sequential()
        
# #         lin1 = nn.Linear(1024,256)
# #         relu = nn.ReLU()
# #         lin2 = nn.Linear(256,20)
      
#         self.fc = nn.Linear(num_features+num_features2,20)
        
       
      
#     def forward(self, input):
#         x1 = self.res(input)
#         x2 = self.res2(input)
#         x = torch.cat((x1,x2),1)
#         return self.fc(x)
    
    
class Net(nn.Module):
    def __init__(self,num_classes=20):
        super(Net,self).__init__()
        
        self.res = models.resnet152(pretrained=True)
        self.inc = models.inception_v3(pretrained=True)
       
        for param in self.inc.parameters():
            param.requires_grad = False
            
        self.inc.aux_logits = False
        
        for param in self.inc.Mixed_7b.parameters() : 
            param.requires_grad = True 
        for param in self.inc.Mixed_7c.parameters() : 
            param.requires_grad = True
       
        num_features = self.inc.fc.in_features
        
        self.inc.fc = nn.Linear(num_features,512)
            
        

            
        for param in self.res.conv1.parameters():
            param.requires_grad = False
        for param in self.res.bn1.parameters():
            param.requires_grad = False
        for param in self.res.layer1.parameters():
            param.requires_grad = False
        for param in self.res.layer2.parameters():
            param.requires_grad = False
        for param in self.res.layer3.parameters():
            param.requires_grad = False     
        
        self.res.avgpool = nn.AvgPool2d(10)
        
        num_features2 = self.res.fc.in_features
        
        self.res.fc = nn.Linear(num_features2, 512)

        lin3 = nn.Linear(1024,20)
      
        self.fc = lin3
      
    def forward(self, input):
        x1 = self.res(input)
        x2 = self.inc(input)
        x = torch.cat((x1,x2),1)
        return self.fc(x)

      
    
    
# class Net(nn.Module):
#     def __init__(self,num_classes=20):
#         super(Net,self).__init__()
        
#         self.res = models.resnet152(pretrained=True)
#         self.inc = models.inception_v3(pretrained=True)
#         self.vgg = models.vgg16_bn(pretrained=True)
#         for param in self.vgg.features.parameters():
#             param.requires_grad = False
# #         for param in list(self.vgg16.classifier.parameters())[:-1] : 
# #             param.requires_grad = False
# #         num_features = self.vgg16.classifier[-1].in_features
#         lin1_vgg = nn.Linear(41472,4096)
#         lin2_vgg = nn.Linear(4096,1024)
#         lin3_vgg = nn.Linear(1024,20)
#         relu = nn.ReLU()
#         self.vgg.classifier = nn.Sequential(lin1_vgg,relu,lin2_vgg,relu,lin3_vgg)
       
#         for param in self.inc.parameters():
#             param.requires_grad = False
       
#         num_features = self.inc.fc.in_features
        
#         self.inc.fc = nn.Linear(num_features,20)
            
#         self.inc.aux_logits = False
# #         num_fea = self.inc.AuxLogits.fc.in_features
        
       
# #         self.inc.AuxLogits.fc = nn.Linear(num_fea,512)
            
#         for param in self.res.conv1.parameters():
#             param.requires_grad = False
#         for param in self.res.bn1.parameters():
#             param.requires_grad = False
#         for param in self.res.layer1.parameters():
#             param.requires_grad = False
#         for param in self.res.layer2.parameters():
#             param.requires_grad = False
#         for param in self.res.layer3.parameters():
#             param.requires_grad = False    
        
        
        
        
#         self.res.avgpool = nn.AvgPool2d(10)
        
#         num_features2 = self.res.fc.in_features
        
#         self.res.fc = nn.Linear(num_features2,20)
        
                

# #         lin3 = nn.Linear(num_features+num_features2+2048, 20)
# # #         lin4 = nn.Linear(1024,20)
      
# #         self.fc = lin3     #nn.Sequential(lin3,lin4)
        
       
      
#     def forward(self, input):
#         x1 = self.res(input)
#         x2 = self.inc(input)
#         x3 = self.vgg(input)
#         x = x1+x2+x3
#         return x
    
    
# class Net(nn.Module):
#     def __init__(self,num_classes=20):
#         super(Net,self).__init__()
        
#         self.res = models.resnet152(pretrained=True)
#         self.inc = models.inception_v3(pretrained=True)
#         self.vgg = models.vgg16_bn(pretrained=True)
#         for param in self.vgg.features.parameters():
#             param.requires_grad = False
            
#         for param in list(self.vgg.features.parameters())[-12:] : 
#             param.requires_grad = True
# #         for param in list(self.vgg16.classifier.parameters())[:-1] : 
# #             param.requires_grad = False
# #         num_features = self.vgg16.classifier[-1].in_features
# #         lin1_vgg = nn.Linear(41472,4096)
# #         lin2_vgg = nn.Linear(4096,1024)
# #         lin3_vgg = nn.Linear(1024,20)
# #         relu = nn.ReLU()
#         self.vgg.classifier = nn.Linear(41472,1024)#nn.Sequential(lin1_vgg,relu,lin2_vgg,relu,lin3_vgg)
       
#         for param in self.inc.parameters():
#             param.requires_grad = False
       
#         for param in self.inc.Mixed_7b.parameters() : 
#             param.requires_grad = True 
#         for param in self.inc.Mixed_7c.parameters() : 
#             param.requires_grad = True
#         num_features = self.inc.fc.in_features
        
        
#         self.inc.fc = nn.Linear(num_features,512)
            
#         self.inc.aux_logits = False
# #         num_fea = self.inc.AuxLogits.fc.in_features
        
       
# #         self.inc.AuxLogits.fc = nn.Linear(num_fea,512)
            
#         for param in self.res.conv1.parameters():
#             param.requires_grad = False
#         for param in self.res.bn1.parameters():
#             param.requires_grad = False
#         for param in self.res.layer1.parameters():
#             param.requires_grad = False
#         for param in self.res.layer2.parameters():
#             param.requires_grad = False
#         for param in self.res.layer3.parameters():
#             param.requires_grad = False    
        
        
        
        
#         self.res.avgpool = nn.AvgPool2d(10)
        
#         num_features2 = self.res.fc.in_features
        
#         self.res.fc = nn.Linear(num_features2,512)
        
                

#         lin3 = nn.Linear(2048, 20)
# # #         lin4 = nn.Linear(1024,20)
      
#         self.fc = lin3     #nn.Sequential(lin3,lin4)
        
       
      
#     def forward(self, input):
#         x1 = self.res(input)
#         x2 = self.inc(input)
#         x3 = self.vgg(input)
#         x = torch.cat((x1,x2,x3),1)
#         return self.fc(x)
    
    
# class Net(nn.Module):
#     def __init__(self,num_classes=20):
#         super(Net,self).__init__()
        
# #         self.nas = pretrainedmodels.pnasnet5large(1000,'imagenet')
#         self.nas = pretrainedmodels.xception(num_classes=1000, pretrained='imagenet')
#         for param in self.nas.parameters() : 
#             param.requires_grad = False
#         for param in self.nas.block12.parameters() : 
#             param.requires_grad = True 
#         for param in self.nas.conv3.parameters() :
#             param.requires_grad = True
#         for param in self.nas.conv4.parameters() :
#             param.requires_grad = True
            
#         num_features = self.nas.last_linear.in_features
        
#         self.nas.last_linear = nn.Linear(num_features, 20)

#     def forward(self, input):
#         return self.nas(input)
    
        
    
# class Net(nn.Module):
#     def __init__(self,num_classes=20):
#         super(Net,self).__init__()
        
#         self.res = models.resnet152(pretrained=True)
#         self.inc = pretrainedmodels.xception(num_classes=1000, pretrained='imagenet')
       
#         for param in self.inc.parameters():
#             param.requires_grad = False
#         for param in self.inc.block12.parameters() : 
#             param.requires_grad = True 
#         for param in self.inc.conv3.parameters() :
#             param.requires_grad = True
#         for param in self.inc.conv4.parameters() :
#             param.requires_grad = True            
            
#         num_features = self.inc.last_linear.in_features
        
#         self.inc.last_linear = nn.Linear(num_features,512)
            
# #         self.inc.aux_logits = False

            
#         for param in self.res.conv1.parameters():
#             param.requires_grad = False
#         for param in self.res.bn1.parameters():
#             param.requires_grad = False
#         for param in self.res.layer1.parameters():
#             param.requires_grad = False
#         for param in self.res.layer2.parameters():
#             param.requires_grad = False
#         for param in self.res.layer3.parameters():
#             param.requires_grad = False    
        
        
        
        
#         self.res.avgpool = nn.AvgPool2d(10)
        
#         num_features2 = self.res.fc.in_features
        
#         self.res.fc = nn.Linear(num_features2, 512)
        
                

#         lin3 = nn.Linear(1024,20)
      
#         self.fc = lin3
        
       
      
#     def forward(self, input):
#         x1 = self.res(input)
#         x2 = self.inc(input)
#         x = torch.cat((x1,x2),1)
#         return self.fc(x)

      