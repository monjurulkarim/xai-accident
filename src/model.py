# import torch
import torch.nn as nn
import torchvision.models as models
import torch

device = ("cuda" if torch.cuda.is_available() else "cpu")

class AccidentClassifier(nn.Module):
    def __init__(self, num_classes, device):
        super(AccidentClassifier,self).__init__()
        self.resnet = models.resnet50(pretrained=True) #for transfer learning


        self.resnet.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, num_classes),
               nn.Softmax(dim=1)).to(device)


    def forward(self,x):
        x = self.resnet(x)
        return x




class AccidentXai(nn.Module):
    def __init__(self, num_classes):
        super(AccidentXai,self).__init__()

        self.num_classes= num_classes
        self.accident = AccidentClassifier(num_classes, device)

    def forward(self,x):
        all_output = []
        for t in range(x.size(1)):
            output = self.accident(x[:, t])
            all_output.append(output)
        return all_output
