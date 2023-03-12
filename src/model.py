# import torch
import torch.nn as nn
import torchvision.models as models
import torch
from torch.autograd import Variable
import torch.nn.functional as F

device = ("cuda" if torch.cuda.is_available() else "cpu")


#Feature extract
class FeatureExtractor(nn.Module):
    def __init__(self, num_classes, device):
        super(FeatureExtractor, self).__init__()
        self.resnet = models.resnet50(pretrained=True) #for transfer learning
        self.resnet.fc = nn.Sequential(
                       nn.Linear(2048, 512))
        self.gradient = None

    def forward(self, x):

        # if x.requires_grad:
        #     #print(torch.nn.Sequential(*(list(self.resnet.children())[:-2])))
        #     #fex = torch.nn.Sequential(*(list(self.resnet.children())[:-2]))
        #     self.resnet.fc = nn.Sequential()
        #     self.resnet.avgpool = nn.Sequential()
        #     x_f = self.resnet(x)
        #     self.resnet.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1)).to(device)
        #     self.resnet.fc = nn.Sequential(nn.Linear(2048, 512)).to(device)
        #
        #     #print(x_f.shape)
        #     #print(self.resnet)
        #     x.register_hook(self.activations_hook)
        if x.requires_grad:
            x = self.resnet.conv1(x)
            x = self.resnet.bn1(x)
            x = self.resnet.relu(x)
            x = self.resnet.maxpool(x)

            x = self.resnet.layer1(x)
            x = self.resnet.layer2(x)
            x = self.resnet.layer3(x)
            x = self.resnet.layer4(x)
            x.register_hook(self.activations_hook)
            x = self.resnet.avgpool(x)
            x = self.resnet.fc(x.reshape(2, 2048))
        else:
            x = self.resnet(x)
        return x

    def activations_hook(self, grad):
        self.gradient = grad

    def get_activations_gradient(self):
        return self.gradient


# class FeatureMapExtractor(nn.Module):
#     def __init__(self, num_classes, device):
#         super(FeatureMapExtractor, self).__init__()
#         self.resnet = models.resnet50(pretrained=True) #for transfer learning
#
#     def forward(self, x):
#         x_r = self.resnet(x)
#         return x_r

#Recurrent Neural Network
class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout=[0,0]):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.dropout = dropout
        self.dense1 = torch.nn.Linear(hidden_dim, 64)
        self.dense2 = torch.nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = F.dropout(out[:,-1],self.dropout[0])
        out = self.relu(self.dense1(out))
        out = F.dropout(out,self.dropout[1])
        out = self.dense2(out)
        return out, h

class AccidentXai(nn.Module):
    def __init__(self, num_classes, x_dim, h_dim, z_dim, n_layers):
        super(AccidentXai,self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.num_classes = num_classes
        self.features = FeatureExtractor(num_classes, device)
        #self.featuremap = FeatureMapExtractor(num_classes, device)
        self.gru_net = GRUNet(h_dim+h_dim, h_dim, 2, n_layers, dropout=[0.5,0.0])
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, y, toa):
        losses = {'total_loss': 0}
        all_output, all_hidden, all_features = [], [], []
        h = Variable(torch.zeros(self.n_layers, x.size(0), self.h_dim))
        h = h.to(x.device)
        #print(x.size(1))
        for t in range(x.size(1)):
            x_t = self.features(x[:,t])
            #x_ft = self.features.featuremap(x[:,t])
            #all_features.append(x_ft)
            x_t = torch.unsqueeze(x_t,1)

            #print('x_t shape: ', x_t.shape)
            # Notes from the paper!
            # The feature vector x_t is given to GRU to learn
            # the hidden representation of the frame

            output, h = self.gru_net(x_t,h)

            #print(output, h)
            # computing losses
            L1 =self._exp_loss(output,y,t,toa=toa,fps=10.0)
            losses['total_loss']+=L1
            #print(output, "output in model")
            all_output.append(output) #TO-DO: all hidden
        return losses, all_output, all_features

    # def forward(self, x):
    #     all_output, all_hidden, all_features = [], [], []
    #     h = Variable(torch.zeros(self.n_layers, x.size(0), self.h_dim))
    #     h = h.to(x.device)
    #     for t in range(x.size(1)):
    #         x_t = self.features(x[:,t])
    #         x_ft = self.features.feature_map(x[:,t])
    #         all_features.append(x_ft)
    #         x_t = torch.unsqueeze(x_t,1)
    #
    #         output, h = self.gru_net(x_t,h)
    #
    #     return output

    def _exp_loss(self, pred, target, time, toa, fps=10.0):
            '''
            :param pred:
            :param target: onehot codings for binary classification
            :param time:
            :param toa:
            :param fps:
            :return:
            '''

            target_cls = target[:, 1]
            target_cls = target_cls.to(torch.long)
            penalty = -torch.max(torch.zeros_like(toa).to(toa.device, pred.dtype), (toa.to(pred.dtype) - time - 1) / fps)
            pos_loss = -torch.mul(torch.exp(penalty), -self.ce_loss(pred, target_cls))
            # negative example
            neg_loss = self.ce_loss(pred, target_cls)

            loss = torch.mean(torch.add(torch.mul(pos_loss, target[:, 1]), torch.mul(neg_loss, target[:, 0])))
            return loss
