import glob
import torch
import pytorch_grad_cam
from natsort import natsorted
from src.model import AccidentXai
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from src.vid_dataloader import MySampler, MyDataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

best_model_path = "../snapshot/saved_model_00.pth"

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = ("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

num_classes = 2
x_dim = 2048
h_dim = 256
z_dim = 128
n_layers = 1

batch_size = 10
num_workers = 10

# Load saved model


model = AccidentXai(num_classes, x_dim, h_dim, z_dim,n_layers).to(device)
model.load_state_dict(torch.load(best_model_path))

test_data_path = '../data/test/'

# -------------Test data-------------------------------
test_class_paths = [d.path for d in os.scandir(test_data_path) if d.is_dir]

test_class_image_paths = []
test_end_idx = []
for c, class_path in enumerate(test_class_paths):
    for d in os.scandir(class_path):
        if d.is_dir:
            paths = natsorted(glob.glob(os.path.join(d.path, '*.jpg')))
            # Add class idx to paths
            paths = [(p, c) for p in paths]
            test_class_image_paths.extend(paths)
            test_end_idx.extend([len(paths)])

test_end_idx = [0, *test_end_idx]
test_end_idx = torch.cumsum(torch.tensor(test_end_idx), 0)
seq_length = 49

test_sampler = MySampler(test_end_idx,seq_length)

test_data = MyDataset(image_paths= test_class_image_paths,
    seq_length=seq_length,
    transform=transform,
    length=len(test_sampler))

test_dataloader = DataLoader(dataset= test_data, batch_size=batch_size, sampler=test_sampler)

# Tuun evaluation mode on in the model
all_pred = []
all_labels = []
losses_all = []
all_toas = []

with torch.no_grad():
    loop = tqdm(test_dataloader,total = len(test_dataloader), leave = True)
    for imgs, labels, toa in loop:
        imgs = imgs.to(device)
        labels = torch.squeeze(labels)
        labels = labels.to(device)
        # outputs = model(imgs)
        loss, outputs = model(imgs,labels,toa)
        loss = loss['total_loss'].item()
        losses_all.append(loss)
        num_frames = imgs.size()[1]
        batch_size = imgs.size()[0]
        pred_frames = np.zeros((batch_size,num_frames),dtype=np.float32)
        for t in range(num_frames):
            pred = outputs[t]
            pred = pred.cpu().numpy() if pred.is_cuda else pred.detach().numpy()
            pred_frames[:, t] = np.exp(pred[:, 1]) / np.sum(np.exp(pred), axis=1)

        #gather results and ground truth
        all_pred.append(pred_frames)
        label_onehot = labels.cpu().numpy()
        label = np.reshape(label_onehot[:, 1], [batch_size,])
        all_labels.append(label)
        toas = np.squeeze(toa.cpu().numpy()).astype(np.int)
        all_toas.append(toas)
        # loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
        loop.set_postfix(val_loss = sum(losses_all))

        all_pred = np.vstack((np.vstack(all_pred[0][:-1]), all_pred[0][-1]))
        all_labels = np.hstack((np.hstack(all_labels[0][:-1]), all_labels[0][-1]))
        all_toas = np.hstack((np.hstack(all_toas[0][:-1]), all_toas[0][-1]))

        # print(all_pred, all_labels, all_toas, losses_all)
        break

plt.plot(all_pred[8])

plt.show()




