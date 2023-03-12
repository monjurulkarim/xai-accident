import glob
import torch
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, \
    AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from natsort import natsorted
from src.model import AccidentXai
import os
from torch.utils.data import DataLoader, SequentialSampler
import torchvision.transforms as transforms
from src.vid_dataloader import MySampler, MyDataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import utils

import shap

best_model_path = "../snapshot/best_model.pth"

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = ("cuda" if torch.cuda.is_available() else "cpu")

torch.cuda.set_per_process_memory_fraction(0.5, 0)
torch.cuda.empty_cache()

total_memory = torch.cuda.get_device_properties(0).total_memory
tmp_tensor = torch.empty(int(total_memory * 0.499), dtype=torch.int8,
                         device='cuda')
del tmp_tensor
torch.cuda.empty_cache()

U = 112
V = 112
transform = transforms.Compose(
    [
        transforms.Resize((U, V)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

num_classes = 2
x_dim = 2048
h_dim = 256
z_dim = 128
n_layers = 1

batch_size = 2
num_workers = 10

# Load saved model
model = AccidentXai(num_classes, x_dim, h_dim, z_dim, n_layers).to(device)
model.load_state_dict(torch.load(best_model_path))

# Currently works using test dataloader
# Must be modified for custom dataset use later
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

test_sampler = MySampler(test_end_idx, seq_length)

test_data = MyDataset(image_paths=test_class_image_paths,
                      seq_length=seq_length,
                      transform=transform,
                      length=len(test_sampler))

test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size)

# Turn on evaluation mode on in the model


def get_predictions(model):
    model.eval()
    all_pred = []
    all_labels = []
    losses_all = []
    all_toas = []

    with torch.no_grad():
        loop = tqdm(test_dataloader, total=len(test_dataloader), leave=True)
        for imgs, labels, toa in loop:
            imgs = imgs.to(device)
            # print(imgs.shape)
            # plt.imshow(np.rot90(imgs[1][38].cpu().detach().numpy().T, -1))
            labels = torch.squeeze(labels)
            labels = labels.to(device)
            # outputs = model(imgs)
            loss, outputs, _ = model(imgs, labels, toa)
            loss = loss['total_loss'].item()
            losses_all.append(loss)
            num_frames = imgs.size()[1]
            batch_size = imgs.size()[0]
            pred_frames = np.zeros((batch_size, num_frames), dtype=np.float32)

            for t in range(num_frames):
                pred = outputs[t]
                pred = pred.cpu().numpy() if pred.is_cuda else pred.detach().numpy()
                # Equation 6 in paper.
                pred_frames[:, t] = np.exp(pred[:, 1]) / np.sum(np.exp(pred),
                                                                axis=1)

            # gather results and ground truth
            all_pred.append(pred_frames)
            label_onehot = labels.cpu().numpy()
            label = np.reshape(label_onehot[:, 1], [batch_size, ])
            all_labels.append(label)
            toas = np.squeeze(toa.cpu().numpy()).astype(np.int)
            all_toas.append(toas)
            # loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(val_loss=sum(losses_all))

            all_pred = np.vstack(
                (np.vstack(all_pred[0][:-1]), all_pred[0][-1]))
            all_labels = np.hstack(
                (np.hstack(all_labels[0][:-1]), all_labels[0][-1]))
            all_toas = np.hstack(
                (np.hstack(all_toas[0][:-1]), all_toas[0][-1]))
            break
            # print(all_pred, all_labels, all_toas, losses_all)
            # Loop breaks after one cycle, otherwise loop throws an error.
    return all_pred, all_labels, all_toas


def forward_recorder(module, input, output):
    list_forward.append(output.data.cpu())


def backward_recorder(module, grad_in, grad_out):
    list_backward.append(grad_out[0].data.cpu())


def weights_calculator(grads):
    # normalize the weights first
    grads = grads / (torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5)
    return torch.nn.AvgPool2d(grads.size()[2:])(grads)


def aggregate_feature_weights(input_weights, input_features):
    # Aggregate Feature map with weights
    # 2, 3, 56, 56 ******** 1, 512
    print("------------------------------------------------------------------")
    print("grad cam input shapes")
    print(input_features.shape)
    print("------------------------------------------------------------------")
    aggre = torch.tensor(input_features) * input_weights.cpu()
    f_rl = torch.relu(aggre)
    f_inter = f_rl
    # f_inter = nn.functional.interpolate(
    #     torch.cat((torch.tensor([1.0, 1.0]), f_rl)), size=(U, V),
    #     mode='bilinear')
    return f_inter


pred, _, _ = get_predictions(model)

# Grad CAM
model = AccidentXai(num_classes, x_dim, h_dim, z_dim, n_layers).to(device)
model.load_state_dict(torch.load(best_model_path))

print(model.features.resnet)

feature_extractor = torch.nn.Sequential(*(list(model.features.resnet.children())[:-2]))

model.train()

loop = tqdm(test_dataloader, total=len(test_dataloader), leave=True)
for imgs, labels, toa in loop:
    imgs = imgs.to(device)
    imgs.requires_grad_()
    labels = torch.squeeze(labels)
    labels = labels.to(device)

    loss, outputs, features = model(imgs, labels, toa)
    L = loss['total_loss']

    # Process each of the 50 frames
    num_frames = imgs.size()[1]
    # Process only batch number 0
    imgs = imgs[0]

    fig, axs = plt.subplots(5, 10)

    t_10 = np.arange(0, 10)
    t_10 = np.tile(t_10, 5)

    t_5 = []
    for a in range(0, 5, 1):
        for a_10 in range(10):
            t_5.append(a)

    activation = feature_extractor(imgs)
    # Image frame loop
    for t in range(num_frames):
        output = outputs[t][0]
        # Backward gradient descent
        output.backward(torch.tensor([1.0, 0.0]).to(device), retain_graph=True)
        # Get feature map
        feature_map = activation[t].cpu().detach()  # Select batch 1
        # Get gradient from model
        gradient = model.features.get_activations_gradient()
        gradient = gradient[0].cpu()

        channel_amount = len(feature_map[:])
        pooled_gradients = torch.mean(gradient, dim=[1, 2])

        for i in range(channel_amount):
            feature_map[i, :, :] *= pooled_gradients[i]

        # Average channelswise
        heatmap = torch.mean(feature_map, dim=0).squeeze()
        # Relu
        heatmap = torch.relu(torch.tensor(heatmap))
        # Normalization
        heatmap /= torch.max(heatmap)
        heatmap = np.uint8(255 * heatmap)

        #plt.matshow(heatmap.squeeze())

        ii = np.rot90(imgs[t].cpu().detach().numpy().T, -1)
        dx, dy = 0.05, 0.05
        x = np.arange(-3.0, 3.0, dx)
        y = np.arange(-3.0, 3.0, dy)
        X, Y = np.meshgrid(x, y)
        extent = np.min(x), np.max(x), np.min(y), np.max(y)

        t1 = t_5[t]
        t2 = t_10[t]
        #
        # if t1 == 2:
        #     break

        axs[t1][t2].set_title(pred[0][t])
        axs[t1][t2].imshow(ii + 0.55, interpolation='nearest', extent=extent)
        axs[t1][t2].imshow(heatmap, extent=extent, cmap=plt.cm.inferno, alpha=.65, interpolation='bilinear')

    break

plt.show()
