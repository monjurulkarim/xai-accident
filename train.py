import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from src.model import AccidentXai
from src.vid_dataloader import MyDataset, MySampler
from tqdm import tqdm
import os
from tensorboardX import SummaryWriter
import glob

device = ("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

num_epochs = 25
num_classes = 2
learning_rate = 0.001
batch_size = 2
shuffle = True
pin_memory = True
num_workers = 1
train_data_path = '../dummy_data/train/'
test_data_path = '../dummy_data/test/'

#--------------train data----------------------------------------
train_class_paths = [d.path for d in os.scandir(train_data_path) if d.is_dir]

train_class_image_paths = []
train_end_idx = []
for c, class_path in enumerate(train_class_paths):
    for d in os.scandir(class_path):
        if d.is_dir:
            paths = sorted(glob.glob(os.path.join(d.path, '*.jpg')))
            # Add class idx to paths
            paths = [(p, c) for p in paths]
            train_class_image_paths.extend(paths)
            train_end_idx.extend([len(paths)])

train_end_idx = [0, *train_end_idx]
train_end_idx = torch.cumsum(torch.tensor(train_end_idx), 0)
seq_length = 49

train_sampler = MySampler(train_end_idx,seq_length)

##-------------Test data-------------------------------
test_class_paths = [d.path for d in os.scandir(test_data_path) if d.is_dir]

test_class_image_paths = []
test_end_idx = []
for c, class_path in enumerate(test_class_paths):
    for d in os.scandir(class_path):
        if d.is_dir:
            paths = sorted(glob.glob(os.path.join(d.path, '*.jpg')))
            # Add class idx to paths
            paths = [(p, c) for p in paths]
            test_class_image_paths.extend(paths)
            test_end_idx.extend([len(paths)])

test_end_idx = [0, *test_end_idx]
test_end_idx = torch.cumsum(torch.tensor(test_end_idx), 0)
seq_length = 49

test_sampler = MySampler(test_end_idx,seq_length)


train_data = MyDataset(image_paths= train_class_image_paths,
    seq_length=seq_length,
    transform=transform,
    length=len(train_sampler))

test_data = MyDataset(image_paths= test_class_image_paths,
    seq_length=seq_length,
    transform=transform,
    length=len(test_sampler))

train_dataloader = DataLoader(dataset= train_data, batch_size=batch_size,sampler=train_sampler)
test_dataloader = DataLoader(dataset= test_data, batch_size=batch_size, sampler=test_sampler)


def write_scalars(logger, epoch, loss):
    logger.add_scalars('train/loss',{'loss':loss}, epoch)

def write_test_scalars(logger, epoch, accuracy):
    # logger.add_scalars('test/loss',{'loss':loss}, epoch)
    logger.add_scalars('test/accuracy',{'accuracy':accuracy}, epoch)


def check_accuracy(loader, model):
    if loader == train_dataloader:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on validation data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            outputs = model(x.float())
            _, predictions = torch.max(outputs, 1)
            predictions = predictions.to(device)
            label = torch.squeeze(y)
            num_correct += (predictions == label).sum()
            num_samples += predictions.size(0)
    print('accuracy : ', float(num_correct)/float(num_samples)*100)
    print('===================================')
    # return f"{float(num_correct)/float(num_samples)*100:.2f}"
    return float(num_correct)/float(num_samples)*100

def custom_loss(predictions, labels):
    '''
    A custom loss: Currently CrossEntropyLoss
    '''
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    for pred in predictions:
        loss = criterion(pred, labels)
        total_loss += loss
    return total_loss



def train():
    model_dir ='../snapshot'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    logs_dir = '../logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    logger = SummaryWriter(logs_dir)

    model = AccidentXai(num_classes).to(device)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



    #Train only the fully connected layers
    for name, param in model.accident.named_parameters():
        if "fc.0.weight" in name or "fc.0.bias" in name:
            param.requires_grad = True
        elif "fc.2.weight" in name or "fc.2.bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


    model.train()
    best_acc = 0
    val_acc = 0
    for epoch in range(num_epochs):
        loop = tqdm(train_dataloader,total = len(train_dataloader), leave = True)
        # if epoch % 2 == 0:
        #     val_acc = check_accuracy(test_dataloader,model)
        #     write_test_scalars(logger,epoch,val_acc)
            # loop.set_postfix(val_acc=)
        for imgs, labels, toa in loop:
            loop.set_description(f"Epoch  [{epoch+1}/{num_epochs}]")
            imgs = imgs.to(device)
            labels = torch.squeeze(labels)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = custom_loss(outputs, labels)
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss = loss.item())

        # save model
        write_scalars(logger,epoch,loss)
        model_file = os.path.join(model_dir, 'crack_model.pth')
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(),model_file)
    logger.close()

if __name__ == "__main__":
    train()
