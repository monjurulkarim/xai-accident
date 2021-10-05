import os
import glob

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from PIL import Image
from src.vid_dataloader import MySampler, MyDataset

root_dir = '../dummy_data/train/'
class_paths = [d.path for d in os.scandir(root_dir) if d.is_dir]


class_image_paths = []
end_idx = []
for c, class_path in enumerate(class_paths):
    for d in os.scandir(class_path):
        if d.is_dir:
            paths = sorted(glob.glob(os.path.join(d.path, '*.jpg')))
            # Add class idx to paths
            paths = [(p, c) for p in paths]
            class_image_paths.extend(paths)
            end_idx.extend([len(paths)])

end_idx = [0, *end_idx]
end_idx = torch.cumsum(torch.tensor(end_idx), 0)
seq_length = 49


sampler = MySampler(end_idx, seq_length) #Optional
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])


dataset = MyDataset(
    image_paths=class_image_paths,
    seq_length=seq_length,
    transform=transform,
    length=len(sampler))

loader = DataLoader(
    dataset,
    batch_size=1,
    sampler=sampler
)

print(loader)

count = 0
for data, target in loader:
    count+=1
    print(data.size(1))
    # print(data.shape)
print('=======================')
print(count)
