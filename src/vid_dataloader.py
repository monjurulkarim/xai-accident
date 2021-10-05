import os
import glob

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from PIL import Image



class MySampler(torch.utils.data.Sampler):
    def __init__(self, end_idx, seq_length):
        indices = []
        for i in range(len(end_idx)-1):
            start = end_idx[i]
            end = end_idx[i+1] - seq_length
            indices.append(torch.arange(start, end))
        indices = torch.cat(indices)
        self.indices = indices

    def __iter__(self):
        indices = self.indices[torch.randperm(len(self.indices))]
        return iter(indices.tolist())

    def __len__(self):
        return len(self.indices)


class MyDataset(Dataset):
    def __init__(self, image_paths, seq_length, transform, length):
        self.image_paths = image_paths
        self.seq_length = seq_length
        self.transform = transform
        self.length = length

    def get_toa_all(self, data_path):
        toa_dict = {}
        annofile = os.path.join(data_path, 'videos', 'Crash-1500.txt')
        annoData = self.read_anno_file(annofile)
        for anno in annoData:
            labels = np.array(anno['label'], dtype=np.int)
            toa = np.where(labels == 1)[0][0]
            toa = min(max(1, toa), self.n_frames-1)
            toa_dict[anno['vid']] = toa
        return toa_dict

    def read_anno_file(self, anno_file):
        assert os.path.exists(anno_file), "Annotation file does not exist! %s"%(anno_file)
        result = []
        with open(anno_file, 'r') as f:
            for line in f.readlines():
                items = {}
                items['vid'] = line.strip().split(',[')[0]
                labels = line.strip().split(',[')[1].split('],')[0]
                items['label'] = [int(val) for val in labels.split(',')]
                assert sum(items['label']) > 0, 'invalid accident annotation!'
                others = line.strip().split(',[')[1].split('],')[1].split(',')
                items['startframe'], items['vid_ytb'], items['lighting'], items['weather'], items['ego_involve'] = others
                result.append(items)
        f.close()
        return result


    def __getitem__(self, index):
        start = index
        end = index + self.seq_length
        print('Getting images from {} to {}'.format(start, end))
        indices = list(range(start, end))
        images = []
        for i in indices:
            image_path = self.image_paths[i][0]
            image = Image.open(image_path)
            if self.transform:
                image = self.transform(image)
            images.append(image)
        x = torch.stack(images)
        if self.image_paths[start][1] == 1:
            label = 0
        else:
            label = 1
        # y = torch.tensor([self.image_paths[start][1]], dtype=torch.long)
        y = torch.tensor([label], dtype=torch.long)

        return x, y

    def __len__(self):
        return self.length
