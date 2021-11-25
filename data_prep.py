from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader as DLGeo
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch


data_training = ModelNet(root='.', name='10', train=True, transform=SamplePoints(5000))
data_testing = ModelNet(root='.', name='10', train=False, transform=SamplePoints(5000))


def points_to_raw(data_training, shuffle=False):
    dataloader = DLGeo(data_training, batch_size=1, shuffle=shuffle)

    raw_x, raw_y = [], []

    labels = data_training.raw_file_names

    for i, data in enumerate(dataloader):
        data_point = data.to_namedtuple()
        raw_x.append(data_point[0].numpy())
        raw_y.append(data_point[1].numpy()[0])
    return np.array(raw_x), np.array(raw_y), np.array(labels)

def points_to_box(data_training, box_width, stack_points=False, shuffle=False):
    dataloader = DLGeo(data_training, batch_size=1, shuffle=shuffle)

    out_x, raw_y = [], []

    labels = data_training.raw_file_names

    for i, data in enumerate(dataloader):
        data_row = data.to_namedtuple()
        raw_y.append(data_row[1].numpy()[0])

        data_point = data_row[0].numpy()
        scales = np.max(data_point, axis=0) - np.min(data_point, axis=0)
        max_scale = max(scales)
        data_point_scaled = (data_point - np.min(data_point, axis=0)+(max_scale - scales)/2) / max_scale * box_width
        data_point_scaled[data_point_scaled == box_width] = box_width - 0.00001
        data_mat = np.zeros(box_width ** 3).reshape((box_width, box_width, box_width))

        for c in range(len(data_point_scaled[:, 0])):
            if stack_points:
                data_mat[int(data_point_scaled[c, 0]), int(data_point_scaled[c, 1]), int(data_point_scaled[c, 2])] += 1
            else:
                data_mat[int(data_point_scaled[c, 0]), int(data_point_scaled[c, 1]), int(data_point_scaled[c, 2])] = 1
        out_x.append(data_mat)

        if i%100 == 0:
            print('Progress: ', i)

    return np.array(out_x), np.array(raw_y), np.array(labels)

def points_to_scale_ordered(data_training, box_width, shuffle=False):
    dataloader = DLGeo(data_training, batch_size=1, shuffle=shuffle)

    out_x, raw_y = [], []

    labels = data_training.raw_file_names

    for i, data in enumerate(dataloader):
        data_row = data.to_namedtuple()
        raw_y.append(data_row[1].numpy()[0])

        data_point = data_row[0].numpy()
        scales = np.max(data_point, axis=0) - np.min(data_point, axis=0)
        max_scale = max(scales)
        data_point_scaled = (data_point - np.min(data_point, axis=0) + (max_scale - scales) / 2) / max_scale * box_width

        distances = data_point_scaled[:,0] ** 2 + data_point_scaled[:,1] ** 2 + data_point_scaled[:,2] ** 2
        ranks = distances.argsort()
        data_point_scaled = data_point_scaled[ranks, :]

        out_x.append(data_point_scaled)

        if i % 100 == 0:
            print('Progress: ', i)

    return np.array(out_x), np.array(raw_y), np.array(labels)



box_width = 40
box, box_y, labels = points_to_box(data_training, box_width, stack_points=True, shuffle=True)
box_test, box_test_y, _ = points_to_box(data_testing, box_width, stack_points=True, shuffle=True)


dataset_train = TensorDataset(torch.Tensor(box), torch.Tensor(box_y))
dataset_test = TensorDataset(torch.Tensor(box_test), torch.Tensor(box_test_y))

train_loader = DataLoader(dataset_train, batch_size=128, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset_test, batch_size=128, shuffle=True, num_workers=0)
