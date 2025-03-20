import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os

class ReadDataset(Dataset):
    def __init__(self,  source):
     
        self.data = torch.from_numpy(source).float()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def RandomSplit(datasets, train_set_percentage):
    lengths = [int(len(datasets)*train_set_percentage), len(datasets)-int(len(datasets)*train_set_percentage)]
    return random_split(datasets, lengths)

def GetDataLoaders(npArray, batch_size, train_set_percentage = 0.7, shuffle=True, num_workers=0, pin_memory=True):
    
    
    pc = ReadDataset(npArray)

    train_set, test_set = RandomSplit(pc, train_set_percentage)

    train_loader = DataLoader(train_set, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory)
    
    return train_loader, test_loader

def CondensePointClouds(input_folder, expected_num_points, x_range=(0,10), y_range=(-5,5), z_range=(-5,1)):
    all_point_clouds = []
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".npz"):
            file_path = os.path.join(input_folder, filename)
            data = np.load(file_path)
            
            point_cloud = data['points']
            
            mask = (
                (point_cloud[:, 0] >= x_range[0]) & (point_cloud[:, 0] <= x_range[1]) &
                (point_cloud[:, 1] >= y_range[0]) & (point_cloud[:, 1] <= y_range[1]) &
                (point_cloud[:, 2] >= z_range[0]) & (point_cloud[:, 2] <= z_range[1])
            )
            
            filtered_points = point_cloud[mask]
            
            if len(filtered_points) > expected_num_points:
                sampled_indices = np.random.choice(len(filtered_points), expected_num_points, replace=False)
                sampled_points = filtered_points[sampled_indices]
            else:
                print("Padding required - may need to modify expected num points")
                padding_shape = (expected_num_points - len(filtered_points), 3)
                sampled_points = np.pad(filtered_points, ((0, padding_shape[0]), (0,0)), 'constant', constant_values=0)
                
            all_point_clouds.append(sampled_points)
    
    combined_data = np.stack(all_point_clouds, axis=0)
    
    return combined_data
