import torch

def loadBoundPredDataset(movie_data):
    dataset = torch.cat((movie_data['place'][:-1], 
                         movie_data['cast'][:-1],
                         movie_data['action'][:-1],
                         movie_data['audio'][:-1],
                         movie_data['scene_transition_boundary_prediction'].unsqueeze(1),
                         movie_data['scene_transition_boundary_ground_truth'].unsqueeze(1)), dim=1)
    
    return dataset

def loadBoundPredDatasetA(movie_data):
    dataset = torch.cat((movie_data['action'][:-1],
                         movie_data['scene_transition_boundary_prediction'].unsqueeze(1),
                         torch.Tensor([0 for i in range(len(movie_data['place'])-1)]).unsqueeze(1),
                         movie_data['scene_transition_boundary_ground_truth'].unsqueeze(1)), dim=1)
    
    return dataset

def loadBoundGroundTruthDataset(movie_data):
    dataset = torch.cat((movie_data['place'][:-1], 
                         movie_data['cast'][:-1],
                         movie_data['action'][:-1],
                         movie_data['audio'][:-1],
                         movie_data['scene_transition_boundary_ground_truth'].unsqueeze(1)), dim=1)
    
    return dataset

def loadPredOnlyDataset(movie_data):
    dataset = torch.cat((movie_data['scene_transition_boundary_prediction'].unsqueeze(1),
                         movie_data['scene_transition_boundary_ground_truth'].unsqueeze(1)), dim=1)
    
    return dataset

def loadFullDataset(movie_data):
    dataset = torch.cat((movie_data['place'][:-1], 
                         movie_data['cast'][:-1],
                         movie_data['action'][:-1],
                         movie_data['audio'][:-1],
                         movie_data['scene_transition_boundary_prediction'].unsqueeze(1),
                         movie_data['shot_end_frame'].unsqueeze(1),
                         movie_data['scene_transition_boundary_ground_truth'].unsqueeze(1)), dim=1)
    
    return dataset

def loadDatasets(movie_data_list, datasetloader):
    dataset = torch.Tensor([])
    for movie_data in movie_data_list:
        dataset = torch.cat((dataset,
                             datasetloader(movie_data)), dim=0)
        
    return dataset

def loadDatasetsCol(movie_data_list, col, action='unsqueeze'):
    dataset = torch.Tensor([])
    for movie_data in movie_data_list:
        if action == 'unsqueeze':
            in_data = movie_data[col].unsqueeze(1)
        elif action == 'drop_last_idx':
            in_data = movie_data[col][:-1]
        else:
            in_data = movie_data[col]
        
        dataset = torch.cat((dataset,
                             in_data), dim=0)
        
    return dataset

def getTrainTestSet(dataset, split):
    train_set, test_set = torch.utils.data.random_split(dataset, split)
    
    return train_set, test_set

def getDataLoader(dataset, batch_size):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    
    return dataloader