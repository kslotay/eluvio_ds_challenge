import os
from tqdm import tqdm
import pickle

def load_files(directory):
    data = []
    file_names = []
    for root,dirs,files in os.walk(directory):
        for file in tqdm(files):
            file_names.append(file)
            data.append(pickle.load(open(os.path.join(directory,file), 'rb')))
            
    return file_names, data