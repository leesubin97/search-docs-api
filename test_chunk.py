import numpy as np
import json

npy_data = np.load('meta_data.npy', allow_pickle=True)

data_list = npy_data.tolist()

with open('meta_data.json', 'w', encoding='utf-8') as json_file:
    json.dump(data_list, json_file, ensure_ascii=False, indent=4)
