#!/usr/bin/env python
# -*- coding: utf-8 -*-
# create SIMdataLoader by reading SIMdata
# author：zenghui time:2020/3/2

import torch
from torchvision import transforms
from torch.utils import data
from PIL import Image
from torch.utils.data import DataLoader
import json

class SIM_data(data.Dataset):
    def __init__(self,
                 directory_data_file,
                 ):
        # data_dict=[]
        # with open(directory_data_file,'r',encoding='utf8') as json_file:
        #     for line in json_file.readlines():
        #         dic = json.loads(line)
        #         data_dict.append(dic)
        #     print(data_dict)
        with open(directory_data_file, 'r') as txtFile:
            self.content = txtFile.readlines()


    def __getitem__(self, index):
        txt_line = self.content[index]
        image_directoty = txt_line.split()[0]
        wave_vector = [float(txt_line.split()[1]),float(txt_line.split()[2])]
        phi = float(txt_line.split()[3])
        SIMdata_image = Image.open(image_directoty)
        SIMdata_image_tensor = transforms.ToTensor()(SIMdata_image)
        wave_vector_phase_tensor = torch.tensor(wave_vector+[phi], dtype=torch.float)
        return SIMdata_image_tensor,wave_vector_phase_tensor

    def __len__(self):
        return len(self.content)

if __name__ == '__main__':
    directory_json_file = "D:\DataSet\DIV2K\DIV2K_valid_LR_unknown\\test\directories_of_images.json"
    directory_txt_file = 'D:\DataSet\DIV2K\DIV2K_valid_LR_unknown\\test\directories_of_images.txt'
    SIM_dataset=SIM_data(directory_txt_file)
    SIM_data_loader= DataLoader(SIM_dataset, batch_size=4,shuffle=True)
