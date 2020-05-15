#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/5/8

import torch
import math
from torchvision import transforms
from Add_Sinpattern_OTF_noise_ByAugumentor import SinusoidalPattern
import Pipeline
import random


def AddSinusoidalPattern(pipeline, probability=1):
    """
    The function is used to add sinusoidal pattern and OTF on the images in pipeline
    :param pipeline: The image pipeline based on module 'Augmentor'
    :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
    :type probability: Float
    :return: None
    """
    if not 0 < probability <= 1:
        raise ValueError(Pipeline._probability_error_text)
    else:
        pipeline.add_operation(SinusoidalPatternWithLabel(probability=probability))


class SinusoidalPatternWithLabel(SinusoidalPattern):

    def _init_(self, directory_txt_file=None):
        self.directory_txt_file = directory_txt_file
        SinusoidalPatternWithLabel._init_(self, probability=1, NumPhase=3, Magnification=150, PixelSizeOfCCD=6800,
                                          EmWaveLength=635, NA=0.9, SNR=500,
                                          image_size=256)

    def perform_operation(self, images):
        """
        Crop the passed :attr:`images` by percentage area, returning the crop as an
        image.

        :param images: The image(s) to crop an area from.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """
        crop_size = self.image_size
        SIMdata_images = []
        for image in images:
            h, w = image.size
            pad_w = max(crop_size - w, 0)
            pad_h = max(crop_size - h, 0)
            img_pad = transforms.Pad(padding=(0, 0, pad_w, pad_h), fill=0, padding_mode='constant')(image)
            center_crop = transforms.CenterCrop(size=(crop_size, crop_size))
            imag_pad_crop = center_crop(img_pad)

            # augmented_images += self.LR_image_generator(imag_pad_crop)
            SIM_images, wave_vector, phi = self.SinusoidalPattern(imag_pad_crop)

            SIMdata_images += self.SR_image_generator(imag_pad_crop)
            SIMdata_images += self.LR_image_generator(imag_pad_crop)
            SIMdata_images += SIM_images

        return SIMdata_images, wave_vector, phi

    def SinusoidalPattern(self, image):
        '''
        :param image:  PIL_Image that will be loaded pattern on
        :param NumPhase:  Number of phase
        :return: SinusoidalPatternImage: Image which loaded sinusoidal pattern
        '''
        resolution = 0.61 * self.EmWaveLength / self.NA
        # xx, yy, _, _ = self.GridGenerate(image=torch.rand(7, 7))
        # xx, yy, fx, fy = self.GridGenerate(image)
        TensorImage = transforms.ToTensor()(image)
        SinPatternPIL_Image = []
        wave_vector = []
        phi = []
        initial_theta = random.uniform(0, 1) * math.pi * 2 / 3
        for i in range(3):
            theta = i * 2 / 3 * math.pi + initial_theta
            SpatialFrequencyX = -0.8 * 1 / resolution * math.sin(theta)  # 0.8倍的极限频率条纹，可调
            SpatialFrequencyY = -0.8 * 1 / resolution * math.cos(theta)
            for j in range(self.NumPhase):
                initial_phase = random.uniform(0, 1) * math.pi * 2 / self.NumPhase
                phase = initial_phase + j * 2 / self.NumPhase * math.pi
                SinPattern = (torch.cos(
                    phase + 2 * math.pi * (SpatialFrequencyX * self.xx + SpatialFrequencyY * self.yy)) + 1) / 2
                SinPattern_OTF_filter = self.OTF_Filter(SinPattern * TensorImage, self.OTF)
                SinPattern_OTF_filter_gaussian_noise = self.add_gaussian_noise(SinPattern_OTF_filter)
                SinPatternPIL_Image.append(SinPattern_OTF_filter_gaussian_noise)
                wave_vector.append([SpatialFrequencyX, SpatialFrequencyY])
                phi.append(phase)

        return SinPatternPIL_Image, wave_vector, phi


if __name__ == '__main__':
    # SourceFileDirectory = "/home/zenghui19950202/SRdataset/test"
    # directory_txt_file = '/home/zenghui19950202/SRdataset/test/directories.txt'

    directory_txt_file = "/home/zenghui19950202/SRdataset/test/valid.txt"
    SourceFileDirectory = "/home/zenghui19950202/SRdataset/test/valid"
    # directory_json_file = "D:\DataSet\DIV2K\DIV2K_valid_LR_unknown\\test\directories_of_images.json"

    p = Pipeline.Pipeline_revise(source_directory=SourceFileDirectory, json_directory=directory_txt_file)
    AddSinusoidalPattern(p, probability=1)
    p.process()

#
# with open(directory_file,'r') as txtFile:
#     content = txtFile.readlines()
#     dataLen = len(content)
#     for i in dataLen:
#         txt_line=content[i]
#         directory_of_image=txt_line.split()[0]
#
#
#         # lr_images = np.array([io.imread(join(imset_dir, f'LR{i}.png')) for i in range(9)], dtype=np.uint16)
#
#
#         PIL_image=Image.open(directory_of_image)
