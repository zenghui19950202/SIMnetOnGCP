#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/3/10

import torch
import math
import numpy as np
from Augmentor import Operations
# from Augmentor import Pipeline
import Pipeline
from torchvision import transforms
from PIL import Image
import os



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
        pipeline.add_operation(SinusoidalPattern(probability=probability))

class SinusoidalPattern(Operations.Operation):
    """
    This class is used to add Sinusoidal Pattern on images.
    """
    def __init__(self,probability, NumPhase=3, Magnification=150, PixelSizeOfCCD=6800, EmWaveLength=635, NA=0.9, SNR=500,
                 image_size=256):  # unit: nm
        """
        As well as the always required :attr:`probability` parameter, the
        constructor requires a :attr:`percentage_area` to control the area
        of the image to crop in terms of its percentage of the original image,
        and a :attr:`centre` parameter toggle whether a random area or the
        centre of the images should be cropped.

        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :type probability: Float
        """
        self.Magnification = Magnification
        self.PixelSizeOfCCD = PixelSizeOfCCD
        self.EmWaveLength = EmWaveLength
        self.NA = NA
        self.NumPhase = NumPhase
        self.SNR = SNR
        self.image_size = image_size
        self.xx, self.yy, fx, fy = self.GridGenerate(image_size)
        self.f0 = 2 * NA / EmWaveLength  # The coherent cutoff frequency
        self.f = pow((fx ** 2 + fy ** 2), 1 / 2)  # The spatial freqneucy fr=sqrt( fx^2 + fy^2 )

        self.OTF = self.OTF_form(fc_ratio=1)
        Operations.Operation.__init__(self, probability)

    def perform_operation(self, images):
        """
        Crop the passed :attr:`images` by percentage area, returning the crop as an
        image.

        :param images: The image(s) to crop an area from.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """
        crop_size=self.image_size
        augmented_images = []
        for image in images:
            h, w = image.size
            pad_w = max(crop_size - w, 0)
            pad_h = max(crop_size - h, 0)
            img_pad = transforms.Pad(padding=(0, 0, pad_w, pad_h), fill=0, padding_mode='constant')(image)
            center_crop = transforms.CenterCrop(size=(crop_size, crop_size))
            imag_pad_crop = center_crop(img_pad)

            # augmented_images += self.LR_image_generator(imag_pad_crop)
            augmented_images += self.SR_image_generator(imag_pad_crop)
            augmented_images += self.LR_image_generator(imag_pad_crop)
            augmented_images += self.SinusoidalPattern(imag_pad_crop)
            # print('hello')

        return augmented_images

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
        for i in range(3):
            theta = i * 2 / 3 * math.pi + math.pi / 2
            SpatialFrequencyX = -0.8 * 1 / resolution * math.sin(theta)  # 0.8倍的极限频率条纹，可调
            SpatialFrequencyY = -0.8 * 1 / resolution * math.cos(theta)
            for j in range(self.NumPhase):
                phase = j * 2 / self.NumPhase * math.pi
                SinPattern = (torch.cos(
                    phase + 2 * math.pi * (SpatialFrequencyX * self.xx + SpatialFrequencyY * self.yy)) + 1) / 2
                SinPattern_OTF_filter = self.OTF_Filter(SinPattern * TensorImage,self.OTF)
                SinPattern_OTF_filter_gaussian_noise = self.add_gaussian_noise(SinPattern_OTF_filter)
                SinPatternPIL_Image.append(SinPattern_OTF_filter_gaussian_noise)

        return SinPatternPIL_Image

    def GridGenerate(self, image_size):
        '''
        :param Magnification: the magnification of the Microscope
        :param PixelSize: the PixleSize of the sCMOS or CCD
        :param EmWaveLength:  emission wavelength of sample
        :param NA:  NA(numerical aperture) of the objective
        :return:
        '''
        y, x = image_size, image_size
        PixelSize = self.PixelSizeOfCCD / self.Magnification
        delta_x = PixelSize  # xy方向的空域像素间隔，单位m
        delta_y = PixelSize
        delta_fx = 1 / x / delta_x  # xy方向的频域像素间隔，单位m ^ -1
        delta_fy = 1 / y / delta_y
        if x % 2 == 1:
            if y % 2 == 1:
                xx, yy = torch.meshgrid(torch.arange(-(x - 1) / 2, (x + 1) / 2, 1),
                                        torch.arange(-(y - 1) / 2, (y + 1) / 2, 1))  # 空域x方向坐标为奇数，y方向坐标为奇数的情况
            else:
                xx, yy = torch.meshgrid(torch.arange(-(x - 1) / 2, (x + 1) / 2, 1),
                                        torch.arange(-y / 2, y / 2 - 1, 1))  # 空域x方向坐标为奇数，y方向坐标为偶数的情况
        else:
            if y % 2 == 1:
                xx, yy = torch.meshgrid(torch.arange(-x / 2, x / 2, 1),
                                        torch.arange(-(y - 1) / 2, (y + 1) / 2, 1))  # 空域x方向坐标为偶数，y方向坐标为奇数的情况
            else:
                xx, yy = torch.meshgrid(torch.arange(-x / 2, x / 2, 1),
                                        torch.arange(-y / 2, y / 2, 1))  # 空域x方向坐标为偶数，y方向坐标为偶数的情况
        fx = xx * delta_fx
        fy = yy * delta_fy
        xx = xx * delta_x
        yy = yy * delta_y
        return xx, yy, fx, fy

    def OTF_Filter(self, image, OTF):
        image = image.squeeze()
        if len(image.size()) == 3:
            image_PIL = transforms.ToPILImage()(image).convert('RGB')
            NumpyImage = np.asarray(image_PIL)
        else:
            image_PIL = transforms.ToPILImage()(image).convert('L')
            NumpyImage = np.asarray(image_PIL)
        FFT_NumpyImage = np.fft.fft2(NumpyImage, axes=(0, 1))
        # def OTF_Filter(image):
        # image_PIL = transforms.ToPILImage()(image).convert('RGB')
        # _, _, fx, fy = self.GridGenerate(image=image_PIL)
        dim = OTF.shape

        if len(FFT_NumpyImage.shape) == 3:
            FilterFFT_NumpyImage = np.fft.fftshift(FFT_NumpyImage, axes=(0, 1)) * OTF.numpy().reshape(dim[0], dim[1],
                                                                                                      1)  # reshape the OTF to fullfill the broadcast mechanism
        elif len(FFT_NumpyImage.shape) == 2:
            FilterFFT_NumpyImage = np.fft.fftshift(FFT_NumpyImage, axes=(0, 1)) * OTF.numpy()
        else:
            raise Exception('The dimensions of input images must be 2 or 3 ')

        Filter_NumpyImage = np.fft.ifft2(np.fft.ifftshift(FilterFFT_NumpyImage, axes=(0, 1)), axes=(0, 1))
        Filter_NumpyImage = abs(Filter_NumpyImage)

        if len(image.size()) == 3:
            Filter_PILImage = Image.fromarray(Filter_NumpyImage.astype('uint8')).convert('RGB')
        else:
            Filter_PILImage = Image.fromarray(Filter_NumpyImage.astype('uint8')).convert('L')
        return Filter_PILImage

    def add_gaussian_noise(self, PIL_Image):  # The type of input image is PIL
        TensorImage = transforms.ToTensor()(PIL_Image)
        # if len(TensorImage)==3:
        #      TensorImage = TensorImage.permute(1, 2, 0) # transope for matplot
        signal_intensity_of_image = (TensorImage ** 2).mean()  # The mean intensity of signal
        noise_intensity_of_image = signal_intensity_of_image / self.SNR
        noise_of_image = torch.zeros_like(TensorImage)
        std_of_noise = noise_intensity_of_image ** (0.5)
        noise_of_image.normal_(mean=0, std=std_of_noise)
        image_with_noise = TensorImage + noise_of_image
        image_with_noise = torch.where(image_with_noise < 0, torch.zeros_like(image_with_noise), image_with_noise)
        image_with_noise_normalized = image_with_noise / image_with_noise.max()
        image_PIL_new = transforms.ToPILImage()(image_with_noise_normalized).convert('RGB')
        return image_PIL_new
        # return image_with_noise_normalized

    def SR_image_generator(self, image):
        TensorImage = transforms.ToTensor()(image)
        TensorImage = TensorImage.type(torch.FloatTensor)
        OTF = self.OTF_form(fc_ratio=1.9)
        SR_image_PIL= self.add_gaussian_noise(self.OTF_Filter(TensorImage,OTF))
        # SR_image_Tensor = self.add_gaussian_noise(self.OTF_Filter(TensorImage, OTF))
        # SR_image_Tensor = transforms.ToTensor()(SR_image_PIL)
        return [SR_image_PIL]
        # return SR_image_Tensor

    def LR_image_generator(self, image):
        TensorImage = transforms.ToTensor()(image)
        TensorImage = TensorImage.type(torch.FloatTensor)
        OTF = self.OTF_form(fc_ratio=1)
        LR_image_PIL= self.add_gaussian_noise(self.OTF_Filter(TensorImage,OTF))
        return [LR_image_PIL]

    def OTF_form(self, fc_ratio=1):
        f0 = fc_ratio * self.f0
        f = self.f
        OTF = torch.where(f < f0, (2 / math.pi) * (torch.acos(f / f0) - (f / f0) * (
            pow((1 - (self.f / f0) ** 2), 0.5))), torch.Tensor([0]))  # Caculate the OTF support
        # OTF = torch.where(f < f0,torch.ones_like(f),torch.zeros_like(f))
        return OTF

if __name__ == '__main__':
    directory_txt_file="D:\DataSet\DIV2K\DIV2K_valid_LR_unknown\\test\\valid.txt"
    SourceFileDirectory="D:\DataSet\DIV2K\DIV2K_valid_LR_unknown\\test\\valid"
    # p = Pipeline.Pipeline_revise(source_directory=SourceFileDirectory,txt_directory=directory_txt_file)

    p = Pipeline.Pipeline_revise(source_directory=SourceFileDirectory)


    AddSinusoidalPattern(p,probability=1)

    with open(directory_txt_file,'w') as f:
        for augmentor_image in p.augmentor_images:
            directories_of_images=os.path.basename(augmentor_image.image_path)
            f.write(augmentor_image.output_directory + '\t' + directories_of_images +'\n')
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