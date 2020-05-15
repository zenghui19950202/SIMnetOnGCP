#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/5/6
from Augmentor.Pipeline import Pipeline
from PIL import Image
import random
import os
import json


class Pipeline_revise(Pipeline):

    def __init__(self, source_directory=None, output_directory="output", save_format=None, json_directory=None):
       super(Pipeline_revise,self).__init__(source_directory=source_directory, output_directory=output_directory, save_format=save_format)
       self.json_directory=json_directory

    def _execute(self, augmentor_image, save_to_disk=True, multi_threaded=True):
        """
        Private method. Used to pass an image through the current pipeline,
        and return the SIM images, and  write image directories, wave vectors, phi into a json file  .

        The returned image can then either be saved to disk or simply passed
        back to the user. Currently this is fixed to True, as Augmentor
        has only been implemented to save to disk at present.

        :param augmentor_image: The image to pass through the pipeline.
        :param save_to_disk: Whether to save the image to disk. Currently
         fixed to true.
        :type augmentor_image: :class:`ImageUtilities.AugmentorImage`
        :type save_to_disk: Boolean
        :return: The augmented image.
        """

        images = []

        if augmentor_image.image_path is not None:
            images.append(Image.open(augmentor_image.image_path))

        # What if they are array data?
        if augmentor_image.pil_images is not None:
            images.append(augmentor_image.pil_images)

        if augmentor_image.ground_truth is not None:
            if isinstance(augmentor_image.ground_truth, list):
                for image in augmentor_image.ground_truth:
                    images.append(Image.open(image))
            else:
                images.append(Image.open(augmentor_image.ground_truth))

        for operation in self.operations:
            r = round(random.uniform(0, 1), 1)
            if r <= operation.probability:
                images, wave_vector, phi = operation.perform_operation(images)

        # TEMP FOR TESTING
        # save_to_disk = False


        # with open(self.json_directory, 'a') as f:
        #     for i in range(len(phi)):
        #         save_name = "SIMdata(" + str(i+1) + ")_" + os.path.basename(augmentor_image.image_path)
        #         save_directory = os.path.join(augmentor_image.output_directory, save_name)
        #         dictObj = {'data'+ str(i+1):{ 'save_directory': save_directory, 'wave_vector': wave_vector[i], 'phi': phi[i]}}
        #         jsObj = json.dumps(dictObj, indent=4)
        #         f.write(jsObj)

        with open(self.json_directory, 'a') as f:
            for i in range(len(phi)):
                save_name = "SIMdata(" + str(i+1) + ")_" + os.path.basename(augmentor_image.image_path)
                save_directory = os.path.join(augmentor_image.output_directory, save_name)
                f.write(save_directory + '\t' + str(wave_vector[0][0]) + '\t' + str(wave_vector[0][1]) + '\t' + str(phi[i]) + '\n' )

        if save_to_disk:
            try:
                for i in range(len(images)):
                    if i == 0:
                        save_name = "SR_" \
                                    + os.path.basename(augmentor_image.image_path)
                        images[i].save(os.path.join(augmentor_image.output_directory, save_name))
                    elif i == 1:
                        save_name = "LR_" \
                                    + os.path.basename(augmentor_image.image_path)
                        images[i].save(os.path.join(augmentor_image.output_directory, save_name))
                    else:
                        save_name = "SIMdata(" \
                                    + str(i-1) \
                                    + ")_" \
                                    + os.path.basename(augmentor_image.image_path)

                        images[i].save(os.path.join(augmentor_image.output_directory, save_name))

            except IOError as e:
                print("Error writing %s, %s. Change save_format to PNG?" % ('LR', e.message))
                print("You can change the save format using the set_save_format(save_format) function.")
                print("By passing save_format=\"auto\", Augmentor can save in the correct format automatically.")

        # TODO: Fix this really strange behaviour.
        # As a workaround, we can pass the same back and basically
        # ignore the multi_threaded parameter completely for now.
        # if multi_threaded:
        #   return os.path.basename(augmentor_image.image_path)
        # else:
        #   return images[0]  # Here we return only the first image for the generators.

        # return images[0]  # old method.
        return images[0]