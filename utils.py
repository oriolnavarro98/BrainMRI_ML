import os
import pandas as pd
import numpy as np


def parse_input_and_mask(data_path, data_code='TCGA'):
    '''
    Take a folder containing Brain MRI scans and tumor masks. The folder needs to be structured in the following way:
    - Main Data Folder
        -- Sub folder for patient 1 (Starting with TCGA_...)
            -- Image 1  (TCGA_<institution-code>_<patient-id>_<slice-number>.tif)
            -- Mask 1   (TCGA_<institution-code>_<patient-id>_<slice-number>_mask.tif)
            -- Image 2  (TCGA_<institution-code>_<patient-id>_<slice-number>.tif)
            -- Mask 2   (TCGA_<institution-code>_<patient-id>_<slice-number>_mask.tif)
            ...
        -- Sub folder for patient 2 (Starting with TCGA_...)
        ...

    The function will return a dictionary with each image path as key, and each corresponding mask path as the value. It
    will also output a list of the input files. 
    It will ignore any file that does not start with the code passed as input. (Eg. TCGA)

    INPUT:
        - data_path --> Path to Root data folder
        - data_code --> String identifier to each file name (default: TCGA)

    OUTPUT:
        - output_dict --> dictionary with MRI scans as keys, and tumor masks as values)
        - input_files_list --> list with all the MRI input files)

    '''

    # Intialize output variables
    output_dict = dict()
    input_files_list = list()

    # Iterate over the subdirectories of the data root folder
    for root, subdirectories, files in os.walk(data_path):
        # Iterate over each file
        for file in files:
            # Check if the file corresponds to an image slice
            if data_code in file:
                image_components = file.split('_')
                # Check if the image file is an input or a mask
                if len(image_components) == 5:

                    # Obatin corresponding mask to input image
                    mask_file = file.split('.')[0] + '_mask.tif'
                    input_image_path = os.path.join(root, file)
                    mask_image_path = os.path.join(root, mask_file)
                    # Load input and mask to output variables
                    input_files_list.append(input_image_path)
                    output_dict[input_image_path] = mask_image_path
    

    return output_dict, input_files_list
