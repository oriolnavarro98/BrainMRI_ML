"""
Define class MRIDataset from Dataset module, to be used with mri_brain_segmentation.py module to feed MRI images and
corresponding mask to the actual model.
"""

from utils import *
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as tvt

class MRIDataset(Dataset):
    """
    The class intialized with a list of image paths, and a dicitonary that correlates
    MRI paths with its corresponding mask. These two variables are obatained using the function parse_input_and_mask() from
    utils.py. It also takes as an argument a transformation pipeline, which corresponds to the transformations that
    need to be applied to the images before they are fed into the model. (As a default we are using ToTensor() to transform
    PIL image to torch tensor format).
    
    INPUT (init)
        --> input_image_list (list with list with all the MRI input files)
        --> image_mask_dic (dictionary with MRI scans as keys, and tumor masks as values)
    
    OUTPUT (getitem)
        --> mri (tensor with 3 channels that represents the MRI image)
        --> mask (tensor with 1 channel representing the corresponding mask)
    
    OUTPUT (len)
        --> length of the dataset
    """

    def __init__(self, input_image_list, image_mask_dic, transform_pipe):
        self.input_image_list_paths = input_image_list
        self.image_mask_dic_paths = image_mask_dic
        self.transform_pipe = transform_pipe
        
    def __len__(self):
        return(len(self.input_image_list_paths))

    def __getitem__(self, id):
        # Extract image corresponding to id location on image list, and corresponding mask
        input_mri_path = self.input_image_list_paths[id]
        mask_mri_path = self.image_mask_dic_paths[input_mri_path]

        mri = Image.open(input_mri_path)
        # mri.show()
        mask = Image.open(mask_mri_path)
        # mask.show()

        # Evaluate image format
        assert mri.mode == "RGB", "MRI should be an RGB image"
        assert mask.mode == "L", "Mask should be an L image (grayscale)"

        # Apply transformation
        if self.transform_pipe is not None:
            mri = self.transform_pipe(mri)
            mask = self.transform_pipe(mask)

        return mri, mask


if __name__ == "__main__":
    # print("Testing...")
    # input_image_list = ["/Users/oriolnavarro/Desktop/PersonalDev/BrainMRI_MLproject/subset_kaggle_3m/TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_12.tif"]
    # image_mask_dict = {
    #     "/Users/oriolnavarro/Desktop/PersonalDev/BrainMRI_MLproject/subset_kaggle_3m/TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_12.tif" : "/Users/oriolnavarro/Desktop/PersonalDev/BrainMRI_MLproject/subset_kaggle_3m/TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_12_mask.tif"
    # }
    # transform_pipe = tvt.Compose([tvt.ToTensor()])

    # MyObject = MRIDataset(input_image_list, image_mask_dict, transform_pipe)

    # print(MyObject.__len__())
    # mri, mask = MyObject.__getitem__(0)

    # print(mri.size())
    # print(mask.size())
    pass



