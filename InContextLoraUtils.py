import json
import os
import random
# import folder_paths
# import comfy.sd
# import comfy.utils
import torch
# from comfy import model_management
import safetensors
import numpy as np
from skimage import util as sk_util
from PIL import Image
import cv2

RESOLUTION_CONFIG = {
    1024: [
        (1536, 1024), # 2.4
    ]
}

def resize(img,resolution,interpolation=cv2.INTER_CUBIC):
    # print(img)
    
    # print(resolution)
    return cv2.resize(img,resolution, interpolation=cv2.INTER_CUBIC)

    
def create_image_from_color(width, height, color=(255, 255, 255)):
    # OpenCV uses BGR, so convert hex color to BGR if necessary
    if isinstance(color, str) and color.startswith('#'):
        color = tuple(int(color[i:i+2], 16) for i in (5, 3, 1))[::-1]
        
    # Create a blank image with the specified color
    blank_image = np.full((height, width, 3), color, dtype=np.uint8)
    return blank_image

def closest_mod_64(value):
    return value - (value % 64)

def fit_image(image,mask=None,output_length=1536,patch_mode="auto"):
    image = image.detach().cpu().numpy()
    if mask is not None:
        mask = mask.detach().cpu().numpy()
    
    # print("np.all(mask == 0)",np.all(mask == 0))
    base_length = int(output_length /3 *2)
    half_length = int(output_length /2)
    image_height, image_width, _ = image.shape
    # print("ori ", image_width, image_height)
    # if image.size is None:
    #     # convert tensor to pil image
    #     image = Image.fromarray(np.uint8(image)).convert('RGB')
    # image_width, image_height  = image.size
    target_width = int(half_length)
    target_height = int(base_length)
    # print("patch_mode",patch_mode)
    if patch_mode == "auto":
        if image_width > image_height:
            patch_mode = "patch_bottom"
            target_width = int(base_length)
            target_height = int(half_length)
        else:
            patch_mode = "patch_right"
    elif patch_mode == "patch_bottom":
        target_width = int(base_length)
        target_height = int(half_length)
    # print("patch_mode",patch_mode)
        
    if image_width < target_width or image_height < target_height:
        # print("image too small, resize to ", target_width, target_height)
        if image_height > image_width:
            new_width = int(image_width*(target_height/image_height))
            new_height = target_height
            # print(new_width,new_height)
            image = resize(image, (new_width,new_height))
            # print("mask",mask)
            # print("mask.shape",mask.shape)
            if mask is not None:
                mask = resize(mask, (new_width,new_height),cv2.INTER_NEAREST_EXACT)
        else:
            new_width = target_width
            new_height = int(image_height*(target_width/image_width))
            # print(new_width,new_height)
            image = resize(image, (new_width,new_height))
            
            # print("mask",mask)
            # print("mask.shape",mask.shape)
            if mask is not None:
                mask = resize(mask, (new_width,new_height),cv2.INTER_NEAREST_EXACT)
            
        image_height, image_width, _ = image.shape
        
        diff_x = target_width - image_width
        # print("diff_x",diff_x)
        diff_y = target_height - image_height
        # print("diff_y",diff_y)
        pad_x = abs(diff_x) // 2
        pad_y = abs(diff_y) // 2
        # add white pixels for padding
        if diff_x > 0 or diff_y > 0:
            resized_image = cv2.copyMakeBorder(
                image,
                pad_y, abs(diff_y) - pad_y,
                pad_x, abs(diff_x) - pad_x,
                cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
            if mask is not None:
                resized_mask = cv2.copyMakeBorder(
                    mask,
                    pad_y, abs(diff_y) - pad_y,
                    pad_x, abs(diff_x) - pad_x,
                    cv2.BORDER_CONSTANT, value=(0, 0, 0)
                )
        # crop extra pixels for square
        else:
            resized_image = image[pad_y:image_height-pad_y, pad_x:image_width-pad_x]
            resized_mask = mask[pad_y:image_height-pad_y, pad_x:image_width-pad_x]
        
    else:
        # get resized image size
        image_height, image_width, _ = image.shape
        # print("new size",image_height, image_width)
        # simple center crop
        scale_ratio = target_width / target_height
        image_ratio = image_width / image_height
        # referenced kohya ss code
        if image_ratio > scale_ratio: 
            up_scale = image_height / target_height
        else:
            up_scale = image_width / target_width
        expanded_closest_size = (int(target_width * up_scale + 0.5), int(target_height * up_scale + 0.5))
        diff_x = abs(expanded_closest_size[0] - image_width)
        diff_y = abs(expanded_closest_size[1] - image_height)
        
        crop_x =  diff_x // 2
        crop_y =  diff_y // 2
        cropped_image = image[crop_y:image_height-crop_y, crop_x:image_width-crop_x]
        resized_image = resize(cropped_image, (target_width,target_height))
        
        if mask is not None:
            # print("mask",mask)
            # print("mask.shape",mask.shape)
            cropped_mask = mask[crop_y:image_height-crop_y, crop_x:image_width-crop_x]
            resized_mask = resize(cropped_mask, (target_width,target_height),cv2.INTER_NEAREST_EXACT)

    if mask is None:
        resized_mask = torch.zeros((target_width,target_height))
    
    return resized_image, resized_mask, target_width, target_height, patch_mode

class AddMaskForICLora:
    # def __init__(self):
    #     self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
                    "required": { 
                        "first_image": ("IMAGE",),
                        "patch_mode": (["auto", "patch_right", "patch_bottom"], {
                            "default": "auto",
                        }),
                        "output_length": ("INT", {
                            "default": 1536,
                        }),
                        "patch_color": (["#FF0000", "#00FF00","#0000FF", "#FFFFFF"], {
                            "default": "#FF0000",
                        }),
                    },
                    "optional":{
                        "first_mask": ("MASK",),
                        "second_image": ("IMAGE",),
                        "second_mask": ("MASK",),
                    }
                }
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "MASK", "x_offset", "y_offset", "target_width", "target_height", "total_width", "total_height")
    FUNCTION = "add_mask"
    OUTPUT_NODE = True

    CATEGORY = "ICLoraUtils/AddMaskForICLora"
    def add_mask(self, first_image, patch_mode, output_length, patch_color, first_mask=None, second_image=None, second_mask=None):
        # patched_images = []
        # patched_masks = []
        if output_length % 64 != 0:
            output_length = output_length - (output_length % 64)
        
        image1 = first_image[0]
        if first_mask is None:
            image1_mask = torch.zeros((image1.shape[0], image1.shape[1]))
        else:
            image1_mask = first_mask[0]
            
        image1,image1_mask,target_width,target_height,patch_mode = fit_image(image1,image1_mask,output_length,patch_mode)
        if second_image is not None:
            image2 = second_image[0]
            if second_mask is None:
                image2_mask = torch.zeros((image2.shape[0], image2.shape[1]))
            else:
                image2_mask = second_mask[0]
            image2,image2_mask,_,_,_ = fit_image(image2,image2_mask,output_length,patch_mode)
            # print("image2_mask",image2_mask)
            # print("image2_mask.shape",image2_mask.shape)
        else:
            image2 = create_image_from_color(target_width,target_height, color=patch_color)
            image2 = torch.from_numpy(image2)
            if second_mask is None:
                image2_mask = torch.zeros((image2.shape[0], image2.shape[1]))
            else:
                image2_mask = second_mask[0]
            image2,image2_mask,_,_,_ = fit_image(image2,image2_mask,output_length)
            
        
        min_y = 0
        # max_y = 100
        min_x = 0
        # max_x = 100
        
        # print("image1.shape")
        # print(image1.shape)
        # print("image2.shape")
        # print(image2.shape)
        if second_mask is None or np.all(image2_mask == 0):
            # print("second_mask is None",second_mask is None)
            # print("np.all(image2_mask[0] == 0)",np.all(image2_mask == 0))
            image2_mask = torch.ones((image1.shape[0], image1.shape[1]))
            
        if patch_mode == "patch_right":
            concatenated_image = np.hstack((image1, image2))
            concatenated_mask = np.hstack((image1_mask, image2_mask))
            min_x = 50
        else:
            concatenated_image = np.vstack((image1, image2))
            concatenated_mask = np.vstack((image1_mask, image2_mask))
            min_y = 50
        # mask = torch.zeros((concatenated_image.shape[0], concatenated_image.shape[1]))
        min_y = int(min_y / 100.0 * concatenated_image.shape[0])
        # max_y = int(max_y / 100.0 * concatenated_image.shape[0])
        min_x = int(min_x / 100.0 * concatenated_image.shape[1])
        # max_x = int(max_x / 100.0 * concatenated_image.shape[1])
        # mask[min_y:max_y+1, min_x:max_x+1] = 1
        # print(mask.shape)
        # print(mask)
        
        return_masks = torch.from_numpy(concatenated_mask)[None,]
        
        concatenated_image = np.clip(255. * concatenated_image, 0, 255).astype(np.float32) / 255.0
        concatenated_image = torch.from_numpy(concatenated_image)[None,]
        
        # print('return_masks',return_masks.shape)
        return_images = concatenated_image
        # print('return_images',return_images.shape)
        return (return_images, return_masks, min_x, min_y, target_width, target_height, concatenated_image.shape[1], concatenated_image.shape[0])

NODE_CLASS_MAPPINGS = {
    "AddMaskForICLora": AddMaskForICLora,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AddMaskForICLora": "Add Mask For IC Lora",
}