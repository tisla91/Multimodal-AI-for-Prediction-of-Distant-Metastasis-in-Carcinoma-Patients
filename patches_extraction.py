import shutil                            
import glob                              
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import openslide
import tiatoolbox
from tiatoolbox.wsicore.wsireader import TIFFWSIReader
from tiatoolbox.tools.patchextraction import get_patch_extractor, SlidingWindowPatchExtractor
from tiatoolbox.tools import stainnorm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model, load_model
import tifffile as tiff
import skimage
import pathml
from PIL import Image



os.getcwd()
root_path = "#######################"
root_dirs = next(os.walk(root_path))[1]


# root = root directory.
def WSI_path_list(root):
    
    """Get into each histopathology image folder, and add all their paths into a single list"""
    
    up_root = next(os.walk(root))[1] 
    try:
        up_root.remove(".ipynb_checkpoints") 
    except:
        pass    
    WSIs_path = []
    for barcode in up_root:       
        if barcode not in done:         
            svs_path = root + "/" + barcode + "/" + barcode + "_histopath"
            extension = "/*.svs"
            svs_file_list = glob.glob(svs_path + extension)                  
            try:
                svs_file = svs_file_list[0]
            except:
                print(barcode,"svs not added")
                pass           
            WSIs_path.append(svs_file)
            print(svs_file, "added to list")  
    return WSIs_path



nuclei_model = load_model("model_for_histo.h5")    ### Pretrained breast cancer segmentation model

# img = An image of size 512 * 512 (4D)
def nuclei_eosin_content(img):
    """
    Takes in an img and use the model to segment nuclei in img.
    Returns nuclei ratio in img.
    """
    exp_dim = np.expand_dims(img, axis = 0)
    pred_img = nuclei_model.predict(exp_dim, verbose= 1)
    binary_pred_img = (pred_img[0] > 0.1).astype(np.uint8)    
    nuclei = np.sum(binary_pred_img)
    
    return nuclei / float(binary_pred_img.size)



# input_img = a single image stored in a variable.
# thresh_min = 40
def filtering(input_img,thresh_min):
    
    """Calculate image tissue percentage."""
    
    rgb = np.dot(input_img[...,:3], [0.299, 0.587, 0.114])     
    rgb[:] = [255 - x for x in rgb]                           
    binary_min = rgb > thresh_min                            
    return np.sum(binary_min)/float(binary_min.size)           


# root = root directory.
# target = image or image path to be used as standard for normalization (Vahadane stain normalization).
# tissue_threshold = Numeric representation (0 to 1) of tissue area in patch. - 0.8 suggested

def extract_patches(root, target, tissue_threshold):
    
    """WSI preprocessing: Read-in, patches extraction, patches filtering, 
    images normalization, and saving into local directory"""
    
    track_done = []
    WSIs_path = WSI_path_list(root)
    print(len(WSIs_path))
    for slide in WSIs_path:
        # Folder to store generated patches
        slide_basename = os.path.basename(slide)
        basename_prefix = slide_basename[:12]
        patches_dir = slide[: -(len(slide_basename))]
        patches_dir = patches_dir + basename_prefix + "_tiles"
        os.mkdir(patches_dir)        
        # Slide object 
        slide_object = TIFFWSIReader(slide, power = 20)
        # Patches extraction
        patch_extractor = get_patch_extractor(
            input_img= slide_object, 
            method_name= "slidingwindow", 
            patch_size= (512, 512), 
            stride=(512, 512))
        
        # Randomly select patches of size 512 X 512 from patch_extractor object
        if len(patch_extractor.coordinate_list) >= 500:
            random_sel = np.random.choice(len((patch_extractor.coordinate_list) - 1), size = 500, replace= False)
        elif len(patch_extractor.coordinate_list) >= 400:
            random_sel = np.random.choice(len((patch_extractor.coordinate_list) - 1), size = 400, replace= False)
        elif len(patch_extractor.coordinate_list) >= 300:
            random_sel = np.random.choice(len((patch_extractor.coordinate_list) - 1), size = 300, replace= False)
        elif len(patch_extractor.coordinate_list) >= 200:
            random_sel = np.random.choice(len((patch_extractor.coordinate_list) - 1), size = 200, replace= False)
        else:
            continue
                        
        # Select first 200 that pass filtering (>80% tissue area)
        sel200_lst = []
        for index in list(random_sel):
            index = int(index)
            if len(sel200_lst) >= 200:
                break
            elif filtering(patch_extractor[int(index)], 40) > 0.8:
                sel200_lst.append(patch_extractor[index])
                
        # Select top 60 images with highest nuclei scores content
        nuclei_scores_dict = {}
        for index, patch in enumerate(sel200_lst):
            patch_nuclei_score = nuclei_eosin_content(patch)
            nuclei_scores_dict[index] = patch_nuclei_score
        sorted_nuclei_scores_dict = dict(sorted(nuclei_scores_dict.items(), key=lambda x:x[1]))
        patch_nuclei_index = list(sorted_nuclei_scores_dict.keys())  
        reversed_patch_nuclei_index = patch_nuclei_index[::-1] 
        
        save_count = 0
        for index in reversed_patch_nuclei_index:            
            check_blank_score = nuclei_eosin_content(sel200_lst[index])           
            # Filter out blank images
            if check_blank_score >= 0.60:
                pass           
            elif save_count >= 60:
                break            
            else:
                try:
                    arr_img = np.asarray(sel200_lst[index])
                    # Save image in appropraite directory
                    img_name = patches_dir + "/" + basename_prefix + "_" + str(index) + ".png"
                    skimage.io.imsave(img_name, arr_img)                    
                    save_count += 1
                except:
                    pass        
        track_done.append(basename_prefix)
        print("Preprocessing completed for", basename_prefix)       
    return "All slides preprocessing completed."
        


# BLCA Preprocessing
blca_root = root_path + "/BLCA_histo"
target_img = skimage.io.imread(r"norm_standard.png")
target_img = np.asarray(target_img)
tissue_area_threshold = 0.8
extract_patches(blca_root, target_img, tissue_area_threshold)