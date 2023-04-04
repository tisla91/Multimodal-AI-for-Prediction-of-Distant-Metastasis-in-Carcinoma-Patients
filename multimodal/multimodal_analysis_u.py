#!/usr/bin/env python
# coding: utf-8

# In[1]:


import shutil                            
import glob                         
import numpy as np
import skimage
import cv2
import matplotlib.pyplot as plt
import os
import openslide
import pandas as pd
import pathml
from pathml.core import HESlide
from pathml.preprocessing import StainNormalizationHE
import pickle
import statistics
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.applications import DenseNet121, InceptionV3, ResNet50 
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling1D, TimeDistributed, GlobalAveragePooling1D, Flatten, concatenate, Dense, Dropout
from sklearn.model_selection import train_test_split

from multimodal_test import *



os.getcwd()
root_path = "#########################"
root_dirs = next(os.walk(root_path))[1]   


# Folder path to each cancer type
# HNSC MET
hnsc_root = root_path + "/HNSC_histo"
hnsc_root_list = os.listdir(hnsc_root)
try:
    hnsc_root_list.remove(".ipynb_checkpoints")   
except:
    pass

# HNSC NO_MET
hnsc_no_met_root = root_path + "/HNSC_histo_no_met"
hnsc_no_met_root_list = os.listdir(hnsc_no_met_root)
try:
    hnsc_no_met_root_list.remove(".ipynb_checkpoints")     
except:
    pass


# PAAD MET
paad_root = root_path + "/PAAD_histo"
paad_root_list = os.listdir(paad_root)
try:
    paad_root_list.remove(".ipynb_checkpoints")     
except:
    pass

# PAAD NO-MET
paad_no_met_root = root_path + "/PAAD_histo_no_met"
paad_no_met_root_list = os.listdir(paad_no_met_root)
try:
    paad_no_met_root_list.remove(".ipynb_checkpoints")     
except:
    pass


# BLCA MET
blca_root = root_path + "/BLCA_histo"
blca_root_list = os.listdir(blca_root)
try:
    blca_root_list.remove(".ipynb_checkpoints")     
except:
    pass

# BLCA NO_MET
blca_no_met_root = root_path + "/BLCA_histo_no_met"
blca_no_met_root_list = os.listdir(blca_no_met_root)
try:
    blca_no_met_root_list.remove(".ipynb_checkpoints")     
except:
    pass



print(len(hnsc_root_list))
print(len(hnsc_no_met_root_list))
print(len(paad_root_list))
print(len(paad_no_met_root_list))
print(len(blca_root_list))
print(len(blca_no_met_root_list))



# Read in Genomic data
gen = pd.read_csv("full_gen_preprocessed_tpm.csv")    # Contains preprocessed gene record of all patients

# Selected HNSC genes
gen_hnsc = pd.read_csv("top_genes_HNC_tpm.csv")
genes_hnsc = list(gen_hnsc["gene_name"])

# Selected PAAD genes
gen_paad = pd.read_csv("top_genes_PAAD_tpm.csv")
genes_paad = list(gen_paad["gene_name"])

# Selected PAAD genes
gen_blca = pd.read_csv("top_genes_BLCA_tpm.csv")
genes_blca = list(gen_blca["gene_name"])

genes_lst = genes_hnsc + genes_paad + genes_blca
genes_HPB = ["barcode", "cancer_type", "met_status"] + genes_lst   # Include the "barcode", "cancer_type", and "met_status" columns

# Select only HPB genes 
sel_genes = gen.loc[ : , genes_HPB]


# Read in img barcodes, and paths df
img_df = pd.read_csv("barc_path_df.csv")           # Contains barcodes, and path to patient's images



# Read in clinical data
clin_df_met = pd.read_csv("clinical3_cancers_df_tn.csv")    # Contains preprocessed clinical data
clin_df_met = clin_df_met[["barcode",
                   "age",
                   "number_of_lymphnodes_positive_by_he",
                   "t_stage",
                   "n_stage"]]

clin_df_nmet = pd.read_csv("all3_clin_df.csv")
clin_df_nmet = clin_df_nmet[["barcode",
                   "age",
                   "number_of_lymphnodes_positive_by_he",
                   "t_stage",
                   "n_stage"]]

clin_df = pd.concat([clin_df_met, clin_df_nmet])

# Normalize age, and number_of_lymphnodes_positive_by_he columns
clin_df["age"] = (clin_df["age"] - clin_df["age"].min()) / (clin_df["age"].max() - clin_df["age"].min())
clin_df["number_of_lymphnodes_positive_by_he"] = (clin_df["number_of_lymphnodes_positive_by_he"] - clin_df["number_of_lymphnodes_positive_by_he"].min()) / (clin_df["number_of_lymphnodes_positive_by_he"].max() - clin_df["number_of_lymphnodes_positive_by_he"].min())


# Merge Img, gen, and clin dfs
img_gen_df = img_df.merge(sel_genes, how="inner", on="barcode")
print("Number of entries before checking for duplicates: ", img_gen_df.shape)
img_gen_clin_df = img_gen_df.merge(clin_df, how="inner", on="barcode")
print("Number of entries before checking for duplicates: ", img_gen_clin_df.shape)
img_gen_clin_df = img_gen_clin_df.drop_duplicates(subset= "barcode", keep= False)
print("Number of entries after checking for duplicates: ", img_gen_clin_df.shape)

img_gen_df = img_gen_clin_df


print("Number of NAs in data: ", img_gen_df.isnull().sum().sum())
#Check number of metastasis, and no-metastasis samples for each cancer type
print("Number of HNSC samples: ", img_gen_df[img_gen_df["cancer_type"] == "HNSC"]["met_status"].value_counts())    # HNSC
print("Number of PAAD samples: ", img_gen_df[img_gen_df["cancer_type"] == "PAAD"]["met_status"].value_counts())    # PAAD
print("Number of BLCA samples: ", img_gen_df[img_gen_df["cancer_type"] == "BLCA"]["met_status"].value_counts())    # BLCA




# Create initial list for MLP scores dataframe


# Create initial list for MLP scores dataframe
MLP_clin_labels = ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]
MLP_clin_accuracy = []
MLP_clin_precision = []
MLP_clin_recall = []
MLP_clin_f1score =[]
MLP_clin_AUC = []

MLP_img_labels = ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]
MLP_img_accuracy = []
MLP_img_precision = []
MLP_img_recall = []
MLP_img_f1score =[]
MLP_img_AUC = []

MLP_gen_labels = ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]
MLP_gen_accuracy = []
MLP_gen_precision = []
MLP_gen_recall = []
MLP_gen_f1score =[]
MLP_gen_AUC = []

MLP_img_gen_labels = ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]
MLP_img_gen_accuracy = []
MLP_img_gen_precision = []
MLP_img_gen_recall = []
MLP_img_gen_f1score =[]
MLP_img_gen_AUC = []

MLP_clin_gen_labels = ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]
MLP_clin_gen_accuracy = []
MLP_clin_gen_precision = []
MLP_clin_gen_recall = []
MLP_clin_gen_f1score =[]
MLP_clin_gen_AUC = []

MLP_img_clin_gen_labels = ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]
MLP_img_clin_gen_accuracy = []
MLP_img_clin_gen_precision = []
MLP_img_clin_gen_recall = []
MLP_img_clin_gen_f1score =[]
MLP_img_clin_gen_AUC = []



# Create initial list for SVM scores dataframe
SVM_clin_labels = ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]
SVM_clin_accuracy = []
SVM_clin_f1score =[]
SVM_clin_AUC = []

SVM_img_labels = ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]
SVM_img_accuracy = []
SVM_img_f1score =[]
SVM_img_AUC = []

SVM_gen_labels = ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]
SVM_gen_accuracy = []
SVM_gen_f1score =[]
SVM_gen_AUC = []

SVM_img_gen_labels = ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]
SVM_img_gen_accuracy = []
SVM_img_gen_f1score =[]
SVM_img_gen_AUC = []

SVM_clin_gen_labels = ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]
SVM_clin_gen_accuracy = []
SVM_clin_gen_f1score =[]
SVM_clin_gen_AUC = []

SVM_img_clin_gen_labels = ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]
SVM_img_clin_gen_accuracy = []
SVM_img_clin_gen_f1score =[]
SVM_img_clin_gen_AUC = []



########## Cross-validation 
num_cv = 5

for fold_val in range(0, num_cv):
    
    print("Fold " + str(fold_val) +  " workflow started.")
    
    # Split data into train. and test set... monte carlo cv
    
    # Seperate independent variables
    X = img_gen_df.drop(labels = ["met_status"], axis = 1)
    print("X shape: ", X.shape)

    # Extract dependent variable (met_status)
    Y = img_gen_df["met_status"].astype("int")
    print("Y shape: ", Y.shape)

    # Train test split... By met status / Cancer type
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=fold_val, stratify= X["cancer_type"])   
    print(f"X_train samples number: {len(X_train)}, Y_train samples number: {len(Y_train)}")
    print(f"X_test samples number: {len(X_test)}, Y_test samples number: {len(Y_test)}")
    
    print("Data splitted for fold " + str(fold_val) +  ". To begin 'norm_scale_load'.")
    
    # Normalize, scale, and load train images
    train_imgs_n_paths = norm_scale_load(X_train["tiles_path"])
    all_train_imgs = train_imgs_n_paths[0]
    tiles_path_train = train_imgs_n_paths[1]
    
    train_arr = np.concatenate(all_train_imgs, axis=0)
    
    # Give label to tiles from patients label
    new_Y_train = label_tiles(all_train_imgs, Y_train)
    
    # Get path to test set tiles
    # For only getting test images paths
    tiles_path_test = []
    for barc_path in list(X_test["tiles_path"]):
        tiles_path_test.append(barc_path)
    print(len(tiles_path_test))
       
    print("Norm_scale_load completed for fold " + str(fold_val) +  ". To begin model training.")
    
    # Train and save model
    model_save_name = "DenseNet121_cv_fold_" + str(fold_val) + ".h5"
    model_weight_save_name = "DenseNet121_weight_cv_fold_" + str(fold_val) + ".h5"
    chkpnt_save_name = "DenseNet121_best_only_cv_fold_" + str(fold_val) + ".h5"
    
    create_model_densenet(train_arr, new_Y_train, model_save_name, model_weight_save_name, chkpnt_save_name)   
    
    print("Model training completed for fold " + str(fold_val) +  ". To begin features extraction.")
    
    # Define paths to extract image features for fold
    train_feat = root_path + "/model_features/ALL3_32_cv" + str(fold_val) + "_conf/all3_32_512_train_features_densenet121.npy"
    train_label = root_path + "/model_features/ALL3_32_cv" + str(fold_val) + "_conf/all3_32_512_train_label_densenet121.npy"
    train_path = root_path + "/model_features/ALL3_32_cv" + str(fold_val) + "_conf/all3_32_512_train_paths_densenet121.npy"
    test_feat = root_path + "/model_features/ALL3_32_cv" + str(fold_val) + "_conf/all3_32_512_test_features_densenet121.npy"
    test_label = root_path + "/model_features/ALL3_32_cv" + str(fold_val) + "_conf/all3_32_512_test_label_densenet121.npy"
    test_path = root_path + "/model_features/ALL3_32_cv" + str(fold_val) + "_conf/all3_32_512_test_paths_densenet121.npy"
    output = [train_feat, train_label, train_path, test_feat, test_label, test_path]
    # Extract Features
    features = extract_feature(img_gen_df, tiles_path_train, tiles_path_test, output, "DenseNet121", model_save_name)
        
    print("Features extraction completed for fold " + str(fold_val) +  ". To begin analysis.")   
    
    # Prep data for prediction    
    pred_data_lst = pred_data_prep(img_gen_df, train_feat, train_label, train_path, test_feat, test_label, test_path)  # ... List contains 14 variables
    
    
    load_train_feat = pred_data_lst[0]
    load_test_feat = pred_data_lst[1]
    train_img_gen = pred_data_lst[2]
    test_img_gen = pred_data_lst[3]
    train_gen = pred_data_lst[4]
    test_gen = pred_data_lst[5]
    train_clin = pred_data_lst[6]
    test_clin = pred_data_lst[7]
    train_img_gen_clin = pred_data_lst[8]
    test_img_gen_clin = pred_data_lst[9]
    train_clin_gen = pred_data_lst[10]
    test_clin_gen = pred_data_lst[11]
    train_lab = pred_data_lst[12]
    test_lab = pred_data_lst[13]
    
    
    # Get MLP prediction scores
    
    # Only Clinical
    mlp_clin = MLP(train_clin, train_lab, test_clin, test_lab, 4, model_layers = 1)
    mlp_clin[2]       # Loss, Acc, Pre, Rec, AUC
    mlp_clin[3]      # f1_score
    
    MLP_clin_accuracy.append(mlp_clin[2][1])
    MLP_clin_precision.append(mlp_clin[2][2])
    MLP_clin_recall.append(mlp_clin[2][3])
    MLP_clin_AUC.append(mlp_clin[2][4])
    MLP_clin_f1score.append(mlp_clin[3])
    
    # Only Image
    mlp_img = MLP(load_train_feat, train_lab, load_test_feat, test_lab, 32, model_layers = 1)
    mlp_img[2]       # Loss, Acc, Pre, Rec, AUC
    mlp_img[3]      # f1_score
    
    MLP_img_accuracy.append(mlp_img[2][1])
    MLP_img_precision.append(mlp_img[2][2])
    MLP_img_recall.append(mlp_img[2][3])
    MLP_img_AUC.append(mlp_img[2][4])
    MLP_img_f1score.append(mlp_img[3])

    # Only Genomic
    mlp_gen = MLP(train_gen, train_lab, test_gen, test_lab, 150, model_layers = 1)
    mlp_gen[2]       # Loss, Acc, Pre, Rec, AUC
    mlp_gen[3]      # f1_score

    MLP_gen_accuracy.append(mlp_gen[2][1])
    MLP_gen_precision.append(mlp_gen[2][2])
    MLP_gen_recall.append(mlp_gen[2][3])
    MLP_gen_AUC.append(mlp_gen[2][4])
    MLP_gen_f1score.append(mlp_gen[3])
    
    # Image + Genomic
    mlp_img_gen = MLP(train_img_gen, train_lab, test_img_gen, test_lab, 182, model_layers = 1)
    mlp_img_gen[2]
    mlp_img_gen[3]

    MLP_img_gen_accuracy.append(mlp_img_gen[2][1])
    MLP_img_gen_precision.append(mlp_img_gen[2][2])
    MLP_img_gen_recall.append(mlp_img_gen[2][3])
    MLP_img_gen_AUC.append(mlp_img_gen[2][4])
    MLP_img_gen_f1score.append(mlp_img_gen[3])
    
    # Clinical + Genomic
    mlp_clin_gen = MLP(train_clin_gen, train_lab, test_clin_gen, test_lab, 154, model_layers = 1)
    mlp_clin_gen[2]
    mlp_clin_gen[3]

    MLP_clin_gen_accuracy.append(mlp_clin_gen[2][1])
    MLP_clin_gen_precision.append(mlp_clin_gen[2][2])
    MLP_clin_gen_recall.append(mlp_clin_gen[2][3])
    MLP_clin_gen_AUC.append(mlp_clin_gen[2][4])
    MLP_clin_gen_f1score.append(mlp_clin_gen[3])
    
    # Image + Clinical + Genomic
    mlp_img_clin_gen = MLP(train_img_gen_clin, train_lab, test_img_gen_clin, test_lab, 186, model_layers = 1)
    mlp_img_clin_gen[2]
    mlp_img_clin_gen[3]

    MLP_img_clin_gen_accuracy.append(mlp_img_clin_gen[2][1])
    MLP_img_clin_gen_precision.append(mlp_img_clin_gen[2][2])
    MLP_img_clin_gen_recall.append(mlp_img_clin_gen[2][3])
    MLP_img_clin_gen_AUC.append(mlp_img_clin_gen[2][4])
    MLP_img_clin_gen_f1score.append(mlp_img_clin_gen[3])
    
    
    
    # Get SVM scores
    # Only Clinical
    svm_clin = SVM_pred(train_clin, train_lab, test_clin, test_lab)
    svm_clin_acc = svm_clin[0]
    svm_clin_f1 = svm_clin[1]
    svm_clin_auc = svm_clin[2]
    
    SVM_clin_accuracy.append(svm_clin_acc)
    SVM_clin_f1score.append(svm_clin_f1)
    SVM_clin_AUC.append(svm_clin_auc)
        
    # Only image
    svm_img = SVM_pred(load_train_feat, train_lab, load_test_feat, test_lab)
    svm_img_acc = svm_img[0]
    svm_img_f1 = svm_img[1]
    svm_img_auc = svm_img[2]
    
    SVM_img_accuracy.append(svm_img_acc)
    SVM_img_f1score.append(svm_img_f1)
    SVM_img_AUC.append(svm_img_auc)
    
    # Only Genomic
    svm_gen = SVM_pred(train_gen, train_lab, test_gen, test_lab)
    svm_gen_acc = svm_gen[0]
    svm_gen_f1 = svm_gen[1]
    svm_gen_auc = svm_gen[2]
    
    SVM_gen_accuracy.append(svm_gen_acc)
    SVM_gen_f1score.append(svm_gen_f1)
    SVM_gen_AUC.append(svm_gen_auc)
    
    # Image + Genomic
    svm_img_gen = SVM_pred(train_img_gen, train_lab, test_img_gen, test_lab)
    svm_img_gen_acc = svm_img_gen[0]
    svm_img_gen_f1 = svm_img_gen[1]
    svm_img_gen_auc = svm_img_gen[2]
    
    SVM_img_gen_accuracy.append(svm_img_gen_acc)
    SVM_img_gen_f1score.append(svm_img_gen_f1)
    SVM_img_gen_AUC.append(svm_img_gen_auc)
    
    # Clinical + Genomic
    svm_clin_gen = SVM_pred(train_clin_gen, train_lab, test_clin_gen, test_lab)
    svm_clin_gen_acc = svm_clin_gen[0]
    svm_clin_gen_f1 = svm_clin_gen[1]
    svm_clin_gen_auc = svm_clin_gen[2]
    
    SVM_clin_gen_accuracy.append(svm_clin_gen_acc)
    SVM_clin_gen_f1score.append(svm_clin_gen_f1)
    SVM_clin_gen_AUC.append(svm_clin_gen_auc)
    
    # Image + Clinical + Genomic
    svm_img_clin_gen = SVM_pred(train_img_gen_clin, train_lab, test_img_gen_clin, test_lab)
    svm_img_clin_gen_acc = svm_img_clin_gen[0]
    svm_img_clin_gen_f1 = svm_img_clin_gen[1]
    svm_img_clin_gen_auc = svm_img_clin_gen[2]
    
    SVM_img_clin_gen_accuracy.append(svm_img_clin_gen_acc)
    SVM_img_clin_gen_f1score.append(svm_img_clin_gen_f1)
    SVM_img_clin_gen_AUC.append(svm_img_clin_gen_auc)
    
    



# Put scores populated in lists into dataframe, and save

MLP_clin_scores = {"clin_folds": MLP_clin_labels,
                  "clin_fold_accuracy": MLP_clin_accuracy,
                  "clin_fold_precision": MLP_clin_precision,
                  "clin_fold_recall": MLP_clin_recall,
                  "clin_fold_f1score": MLP_clin_f1score,
                  "clin_fold_AUC": MLP_clin_AUC
                 }
MLP_clin_df = pd.DataFrame.from_dict(MLP_clin_scores)
print("MLP_clin_df:")
print(MLP_clin_df)
MLP_clin_df.to_csv(root_path + "/multimodal_cv_results/CONF/MLP_clin_fold_scores.csv")


MLP_img_scores = {"img_folds": MLP_img_labels,
                  "img_fold_accuracy": MLP_img_accuracy,
                  "img_fold_precision": MLP_img_precision,
                  "img_fold_recall": MLP_img_recall,
                  "img_fold_f1score": MLP_img_f1score,
                  "img_fold_AUC": MLP_img_AUC
                 }
MLP_img_df = pd.DataFrame.from_dict(MLP_img_scores)
print("MLP_img_df:")
print(MLP_img_df)
MLP_img_df.to_csv(root_path + "/multimodal_cv_results/CONF/MLP_img_fold_scores.csv")


MLP_gen_scores = {"gen_folds": MLP_gen_labels,
                  "gen_fold_accuracy": MLP_gen_accuracy,
                  "gen_fold_precision": MLP_gen_precision,
                  "gen_fold_recall": MLP_gen_recall,
                  "gen_fold_f1score": MLP_gen_f1score,
                  "gen_fold_AUC": MLP_gen_AUC
                 }
MLP_gen_df = pd.DataFrame.from_dict(MLP_gen_scores)
print("MLP_gen_df:")
print(MLP_gen_df)
MLP_gen_df.to_csv(root_path + "/multimodal_cv_results/CONF/MLP_gen_fold_scores.csv")


MLP_img_gen_scores = {"img_gen_folds": MLP_img_gen_labels,
                  "img_gen_fold_accuracy": MLP_img_gen_accuracy,
                  "img_gen_fold_precision": MLP_img_gen_precision,
                  "img_gen_fold_recall": MLP_img_gen_recall,
                  "img_gen_fold_f1score": MLP_img_gen_f1score,
                  "img_gen_fold_AUC": MLP_img_gen_AUC
                 }
MLP_img_gen_df = pd.DataFrame.from_dict(MLP_img_gen_scores)
print("MLP_img_gen_df:")
print(MLP_img_gen_df)
MLP_img_gen_df.to_csv(root_path + "/multimodal_cv_results/CONF/MLP_img_gen_fold_scores.csv")


MLP_clin_gen_scores = {"clin_gen_folds": MLP_clin_gen_labels,
                  "clin_gen_fold_accuracy": MLP_clin_gen_accuracy,
                  "clin_gen_fold_precision": MLP_clin_gen_precision,
                  "clin_gen_fold_recall": MLP_clin_gen_recall,
                  "clin_gen_fold_f1score": MLP_clin_gen_f1score,
                  "clin_gen_fold_AUC": MLP_clin_gen_AUC
                 }
MLP_clin_gen_df = pd.DataFrame.from_dict(MLP_clin_gen_scores)
print("MLP_clin_gen_df:")
print(MLP_clin_gen_df)
MLP_clin_gen_df.to_csv(root_path + "/multimodal_cv_results/CONF/MLP_clin_gen_fold_scores.csv")


MLP_img_clin_gen_scores = {"img_clin_gen_folds": MLP_img_clin_gen_labels,
                  "img_clin_gen_fold_accuracy": MLP_img_clin_gen_accuracy,
                  "img_clin_gen_fold_precision": MLP_img_clin_gen_precision,
                  "img_clin_gen_fold_recall": MLP_img_clin_gen_recall,
                  "img_clin_gen_fold_f1score": MLP_img_clin_gen_f1score,
                  "img_clin_gen_fold_AUC": MLP_img_clin_gen_AUC
                 }
MLP_img_clin_gen_df = pd.DataFrame.from_dict(MLP_img_clin_gen_scores)
print("MLP_img_clin_gen_df:")
print(MLP_img_clin_gen_df)
MLP_img_clin_gen_df.to_csv(root_path + "/multimodal_cv_results/CONF/MLP_img_clin_gen_fold_scores.csv")




# Get MLP CV scores
MLP_cv_score = {"data_type" : ["clinical", "image", "genomic", "image_genomic", "clinical_genomic", "image_clinical_genomic"],
                "cv_accuracy_mean" : [statistics.mean(MLP_clin_accuracy), statistics.mean(MLP_img_accuracy), statistics.mean(MLP_gen_accuracy), statistics.mean(MLP_img_gen_accuracy), statistics.mean(MLP_clin_gen_accuracy), statistics.mean(MLP_img_clin_gen_accuracy)],
                
                "cv_accuracy_std" : [statistics.stdev(MLP_clin_accuracy), statistics.stdev(MLP_img_accuracy), statistics.stdev(MLP_gen_accuracy), statistics.stdev(MLP_img_gen_accuracy), statistics.stdev(MLP_clin_gen_accuracy), statistics.stdev(MLP_img_clin_gen_accuracy)],
                
                "cv_f1score_mean" : [statistics.mean(MLP_clin_f1score), statistics.mean(MLP_img_f1score), statistics.mean(MLP_gen_f1score), statistics.mean(MLP_img_gen_f1score), statistics.mean(MLP_clin_gen_f1score), statistics.mean(MLP_img_clin_gen_f1score)],
                
                "cv_f1score_std" : [statistics.stdev(MLP_clin_f1score), statistics.stdev(MLP_img_f1score), statistics.stdev(MLP_gen_f1score), statistics.stdev(MLP_img_gen_f1score), statistics.stdev(MLP_clin_gen_f1score), statistics.stdev(MLP_img_clin_gen_f1score)],
                
                "cv_AUC_mean" : [statistics.mean(MLP_clin_AUC), statistics.mean(MLP_img_AUC), statistics.mean(MLP_gen_AUC), statistics.mean(MLP_img_gen_AUC), statistics.mean(MLP_clin_gen_AUC), statistics.mean(MLP_img_clin_gen_AUC)],
                
                "cv_AUC_std" : [statistics.stdev(MLP_clin_AUC), statistics.stdev(MLP_img_AUC), statistics.stdev(MLP_gen_AUC), statistics.stdev(MLP_img_gen_AUC), statistics.stdev(MLP_clin_gen_AUC), statistics.stdev(MLP_img_clin_gen_AUC)]
               }

MLP_cv_score_df = pd.DataFrame.from_dict(MLP_cv_score)
MLP_cv_score_df.to_csv(root_path + "/multimodal_cv_results/CONF/MLP_cv_scores.csv")
print("MLP cross validation scores saved")



# Created initial list for SVM scores dataframe

SVM_clin_scores = {"clin_folds": SVM_clin_labels,
                  "clin_fold_accuracy": SVM_clin_accuracy,
                  "clin_fold_f1score": SVM_clin_f1score,
                  "clin_fold_AUC": SVM_clin_AUC
                 }
SVM_clin_df = pd.DataFrame.from_dict(SVM_clin_scores)
print("SVM_clin_df:")
print(SVM_clin_df)
SVM_clin_df.to_csv(root_path + "/multimodal_cv_results/CONF/SVM_img_fold_scores.csv")




SVM_img_scores = {"img_folds": SVM_img_labels,
                  "img_fold_accuracy": SVM_img_accuracy,
                  "img_fold_f1score": SVM_img_f1score,
                  "img_fold_AUC": SVM_img_AUC
                 }
SVM_img_df = pd.DataFrame.from_dict(SVM_img_scores)
print("SVM_img_df:")
print(SVM_img_df)
SVM_img_df.to_csv(root_path + "/multimodal_cv_results/CONF/SVM_img_fold_scores.csv")


SVM_gen_scores = {"gen_folds": SVM_gen_labels,
                  "gen_fold_accuracy": SVM_gen_accuracy,
                  "gen_fold_f1score": SVM_gen_f1score,
                  "gen_fold_AUC": SVM_gen_AUC
                 }
SVM_gen_df = pd.DataFrame.from_dict(SVM_gen_scores)
print("SVM_gen_df:")
print(SVM_gen_df)
SVM_gen_df.to_csv(root_path + "/multimodal_cv_results/CONF/SVM_gen_fold_scores.csv")


SVM_img_gen_scores = {"img_gen_folds": SVM_img_gen_labels,
                  "img_gen_fold_accuracy": SVM_img_gen_accuracy,
                  "img_gen_fold_f1score": SVM_img_gen_f1score,
                  "img_gen_fold_AUC": SVM_img_gen_AUC
                 }
SVM_img_gen_df = pd.DataFrame.from_dict(SVM_img_gen_scores)
print("SVM_img_gen_df:")
print(SVM_img_gen_df)
SVM_img_gen_df.to_csv(root_path + "/multimodal_cv_results/CONF/SVM_img_gen_fold_scores.csv")


SVM_clin_gen_scores = {"clin_gen_folds": SVM_clin_gen_labels,
                  "clin_gen_fold_accuracy": SVM_clin_gen_accuracy,
                  "clin_gen_fold_f1score": SVM_clin_gen_f1score,
                  "clin_gen_fold_AUC": SVM_clin_gen_AUC
                 }
SVM_clin_gen_df = pd.DataFrame.from_dict(SVM_clin_gen_scores)
print("SVM_clin_gen_df:")
print(SVM_clin_gen_df)
SVM_clin_gen_df.to_csv(root_path + "/multimodal_cv_results/CONF/SVM_clin_gen_fold_scores.csv")


SVM_img_clin_gen_scores = {"img_clin_gen_folds": SVM_img_clin_gen_labels,
                  "img_clin_gen_fold_accuracy": SVM_img_clin_gen_accuracy,
                  "img_clin_gen_fold_f1score": SVM_img_clin_gen_f1score,
                  "img_clin_gen_fold_AUC": SVM_img_clin_gen_AUC
                 }
SVM_img_clin_gen_df = pd.DataFrame.from_dict(SVM_img_clin_gen_scores)
print("SVM_img_clin_gen_df:")
print(SVM_img_clin_gen_df)
SVM_img_clin_gen_df.to_csv(root_path + "/multimodal_cv_results/CONF/SVM_img_clin_gen_fold_scores.csv")





# Get SVM CV scores
SVM_cv_score = {"data_type" : ["clinical", "image", "genomic", "image_genomic", "clinical_genomic", "image_clinical_genomic"],
                "cv_accuracy_mean" : [statistics.mean(SVM_clin_accuracy), statistics.mean(SVM_img_accuracy), statistics.mean(SVM_gen_accuracy), statistics.mean(SVM_img_gen_accuracy), statistics.mean(SVM_clin_gen_accuracy), statistics.mean(SVM_img_clin_gen_accuracy)],
                "cv_accuracy_std" : [statistics.stdev(SVM_clin_accuracy), statistics.stdev(SVM_img_accuracy), statistics.stdev(SVM_gen_accuracy), statistics.stdev(SVM_img_gen_accuracy), statistics.stdev(SVM_clin_gen_accuracy), statistics.stdev(SVM_img_clin_gen_accuracy)],
                "cv_f1score_mean" : [statistics.mean(SVM_clin_f1score), statistics.mean(SVM_img_f1score), statistics.mean(SVM_gen_f1score), statistics.mean(SVM_img_gen_f1score), statistics.mean(SVM_clin_gen_f1score), statistics.mean(SVM_img_clin_gen_f1score)],
                "cv_f1score_std" : [statistics.stdev(SVM_clin_f1score), statistics.stdev(SVM_img_f1score), statistics.stdev(SVM_gen_f1score), statistics.stdev(SVM_img_gen_f1score), statistics.stdev(SVM_clin_gen_f1score), statistics.stdev(SVM_img_clin_gen_f1score)],
                "cv_AUC_mean" : [statistics.mean(SVM_clin_AUC), statistics.mean(SVM_img_AUC), statistics.mean(SVM_gen_AUC), statistics.mean(SVM_img_gen_AUC), statistics.mean(SVM_clin_gen_AUC), statistics.mean(SVM_img_clin_gen_AUC)],
                "cv_AUC_std" : [statistics.stdev(SVM_clin_AUC), statistics.stdev(SVM_img_AUC), statistics.stdev(SVM_gen_AUC), statistics.stdev(SVM_img_gen_AUC), statistics.stdev(SVM_clin_gen_AUC), statistics.stdev(SVM_img_clin_gen_AUC)]
               }
SVM_cv_score_df = pd.DataFrame.from_dict(SVM_cv_score)
SVM_cv_score_df.to_csv(root_path + "/multimodal_cv_results/CONF/SVM_cv_scores.csv")
print("SVM cross validation scores saved")


# In[ ]:




