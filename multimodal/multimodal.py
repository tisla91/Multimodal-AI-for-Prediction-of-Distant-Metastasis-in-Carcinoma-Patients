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



  

# root = root directory
def barcode_tiles_path_list(root):
    
    """Create a list of paths to each patient tiles folder (All patients)"""
    
    tiles_path_list = []
    for barcode in os.listdir(root) :
        tiles_path = root + "/" + barcode + "/" + barcode + "_histopath" + "/" + barcode + "_tiles"
        tiles_path_list.append(tiles_path)
    return tiles_path_list




# data_root = Root folder (for HNSC, PAAD, or BLCA) containing dir for each patient
def barc_path_df(data_root):
    """
    Creates a 2 column dataframe of patient barcode, and path to image files.
    """
    tiles_path_list = barcode_tiles_path_list(data_root)
    barcode_path_tuple_list = []
    for patient_path in tiles_path_list:
        barc_srt_ind = patient_path.find("TCGA")
        patient_barcode = patient_path[barc_srt_ind : barc_srt_ind + 12]
        tupl = (patient_barcode, patient_path)
        barcode_path_tuple_list.append(tupl)

    dataframe = pd.DataFrame(barcode_path_tuple_list, columns=["barcode", "tiles_path"])
    
    return dataframe



# tiles_path = One patient's tiles directory
def barcode_tiles_arr(tiles_path):
    
    """Stain, and scale normalizes all tiles inside one patient's tiles directory, and loads them into an array."""
    
    tiles_list = []  
    extension = "/*.png"
    png_file_list = glob.glob(tiles_path + extension)  
    for tile in png_file_list:       
        try:
            read_tile = skimage.io.imread(tile)
            normalizer = StainNormalizationHE(target = "normalize", stain_estimation_method = "macenko")
            norm_tile = normalizer.F(read_tile)
            scale_tile = norm_tile / 255
            tiles_list.append(scale_tile)
        except:
            print(tile)  
            pass       
    tiles_arr = np.asarray(tiles_list)  
    return tiles_arr



# pat_tiles_path = A list containing path to each patients train tiles
def norm_scale_load(pat_tiles_path):
    
    '''
    Normalize, scale, and load train images
    ''' 
    tiles_path_train = []         # Stores path to tiles of patients included in train set
    all_train_imgs = []           # Stores images of all patients in train set
    
    for barc_path in list(pat_tiles_path):
        pat_img = barcode_tiles_arr(barc_path)
        # Exclude patients with less than 12 images
        if pat_img.shape[0] < 12:
            print("Patient tiles shape: ", pat_img.shape)
            print(barc_path,  "has less than 12 images. Excluded.")
        else:
            print("Patient tiles shape: ", pat_img.shape)
            tiles_path_train.append(barc_path)
            print("Patient images loaded.")
            all_train_imgs.append(pat_img)
            
    print("All train images loaded.")    
    print("Train images number: ", len(all_train_imgs))  
    return (all_train_imgs, tiles_path_train)



# Give slide label to the tiles

# train_imgs_list = List of patients' train image list.
# y_train = label for each patient from train_test split.
def label_tiles(train_imgs_list, y_train):
    '''
    Give label to  tiles from patient label
    '''
    tiles_label= []
    for i in range(0, len(train_imgs_list)):
        imgs_num = train_imgs_list[i].shape[0]   
        if list(y_train)[i] == 0:        
            imgs_labels = np.zeros(imgs_num, dtype = np.uint8)
        elif list(y_train)[i] == 1:
            imgs_labels = np.ones(imgs_num, dtype = np.uint8)
        tiles_label.append(imgs_labels)

    labels = np.concatenate(tiles_label, axis = 0)
    
    return labels



# train_imgs= Images for training model
# train_labels = Class labels for training model
# model_save_name = Filename to save model
# model_save_weight = Filename to save model weight
# model_chkpnt_save_name = Filename to save mode check point
def create_model_densenet(train_imgs, train_labels, model_save_name, model_weight_save_name, chkpnt_save_name):
    """
    Defines, build and save a model. Returns the model.
    """
    
    # Define Model
    np.random.seed = 10
    model_base = DenseNet121(weights = 'imagenet', include_top = False, input_shape=(512,512,3))
    for layer in model_base.layers:
            layer.trainable=False                 
    inputs=tf.keras.Input(shape=(512,512,3))
    x=model_base(inputs)                            
    x=GlobalAveragePooling2D()(x) 
    x=Dropout(0.1)(x)                             
    x=Dense(512, activation = "relu")(x)
    x=Dropout(0.2)(x)
    x=Dense(32, activation = "relu")(x)
    x=Dropout(0.05)(x)   
    x=Dense(1, activation = "sigmoid")(x)
    model=Model(inputs=[inputs],outputs=[x])
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)  
    model.compile(loss = "binary_crossentropy", optimizer = opt, metrics = ["accuracy"])
    model.summary()
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(chkpnt_save_name, verbose = 1, save_best_only = True)  
    callbacks = [checkpoint]                  

    model.fit(train_imgs, 
                train_labels,                           
                shuffle = True,
                validation_split = 0.25,             
                batch_size = 32, 
                epochs = 40, 
                callbacks = callbacks,
                verbose = 2)
    
    model.save(model_save_name)
    
    model.save_weights(model_weight_save_name)
    
    print("Model and weight saved!")
    
    return model




# train_imgs= Images for training model
# train_labels = Class labels for training model
# model_save_name = Filename to save model
# model_save_weight = Filename to save model weight
# model_chkpnt_save_name = Filename to save mode check point
def create_model_kimia(train_imgs, train_labels, model_save_name, model_weight_save_name, chkpnt_save_name):
    """
    Defines, build and save a model. Returns the model.
    """
    np.random.seed = 10
    dnx = DenseNet121(include_top=False, weights="KimiaNetKerasWeights.h5", 
                      input_shape=(512, 512, 3), pooling='avg')
    layer_name = "conv5_block16_concat"    
    layer_output = dnx.get_layer(layer_name).output        
    dnx_model = Model(dnx.input, layer_output)  

    dnx_model.trainable = False

    inputs=tf.keras.Input(shape=(512,512,3))    
    x=dnx_model(inputs)
    x=GlobalAveragePooling2D()(layer_output)
    x=Dropout(0.1)(x)                             
    x=Dense(512, activation = "relu")(x)
    x=Dropout(0.2)(x)
    x=Dense(32, activation = "relu", name = "features_extractor")(x)    
    x=Dropout(0.05)(x)   
    x=Dense(1, activation = "sigmoid")(x)
    model = Model(dnx.input, outputs=[x])    
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)  
    model.compile(loss = "binary_crossentropy", optimizer = opt, metrics = ["accuracy"])  
    model.summary()
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(chkpnt_save_name, verbose = 1, save_best_only = True)
    callbacks = [checkpoint] 

    model.fit(train_imgs, 
                train_labels,          
                shuffle = True,
                validation_split = 0.25,          
                batch_size = 32, 
                epochs = 40, 
                callbacks = callbacks,
                verbose = 2)
    
    model.save(model_save_name)    
    model.save_weights(model_weight_save_name)  
    print("Model and weight saved!")
    
    return model





# model_type = Type of CNN model. ex: "DenseNet121", "KimiaNet"   etc
# input_data = 4D or (5D) array of all tiles for a single patient.
# model_save_name = Name of trained model... For features extraction.
# Weight_address = Path to KimiaNet weight (model)- Only to useful if model_type = "KimiaNet".
def model_definition(model_type, input_data, model_save_name, weights_address = "/home/io3247/HNC/KimiaNetKerasWeights.h5"):
    
    """Extracts features from input data. Output features is 1D."""
    
    if model_type == "KimiaNet":
        loaded_model = load_model(model_save_name)
        layer_name = "features_extractor"    
        layer_output = loaded_model.get_layer(layer_name).output        
        kimia_model = Model(loaded_model.input, layer_output)
        for layer in kimia_model.layers:
            layer.trainable = False
        inputs = tf.keras.Input(shape = (None, 512, 512, 3))
        x=TimeDistributed(kimia_model)(inputs)
        x=GlobalAveragePooling1D()(x)
        custom_kimia = Model(inputs, x)
        features = custom_kimia.predict(input_data)[0,:]  
        
        return features
    
    elif model_type == "DenseNet121":
        loaded_model = load_model(model_save_name)    
        for layer in loaded_model.layers:
            layer.trainable=False  
        inputs = tf.keras.Input(shape = (None, 512, 512, 3))
        x = inputs
        for layer in loaded_model.layers[1: -1]: 
            layer.trainable=False
            x=TimeDistributed(loaded_model.get_layer(layer.name))(x)   
        x=GlobalAveragePooling1D()(x)
        custom_model = Model(inputs, x, name = 'custom_model')
        features = custom_model.predict(input_data)[0,:]    #[0,:]

        return features


# pat_paths_lst = A list containing path to each patients train tiles
# model_type = Type of CNN model. ex: "DenseNet121", "KimiaNet", etc
def norm_scale_ext_feat(pat_paths_lst, model_type, model_save_name):

    # Normalize, scale, and extract features for train images
    tiles_path_train = []
    all_train_features = []
    for barc_path in pat_paths_lst:
        pat_img = barcode_tiles_arr(barc_path)
        # Exclude patients with less than 12 images
        if pat_img.shape[0] < 12:
            print("Patient tiles shape: ", pat_img.shape)
            print(barc_path,  "has less than 12 images. Excluded.")
        else:
            print("Patient tiles shape: ", pat_img.shape)
            # Save path
            tiles_path_train.append(barc_path) 
            pat_img_5d = np.expand_dims(pat_img, axis = 0)
            pat_img_features = model_definition(model_type, pat_img_5d, model_save_name)
            print("Feature extracted.")
            print("Features shape: ", pat_img_features.shape)
            all_train_features.append(pat_img_features)
    
    all_train_features = np.asarray(all_train_features)
    print("All features extracted.")    
    print("Features shape: ", all_train_features.shape)   
    
    return [tiles_path_train, all_train_features]
    
    
    

# full_df = A df containing barcodes, path to tiles, genomic, and clinical data
# full_df = A df containing barcodes, path to tiles, genomic, and clinical data
# Output_files = A list of (6) paths to save features, and tiles path. 
# (Order: train_features_file, y_train_file, train_paths_file, test_features_file,  y_test_file, test_paths_file)
# model_type = Type of CNN model. ex: "VGG16", "VGG19", "ResNet50", "InceptionV3".
# model_save_name = Name of trained model for current fold.

def extract_feature(full_df, train_paths, test_paths, output_files, model_type, model_save_name):
    
    """Collects features for all patients, and saves it in output directory."""
    
    #save files
    train_features_file = output_files[0]
    y_train_file = output_files[1]
    train_paths_file = output_files[2]
    test_features_file = output_files[3]
    y_test_file = output_files[4]
    test_paths_file = output_files[5]
       
    X_train = full_df[full_df["tiles_path"].isin(train_paths)]
    Y_train = X_train["met_status"].astype("int")
    X_test = full_df[full_df["tiles_path"].isin(test_paths)]
    Y_test = X_test["met_status"].astype("int")
        
    # Normalize, scale, and extract features for train images 
    train_ext = norm_scale_ext_feat(X_train["tiles_path"], model_type, model_save_name)
    tiles_path_train = train_ext[0]
    all_train_features = train_ext[1]
    # Normalize, scale, and extract features for test images
    test_ext = norm_scale_ext_feat(X_test["tiles_path"], model_type, model_save_name)
    tiles_path_test = test_ext[0]
    all_test_features = test_ext[1]
    
    print("Features extraction completed. To save features...")
    
    # Save train, and test features as numpy array
    np.save(train_features_file, all_train_features)
    np.save(y_train_file, Y_train)
    np.save(test_features_file, all_test_features)
    np.save(y_test_file, Y_test)
    
    # Save barcode path to tiles for train, and test set
    np.save(train_paths_file, tiles_path_train)
    np.save(test_paths_file, tiles_path_test)
    
    print("Features, and paths saved for train, and test sets.")
    
    
    return [all_train_features, all_test_features]



# img_gen_df = A df containing barcodes, path to tiles, genomic, and clinical data
# train_feat = Features extracted from train set images
# train_label = class of training dataset
# train_path = Path to images in train dataset
# test_feat = Features extracted from test set images
# test_label = class of test dataset
# test_path = Path to images in test dataset
def pred_data_prep(img_gen_df, train_feat, train_label, train_path, test_feat, test_label, test_path):
    
    """
    Prepares features for prediction.
    """
    # Load train features
    load_train_feat = np.load(train_feat)
    print(load_train_feat.shape)

    # Load train labels
    load_train_lab = np.load(train_label)
    print(load_train_lab.shape)

    # Load train paths
    load_train_path = np.load(train_path)
    print(load_train_path.shape)

    # Load test features
    load_test_feat = np.load(test_feat)
    print(load_test_feat.shape)

    # Load test labels
    load_test_lab = np.load(test_label)
    print(load_test_lab.shape)

    # Load test paths
    load_test_path = np.load(test_path)
    print(load_test_path.shape)
    
    
    # Load already saved RF top genes
    # hnsc_RF_top_genes = pd.read_csv("top_genes_HNC_tpm.csv")
    hnsc_RF_top_genes = pd.read_csv("/home/io3247/GenFeatSel/top_genes_HNC_tpm.csv")
    hnsc_ML_top = list(hnsc_RF_top_genes["gene_name"])         # All 50 ML slected genes

    # paad_RF_top_genes = pd.read_csv("top_genes_PAAD_tpm.csv")
    paad_RF_top_genes = pd.read_csv("/home/io3247/GenFeatSel/top_genes_PAAD_tpm.csv")
    paad_ML_top = list(paad_RF_top_genes["gene_name"])         # All 50 ML slected genes

    # blca_RF_top_genes = pd.read_csv("top_genes_BLCA_tpm.csv")
    blca_RF_top_genes = pd.read_csv("/home/io3247/GenFeatSel/top_genes_BLCA_tpm.csv")
    blca_ML_top = list(blca_RF_top_genes["gene_name"])         # All 50 ML slected genes

    combined = hnsc_ML_top + paad_ML_top + blca_ML_top
    
    
    # Get train genomic data from df
    train_paths = list(load_train_path)
    train_gen_df = img_gen_df[img_gen_df["tiles_path"].isin(train_paths)]   ###.isin()
    train_gen = train_gen_df.loc[ : , combined]                      # genes_lst]
    train_gen = train_gen.to_numpy()                        # Convert genes df to numpy array
    # Concatenate train  image, and genomic data
    train_img_gen = np.concatenate((load_train_feat, train_gen), axis = 1)
    train_lab = np.asarray(train_gen_df["met_status"]).astype("int")
    print("Genomic train data shape: ", train_gen.shape)
    print("Image train features shape: ", load_train_feat.shape)
    print("Image + Genomic train data shape: ", train_img_gen.shape)
    print("Train labels shape: ", train_lab.shape)

    # Get test genomic data from df
    test_paths = list(load_test_path)
    test_gen_df = img_gen_df[img_gen_df["tiles_path"].isin(test_paths)]   ###.isin()
    test_gen = test_gen_df.loc[ : , combined]                      # genes_lst]
    test_gen = test_gen.to_numpy()                        # Convert genes df to numpy array

    # Concatenate test  image, and genomic data
    test_img_gen = np.concatenate((load_test_feat, test_gen), axis = 1)
    test_lab = np.asarray(test_gen_df["met_status"]).astype("int")
    print("Genomic test data shape: ", test_gen.shape)
    print("Image test features shape: ", load_test_feat.shape)
    print("Image + Genomic test data shape: ", test_img_gen.shape)
    print("Test labels shape: ", test_lab.shape)
    
    # Clinical variables
    clin_feats = ["age", "number_of_lymphnodes_positive_by_he", "t_stage", "n_stage"]    
    train_clin_df = img_gen_df[img_gen_df["tiles_path"].isin(train_paths)]   ###.isin()
    train_clin = train_clin_df.loc[ : , clin_feats]                      # genes_lst]
    train_clin = train_clin.to_numpy()                        # Convert genes df to numpy array
    print("Clinical train shape: ", train_clin.shape) 
    
    # Concatenate train  image, genomic, and clinical data
    train_img_gen_clin = np.concatenate((load_train_feat, train_gen, train_clin), axis = 1)
    print("Image + Genomic + clinical train shape : ", train_img_gen_clin.shape)
    train_lab = np.asarray(train_gen_df["met_status"]).astype("int")
    print("Trian label shape: ", train_lab.shape)

    # Get test clinical data from df
    test_paths = list(load_test_path)
    test_clin_df = img_gen_df[img_gen_df["tiles_path"].isin(test_paths)]   ###.isin()
    test_clin = test_clin_df.loc[ : , clin_feats]                      # genes_lst]
    test_clin = test_clin.to_numpy()                        # Convert genes df to numpy array
    print("Clinical test shape: ", test_clin.shape)

    # Concatenate test  image, genomic, and clinical data
    test_img_gen_clin = np.concatenate((load_test_feat, test_gen, test_clin), axis = 1)
    print("Image + Genomic + Clinical test shape : ", test_img_gen_clin.shape)
    test_lab = np.asarray(test_gen_df["met_status"]).astype("int")
    print("Test label shape: ", test_lab.shape)
    
    # CLinical + Genomic
    train_clin_gen = np.concatenate((train_clin, train_gen), axis = 1)
    print("Clinical + Genomic train shape: ", train_clin_gen.shape)
    test_clin_gen = np.concatenate((test_clin, test_gen), axis = 1)
    print("Clinical + Genomic test shape: ", test_clin_gen.shape)
    
    return [load_train_feat, load_test_feat, train_img_gen, test_img_gen, train_gen, test_gen, train_clin, test_clin, train_img_gen_clin, test_img_gen_clin, train_clin_gen, test_clin_gen, train_lab, test_lab]




from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredLogarithmicError
from sklearn import metrics

# train_arr = Training set array
# train_lab = class of train dataset
# test_arr = Test set array
# test_lab = class of test dataset
# features_dim = number of features
def MLP(train_arr, train_lab, test_arr, test_lab, features_dim, model_layers = 1):
    
    """
    Train model on tabular features extracted from uni- or multi- modal data.
    
    Predicts on test set, and outputs prediction metrics.
    """
    # Set reproducibility seed
    tf.random.set_seed(10)
    
    # Build model
    if model_layers == 1:
        model = Sequential()
        model.add(Dense(features_dim, input_shape = (features_dim, ), kernel_initializer = "he_normal", activation = "relu"))
        model.add(Dense(1, activation= 'sigmoid'))
        
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)  
    
    METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc')
       ]
    
    model.compile(loss = "binary_crossentropy", optimizer = opt, metrics = METRICS)  # alternative: loss = "mse"
    model.summary()
    
    model.fit(train_arr, 
                train_lab,        
                epochs = 400,         
                verbose = 2)
    print("Model fitted. About to start prediction.")
    
    # Predict with model
    from sklearn.metrics import confusion_matrix, classification_report
    predictions = model.predict(test_arr)
    predictions
    print("Prediction on test set carried out. Getting confusion matrix...")
    
    # Confusion Matrix
    pred = (predictions[:, 0] > 0.5).astype(np.uint8)
    print(pred.shape, test_lab.shape)
    conf_mat = confusion_matrix(test_lab, pred)
    conf_mat
    print("Confusion matrix generated.")
    
    # Other metrics... Precision, Recall, AUC
    testset_metrics = model.evaluate(test_arr, test_lab)
    
    # F1 score
    f1_scr = metrics.f1_score(test_lab, pred)
    
    return (pred, conf_mat, testset_metrics, f1_scr)



import matplotlib.pylab as plt
from scipy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import StratifiedKFold
import matplotlib.patches as patches
from subprocess import check_output


# training a linear SVM classifier
from sklearn.svm import SVC
def SVM_pred(x_train, y_train, x_test, y_test):
    
    '''
    Predicts with SVM an returns scores ... Accuracy, f1, AUC, and confusion matrix
    '''
    
    svm_linear_img = SVC(kernel = 'linear', C = 1, random_state=10).fit(x_train, y_train)
    svm_pred_img = svm_linear_img.predict(x_test)

    # model accuracy for X_test  
    svm_acc_img = svm_linear_img.score(x_test, y_test)
    print(svm_acc_img)

    # model F1 score for X_test  
    svm_f1_img = metrics.f1_score(y_test, svm_pred_img)
    print("F1 score: ", svm_f1_img)

    # creating a confusion matrix
    svm_cm_img = metrics.confusion_matrix(y_test, svm_pred_img)
    print(svm_cm_img)

    # AUC Score
    svm_auc_mod = SVC(kernel = 'linear', C = 1, probability=True).fit(x_train, y_train)
    svm_auc_pred = svm_auc_mod.predict_proba(x_test)    # ROC_AUC calculation require probability of positive class!
    fpr, tpr, t = roc_curve(y_test, svm_auc_pred[:, 1])
    # tprs.append(interp(mean_fpr, fpr, tpr))
    svm_roc_auc = auc(fpr, tpr)
    print("AUC Score: ", svm_roc_auc)
    
    return [svm_acc_img, svm_f1_img, svm_roc_auc, svm_cm_img]

