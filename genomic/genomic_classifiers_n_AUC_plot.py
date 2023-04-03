
import shutil                            
import glob                             
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
import sklearn
from IPython.display import display
import matplotlib.pylab as plt
from scipy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import StratifiedKFold
import matplotlib.patches as patches
from subprocess import check_output




# clf = classifier model
# kfold = no of crossvalidation sets
# x= df of independent variable
# y= df (or series) of dependent variable

def cv_roc(clf, k_fold, x, y):
    """
    Takes in classifier model, kfold = no of crossvalidation sets, x= independent variable, y= dependent variable.
    Returns a list cointaining at index 0: list of auc scores in 5 cross validation sets; index 1: Mean of auc score;
    index 2: Standard deviation of AUC scores.
    """
    cv = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=10)     # ... shuffle=False / random_state=1

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0,1,100)
    i = 1
    for train,test in cv.split(x,y):
        prediction = clf.fit(x.iloc[train],y.iloc[train]).predict_proba(x.iloc[test])
        fpr, tpr, t = roc_curve(y[test], prediction[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        i= i+1
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)

    std_auc = np.asarray(aucs).std()

    return [aucs, mean_auc, std_auc]




# full_df = Full dataframe containing both dependent and independent variables
# genes_list= list data_type containing gene names
# genes_num_list= list containing number of genes to be used in one prediction.


def results(full_df, genes_list, genes_num_list):
    
    """
    Takes in full_df = Full dataframe containing both dependent and independent variables, genes_list= list data_type containing gene names
    genes_num_list= list containing number of genes to be used in one prediction.
    
    Returns a dataframe of prediction values using RF, SVM, and KNN models instance, and 5 fold cross-validation using
    selected number of genes each time.
    
    """
    
    
    genes_num = []
    single_acc_RF = []
    single_acc_SVM = []
    single_acc_KNN = []
    RF_cv_acc_mean = []
    RF_cv_acc_std = []
    RF_cv_f1_mean = []
    RF_cv_f1_std = []
    RF_cv_auc_mean = []
    RF_cv_auc_std = []
    SVM_cv_acc_mean = []
    SVM_cv_acc_std = []
    SVM_cv_f1_mean = []
    SVM_cv_f1_std = []
    SVM_cv_auc_mean = []
    SVM_cv_auc_std = []
    KNN_cv_acc_mean = []
    KNN_cv_acc_std = []
    KNN_cv_f1_mean = []
    KNN_cv_f1_std = []
    KNN_cv_auc_mean = []
    KNN_cv_auc_std = []
    
    
    # Select top 30, 25, 20, 15, 10, 5, 3, 2, 1 for prediction, and record results
    for num in genes_num_list:
        
        print(f"Now getting results for {num} genes...")
        sel_genes = genes_list[:num]
        
        # Seperate independent variables
        X = full_df.loc[ : , sel_genes]
        print(X.shape)
        # Extract dependent variable
        Y = full_df["metastasis"].astype("int")
        print(Y.shape)
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=10)
        print(f"X_train samples number: {len(X_train)}, Y_train samples number: {len(Y_train)}")
        print(f"X_test samples number: {len(X_test)}, Y_test samples number: {len(Y_test)}")
        
        
        # Train an instance RF model
        model = RandomForestClassifier(n_estimators=70, random_state = 10)
        # Fit model
        model.fit(X_train, Y_train)
        # Test model on X_test
        model_test = model.predict(X_test)
        # Get test accuracy score for each model training instance
        instance_accuracy = metrics.accuracy_score(Y_test, model_test)  
        conf_mat = metrics.confusion_matrix(Y_test, model_test)
        single_acc_RF.append(instance_accuracy)
        
        
        # Train an instance SVM model
        svm_model_linear = SVC(kernel = 'linear', C = 1, random_state = 10).fit(X_train, Y_train)
        svm_predictions = svm_model_linear.predict(X_test)
        # model accuracy for X_test  
        svm_accuracy = svm_model_linear.score(X_test, Y_test)
        svm_cm = metrics.confusion_matrix(Y_test, svm_predictions)
        single_acc_SVM.append(svm_accuracy)
        
        
        # Train an instance KNN model
        knn = KNeighborsClassifier(n_neighbors = 19).fit(X_train, Y_train)
        knn_predictions = knn.predict(X_test)
        # accuracy on X_test
        knn_accuracy = knn.score(X_test, Y_test)
        knn_cm = metrics.confusion_matrix(Y_test, knn_predictions)
        single_acc_KNN.append(knn_accuracy)
        
        
        # Train RF model with 5-fold cross-validation
        #n_samples = X.shape[0]
        RF_cv = RandomForestClassifier(n_estimators=70, random_state = 10)
        cv = ShuffleSplit(n_splits=5, test_size=0.25, random_state = 10)
        # Accuracy on each cross validation set
        RF_cv_scores = cross_val_score(RF_cv, X, Y, cv=cv)
        # F1 score on each cross validation set
        RF_cv_f1 = cross_val_score(RF_cv, X, Y, scoring = "f1", cv=cv)
        # AUC score on each cross validation set
        RF_cv_auc = cv_roc(RF_cv, 5, X, Y)
        RF_cv_acc_mean.append(RF_cv_scores.mean())
        RF_cv_acc_std.append(RF_cv_scores.std())
        RF_cv_f1_mean.append(RF_cv_f1.mean())
        RF_cv_f1_std.append(RF_cv_f1.std())
        RF_cv_auc_mean.append(RF_cv_auc[1])
        RF_cv_auc_std.append(RF_cv_auc[2])
        
        
        
        # Cross-validation model SVM
        svm_cv = SVC(kernel='linear', C=1, probability=True, random_state = 10)
        cv = ShuffleSplit(n_splits=5, test_size=0.25, random_state = 10)
        svm_cv_scores = cross_val_score(svm_cv, X, Y, cv=cv)
        # F1 score on each cross validation set
        svm_cv_f1 = cross_val_score(svm_cv, X, Y, scoring = "f1", cv=cv)
        # AUC score on each cross validation set
        svm_cv_auc = cv_roc(svm_cv, 5, X, Y)
        SVM_cv_acc_mean.append(svm_cv_scores.mean())
        SVM_cv_acc_std.append(svm_cv_scores.std())
        SVM_cv_f1_mean.append(svm_cv_f1.mean())
        SVM_cv_f1_std.append(svm_cv_f1.std())
        SVM_cv_auc_mean.append(svm_cv_auc[1])
        SVM_cv_auc_std.append(svm_cv_auc[2])
        
        
        # Cross-validation model KNN
        knn_cv = KNeighborsClassifier(n_neighbors = 19)
        cv = ShuffleSplit(n_splits=5, test_size=0.25, random_state = 10)
        knn_cv_scores = cross_val_score(knn_cv, X, Y, cv=cv)
        # F1 score on each cross validation set
        knn_cv_f1 = cross_val_score(knn_cv, X, Y, scoring = "f1", cv=cv)
        # AUC score on each cross validation set
        knn_cv_auc = cv_roc(knn_cv, 5, X, Y)
        KNN_cv_acc_mean.append(knn_cv_scores.mean())
        KNN_cv_acc_std.append(knn_cv_scores.std())
        KNN_cv_f1_mean.append(knn_cv_f1.mean())
        KNN_cv_f1_std.append(knn_cv_f1.std())
        KNN_cv_auc_mean.append(knn_cv_auc[1])
        KNN_cv_auc_std.append(knn_cv_auc[2])
        
        # Save number of genes used here
        genes_num.append(num)
        
        
    res_dict = {"no_of_genes": genes_num,
                "instance_acc_RF": single_acc_RF,
                "RF_cv_acc_mean": RF_cv_acc_mean,
                "RF_cv_acc_std": RF_cv_acc_std,
                "RF_cv_f1_mean": RF_cv_f1_mean,
                "RF_cv_f1_std": RF_cv_f1_std,
                "RF_cv_auc_mean": RF_cv_auc_mean,
                "RF_cv_auc_std": RF_cv_auc_std,
                "instance_acc_SVM": single_acc_SVM,
                "SVM_cv_acc_mean": SVM_cv_acc_mean,
                "SVM_cv_acc_std": SVM_cv_acc_std,
                "SVM_cv_f1_mean": SVM_cv_f1_mean,
                "SVM_cv_f1_std": SVM_cv_f1_std,
                "SVM_cv_auc_mean": SVM_cv_auc_mean,
                "SVM_cv_auc_std": SVM_cv_auc_std,
                "instance_acc_KNN": single_acc_KNN,             
                "KNN_cv_acc_mean": KNN_cv_acc_mean,
                "KNN_cv_acc_std": KNN_cv_acc_std,
                "KNN_cv_f1_mean": KNN_cv_f1_mean,
                "KNN_cv_f1_std": KNN_cv_f1_std,
                "KNN_cv_auc_mean": KNN_cv_auc_mean,
                "KNN_cv_auc_std": KNN_cv_auc_std
               }
    res_df = pd.DataFrame.from_dict(res_dict)
    
    return res_df 



# full_df = Full dataframe containing both dependent and independent variables
# genes_list= list data_type containing gene names
# genes_num_list= list containing number of genes to be used in one prediction.

def cv_roc_plot(clf, k_fold, x, y):
    
    """
    Takes in classifier model, kfold = no of crossvalidation sets, x= independent variable, y= dependent variable.
    - SAVES PLOT;     AND
    Returns a list cointaining at index 0: list of auc scores in 5 cross validation sets; index 1: Mean of auc score;
    index 2: Standard deviation of AUC scores.
    """
    
    cv = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=10)     # ... shuffle=False / random_state=1
    
    fig1 = plt.figure(figsize=[12,12])
    ax1 = fig1.add_subplot(111,aspect = 'equal')

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0,1,100)
    i = 1
    for train,test in cv.split(x,y):
        prediction = clf.fit(x.iloc[train],y.iloc[train]).predict_proba(x.iloc[test])
        fpr, tpr, t = roc_curve(y[test], prediction[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i= i+1
    plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='blue',
             label=r'Mean ROC (AUC = %0.2f )' % (mean_auc),lw=2, alpha=1)

    plt.xticks(fontsize= 40)
    plt.xlabel('False Positive Rate', fontsize= 40)
    plt.yticks(fontsize= 40)
    plt.ylabel('True Positive Rate', fontsize= 40)
    plt.title('ROC')
    plt.legend(loc="lower right", fontsize= 30)
    plt.show()
    
    # Save plot
    save_auroc_plot = input("Would you like to save auroc plot for 5 fold cross-validation? (y/n) \n")
    if save_auroc_plot == "y":
        plot_filename = input("Please enter filename to save with... add .png at the end. \n")
        fig1.savefig(plot_filename)  
        print(f"{plot_filename} saved to current working directory.")

    std_auc = np.asarray(aucs).std()

    return [aucs, mean_auc, std_auc]

