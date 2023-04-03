
import shutil              
import glob                             
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
import sklearn
from IPython.display import display


root_path = "/home/io3247/HNC"
root_dirs = next(os.walk(root_path))[1]   #[0] - Root, [1] - Dirs, [2] - Files




# gen_file_path = Path to a single rna-seq file downloaded from TCGA
# label = Tag for sample... Can be patient barcode of any other designated name
# data_type = Type of data to be extracted from TCGA RNASeq record (May be "tpm_unstranded", "unstranded", etc)
def get_tpm(gen_file_path, label, data_type= "tpm_unstranded"):
    
    """Function reads in a rna_seq file, formats it appropraitely, and returns gene_name, and the TPM normalization columns."""
    """Takes in path to a single rna_seq file, and label for sample (ex. TCGA barcode or assigned number)"""
    """Returns gene_name and "tpm_unstranded" columns from a single rna_seq data."""
    
    read_seq = pd.read_csv(gen_file_path, delimiter= "\t")
    # Reset indexes... removing multiple indexing
    read_seq.reset_index(level=(0,1,2,3,4,5,6,7), inplace= True) 
    # Save the whole index 0 row into a variable
    index0row2 = read_seq.loc[0, "level_0" : "# gene-model: GENCODE v36"]    
    # Drop index 0 row from dataframe
    read_seq.drop(index = 0, inplace= True) 
    # Rename the columns using the appropraite names already saved inside "index0row" variable
    read_seq.columns = index0row2   
    # Drop rows that have NaN values under gene_name column
    read_seq.dropna(subset = ["gene_name"], inplace=True) 
    # Extract only rows that have "protein_coding" under gene_types column
    read_seq = read_seq[read_seq["gene_type"] == "protein_coding"]
    # Extract only "gene_name", and "tpm_unstranded" columns
    tpm = read_seq[["gene_name", data_type]]
    # Rename "tpm_unstanded" column to patient barcode (or label)
    tpm = tpm.rename(columns = {data_type: label})
    
    return tpm






##### FOR TPM OBLECTS   ################

### Used for comparism- 2 sets of data

# all_rna_seq_paths = A list containing paths to all rna-seq file in a category, if category exists- (in this case, can be metastasis or no metastasis categorized samples)
# data_type = Type of data extracted from TCGA RNASeq record (May be "tpm_unstranded", "unstranded", etc)
# metastasis = Whether samples are metastatic or not... Default False
def all_tpm_objs_list(all_rna_seq_paths, data_type= "tpm_unstranded", metastasis = False):
    """Function creates a list of tpm_datafame objects from each rna_seq data."""
    """Takes in a list of paths to each rna_seq file."""
    """Returns a list containing tpm_datafame objects, and sample labels of all rna_seq files."""
    objs_list = []
    samples_label_list = []
    samples_list = []
    for label, gen_file in enumerate(all_rna_seq_paths):
        if metastasis == True:           
            patient_label = "met_patient" + str(label+1)    
            patient_tpm = get_tpm(gen_file, patient_label, data_type = data_type)
        else:
            patient_label = "patient" + str(label+1)
            patient_tpm = get_tpm(gen_file, patient_label, data_type = data_type)
            
        objs_list.append(patient_tpm)
        samples_label_list.append(patient_label)
            
    samples_list =[objs_list, samples_label_list]
            
    return samples_list






### For processing genomic data for multimodal integration

# all_rna_seq_paths = A list containing paths to all rna-seq file in a category, if category exists- (in this case, can be metastasis or no metastasis categorized samples)
# data_type = Type of data extracted from TCGA RNASeq record (May be "tpm_unstranded", "unstranded", etc)
def multimodal_all_tpm_objs_list(all_rna_seq_paths, data_type = "tpm_unstranded"):       # Remove metastasis
    """Function creates a list of tpm_datafame objects from each rna_seq data."""
    """Takes in a list of paths to each rna_seq file."""
    """Returns a list containing tpm_datafame objects, and sample labels of all rna_seq files."""
    objs_list = []
    samples_label_list = []
    samples_list = []
    for gen_file in all_rna_seq_paths:
        # To get barcode
        barc_srt_ind = gen_file.find("TCGA")
        barcode = gen_file[barc_srt_ind : barc_srt_ind + 12]     
        patient_tpm = get_tpm(gen_file, barcode, data_type = data_type)            
        objs_list.append(patient_tpm)
        samples_label_list.append(barcode)
            
    samples_list = [objs_list, samples_label_list]
            
    return samples_list





#tpm_objs_lst = A list containing dataframe objects of each rna_seq data

def merge_dataframe(tpm_objs_list):
    """Function merges all dataframes of gene_name, and tpm columns from each rna_seq data, remove gene_name duplicates, and fill NaN with 0"""
    """Takes in a list containing dataframe objects of each rna_seq data"""
    """Returns a gene_name deduplicated, NaN filled whole dataframe of all rna_seq data after merging."""
    full_set = None
    for i in range(len(tpm_objs_list)):
        if i == 0:
            pass
        elif i == 1:
            full_set = pd.merge(tpm_objs_list[i-1], tpm_objs_list[i], how = "inner", on = "gene_name")
            full_set.drop_duplicates(subset ="gene_name", keep = False, inplace = True)  ######
        else:
            full_set = pd.merge(full_set, tpm_objs_list[i], how = "inner", on = "gene_name")
            full_set.drop_duplicates(subset ="gene_name", keep = False, inplace = True) ######
    # Fill NaN values with 0       
    full_set = full_set.fillna(0)
            
    return full_set





# dataframe = A dataframe of all rna_seq data after merging
def log_transform(dataframe):
    """Function carries out log to base 10 transformation on dataframe."""
    """Takes in a dataframe."""
    """ Returns a log10 dataframe after replacing NaN, and Inf values with 0."""
    # Dataframe log to base 10 transformation
    log_df = np.log10(dataframe)
    # Check again for NaN values in data (... After log transformation)
    NaNinData = log_df.isnull().sum().sum()    
    print(f"There are {NaNinData} NaN values in dataframe after log transformation")
    # Check for inf values in data (... After log transformation)
    InfInData = np.isinf(log_df).values.sum()
    print(f"There are {InfInData} Inf values in data after log transformation")
    log_df.replace([np.inf, -np.inf], 0, inplace=True)       
    log_df = log_df.fillna(0) 
    print(np.isinf(log_df).values.sum())  #Confirm Inf values are handled
    print("NaN and Inf values replaced with 0.")
    print("Log transformation complete.")
    
    return log_df





def remove_zero_genes(merged_dataframe):
    """Function removes rows(genes) with 0 value in greater than 80% of samples"""
    """Takes in a "merged dataframe" with float values"""
    """Returns a dataframe with filtered rows number based on rows(genes) values accross all samples."""
    rows_gene = list(merged_dataframe.index)
    samples_no = len(merged_dataframe.columns)
    zero_genes_threshold = round(samples_no * 0.8)    # Genes must not have 0 value in greater than 80% of samples
    for gene in rows_gene:
        row_data = list(merged_dataframe.loc[gene, : ])
        zero_count = row_data.count(0.0000)
        if zero_count > zero_genes_threshold:
            merged_dataframe = merged_dataframe.drop(index = gene)   
    print("Genes with 0 value in greter than 80% of samples removed.")
    # Re-arrange index after removing 0 genes
    # merged_dataframe = merged_dataframe.reset_index(drop=True)
    
    return merged_dataframe





def variance_filter(merged_dataframe):
    """Function filters dataframe based on value of gene variance accross all samples"""
    """Takes in a "merged dataframe" """
    """Returns a dataframe containing top 10,000 rows(genes) ordered based on genes variance accross all samples"""
    # Create a new column for variance per row (gene) accross samples
    merged_dataframe["gene_variance"] = merged_dataframe.var(axis = 1)
    # Sort dataframe by variance using the created gene_variance column
    var_sorted_df = merged_dataframe.sort_values(by = 'gene_variance', ascending = False)
    # Extract top 10,000 genes
    var_sorted_df = var_sorted_df.iloc[:10000 , : ]
    # Drop "gene_variance" column
    var_sorted_df = var_sorted_df.drop(["gene_variance"], axis = 1)
    # Re-arrange index after removing sorting genes by variance and selecting only 10,000 genes
    #var_sorted_df = var_sorted_df.reset_index(drop=True)
    print("Dataframe sorted by variance. Top 10,000 genes selected")
    
    return var_sorted_df




# apply_var_filter = Whether to apply variance filter or not
# save_filename_dot_csv = Name to save file after completing operations.

def further_preprocessing(merged_dataframe, save_filename_dot_csv = None, apply_var_filter = True):
    """Function carries out further preprocessing (float32 type conversion, zero filter, variance filter, and transposition) on dataframe after merging data from all patients(samples)."""
    """Takes in immediate "merged dataframe" without NaN values."""
    """Returns a transposed dataframe (gene on columns, samples on rows) of selected top 10,000 genes based on variance accross all samples."""
    
    
    # Confirm there is no NaN value
    NaNinCol = merged_dataframe[merged_dataframe.isna()].count()  # No of NaN values in each column (should be 0)
    NaNinData = NaNinCol.sum()  # NaN in dataframe 
    print(f"There are {NaNinData} NaN values in dataframe.")
    
    # Set "gene_name" column as index
    merged_dataframe = merged_dataframe.set_index("gene_name")
    
    # All columns currently have their values in string format which makes mathematical operations impossible
    # Convert samples tpm columns ("patient1" : ...) to float64 format
    merged_dataframe = merged_dataframe.astype("float32")
    print("Samples values converted to float32")    
    
    # Log10 transform transposed_df
    log_trans_df = log_transform(merged_dataframe)
    
    # Remove genes with 0 value in greater than 80% of samples- using remove_zero_genes() function
    removed0genes_df = remove_zero_genes(log_trans_df)
    
    # Sort dataframe by variance, and select only top 10,000 genes with highest variance accross all samples- variance_filter() function
    if apply_var_filter == True:
        var_sorted_df = variance_filter(removed0genes_df)
    else:
        var_sorted_df = removed0genes_df
    
    # Transpose dataframe for subsequent patient based analysis
    transposed_df = var_sorted_df.transpose()
    print("Dataframe transposed: Genes are now arranged in columns, and samples in rows")

    # Save transposed_df?
    save_transposed_df = input("Would you like to save transposed, preprocessed full fpkm dataframe? (y/n)")
    if save_transposed_df == "y":
        transposed_df.to_csv(save_filename_dot_csv)  # "preprocessed_df.csv" or path to preferred location, header=False, index=False)
        print(f"{save_filename_dot_csv} saved to current working directory.")
    
    return transposed_df
    


    

# X = Dependent variables dataframe from full preprocesed dataframe.
# Y = Independent variable dataframe from full preprocesed dataframe.
def rf_model(X, Y):
    """Function build multiple models, and does gene selection."""
    """Takes in extracted dependent, and independent variables from preprocessed dataframe."""
    """Returns a dataframe of top 50 genes (ranked according to highest frequency for importance within top 100 of each built model)."""

    
    # Top (100) genes, and how many times they appear in top 100 over 1000(random_states) repeats. (n_estimator= fixed no)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import metrics
    gene_counter_dict  = {}
    repeats = 1000
    for i in range(0, repeats):
        # Define model      
        model = RandomForestClassifier(n_estimators=100, random_state = i)
        # Fit model
        model.fit(X, Y)
        genes_list = list(X.columns)
        genes_imp_rank = pd.Series(model.feature_importances_, index= genes_list).sort_values(ascending=False)

        for gene in genes_imp_rank.index[:100]:
            if gene in gene_counter_dict:
                gene_counter_dict[gene] += 1
            else:
                gene_counter_dict[gene] = 1
                
    # Convert gene_counter_dict into a pd.Dataframe for easy ordering and visualization according to frequency count.
    gene_counter_list = []
    for k, v in gene_counter_dict.items():
        gene_counter_list.append((k,v))

    genes_rank = pd.DataFrame(gene_counter_list)
    genes_rank = genes_rank.rename(columns = {0: "gene_name", 1: "frequency_in_top_100"})
    genes_rank = genes_rank.sort_values(by = "frequency_in_top_100", ascending=False)
    
    return genes_rank.head(50)

