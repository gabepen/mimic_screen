import argparse
import numpy as np
from glob import glob 
import os 
from tqdm import tqdm
import json
import re
import sys
from statistics import mean 
from Bio.PDB import PDBParser
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter, PercentFormatter
import parse_hyphy_output
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu
from skbio.stats.distance import DistanceMatrix
from skbio.stats.distance import permanova
from scipy.spatial.distance import pdist, squareform
import alignment_stat_tests


def calculate_average_pLDDT(pdb_file):
    
    '''Calculates the average pLDDT score from a pdb file 
       pLDDT is located in the b-factor feild of an atom entry
       in a standard format pdb
    '''

    # open and read the pdb file
    try:
        with open(pdb_file) as pf:
            lines = pf.readlines()
    except FileNotFoundError:
        return 'NA'
    
    # parse
    pLDDTs = []
    for l in lines:
        flds = l.strip().split()

        # verify line in file represents an atom
        if flds[0] == 'ATOM':

            # append to pLDDT list catching issues with formatting of line
            try:
                pLDDTs.append(float(flds[-2]))
            except IndexError:
                print('pLDDT ERROR')
                print(pdb_file)
                return 'ERR' 
            
    # return average pDDT for structure
    return mean(pLDDTs)
    
def generate_control_dictionary(control_dir):

    '''Generates and returns control dictionary which contains statistics for each query protein
       as it pertains to its RBH hits in control proteomes
    '''

    control_dict = {}

    # collect free-living control alignments from directory 
    control_alignments = glob(control_dir + '/*.tsv')
    total_taxon = len(control_alignments)

    for fseek_align in tqdm(control_alignments):

        # open and parse alignment file
        with open(fseek_align, 'r') as tsv_f:
            lines = tsv_f.readlines()

            # parse line components
            for l in lines:
                lp = l.strip().split()

                # collect foldseek output values
                query = lp[0]
                target= lp[1]
                evalue = float(lp[2])
                score = float(lp[3])
                tcov = float(lp[6]) 
                qcov = float(lp[7])
                fident = float(lp[12])
                

                # check for RBH hit within accepted evalue range
                if evalue < 0.01 and score > 0.3:

                    # initialize query in stat dictionary
                    if query not in control_dict:
                        control_dict[query] = {'algn_fraction': 0, 'counts': 1, 'tm_score_avg': score, 'tcov_avg': tcov, 'fident_avg': fident}

                    # update running totals  
                    else:
                        control_dict[query]['counts'] += 1
                        control_dict[query]['tm_score_avg'] += score
                        control_dict[query]['tcov_avg'] += tcov
                        control_dict[query]['fident_avg'] += fident

    # update running totals and save as averages 
    hc_count = 0 
    for query in control_dict:
        control_dict[query]['algn_fraction'] = control_dict[query]['counts'] / total_taxon
        control_dict[query]['tm_score_avg'] =  control_dict[query]['tm_score_avg'] /  control_dict[query]['counts']
        control_dict[query]['tcov_avg'] =  control_dict[query]['tcov_avg'] /  control_dict[query]['counts']
        control_dict[query]['fident_avg'] =  control_dict[query]['fident_avg'] /  control_dict[query]['counts']
        if control_dict[query]['algn_fraction'] >= 0.8:
            hc_count += 1
    
    
    # save to json 
    with open(control_dir + '/control_alignment_statistics.json', 'w+') as o_file:
        json.dump(control_dict, o_file, indent=2) 
    
    return control_dict
          
def alignment_stats(alignment, control_dict):

    '''
    checks foldseek controls output for high structural similarity
    stores name of query protein in dict object for look up when parsing expirement alignment
    stores all alignments in a data table with selection indicating field (controlled or not)

    currently configured to collect average stats of each query across the free-living proteomes
    ''' 

    data_table = []
    average_pLDDTs = {}
    
    #parse expiremental alignment 
    with open(alignment, 'r') as tsv:
        lines = tsv.readlines()

        # for alignment
        for l in lines:
            lp = l.strip().split()

            # foldseek alignment values 
            query = lp[0]
            target = lp[1] 
            evalue = float(lp[2])
            fident = float(lp[12])
            score = lp[3]
            tcov =  float(lp[6])
            qcov = float(lp[7])
            
                
            # alignment base statistic requirements
            if float(score) > 0.4 and  evalue < 0.01 and (tcov > 0.25 or qcov >= 0.5):
                
                # check presence in control proteomes
                if query in control_dict:
                    data_table.append([query, float(score), tcov, qcov, fident, control_dict[query]['algn_fraction'], target, 'candidate'])
                else:
                    data_table.append([query, float(score), tcov, qcov, fident,0.0, target,'candidate'])      

            else:
                try:
                    data_table.append([query, float(score), tcov, qcov, fident, control_dict[query]['algn_fraction'], target, 'filtered'])
                except KeyError:
                    data_table.append([query, float(score), tcov, qcov, fident,0.0, target,'filtered']) 

    
    return data_table

def validation(df, ids_of_interest, structure_db=None):

    # find validation IDs in results
    average_pLDDTs = {}
    for index, row in df.iterrows():
        for uni_id in ids_of_interest:
            if uni_id in row['query']:
  
                # calculate structure prediction confidence
                if uni_id not in average_pLDDTs and structure_db != None:
                    average_pLDDTs[uni_id] = calculate_average_pLDDT(structure_db + '/AF-' + uni_id + '-F1-model_v4.pdb')

                row_string = row[['query', 'target', 'score', 'tcov', 'qcov', 'fident', 'algn_fraction', 'branch_fraction']].astype(str).str.cat(sep=',') + ',' + str(average_pLDDTs[uni_id])
                print(row_string)           
   
def make_output_df(data_table):
    
    '''cleans up the results table and returns as a pandas dataframe
    '''

    output_table = []
    # get results 
    pairs = set()
    ids = set()
    for row in data_table:

        if row[-1] == 'filtered':
            continue
        
        # clean up file name to just have UNIProt ID
        start = 'AF-'
        end = '-F'
        target_id = re.search(f'{start}(.*?){end}', row[6]).group(1) if re.search(f'{start}(.*?){end}', row[6]) else row[6]
        query_id = re.search(f'{start}(.*?){end}', row[0]).group(1) if re.search(f'{start}(.*?){end}', row[0]) else row[0].split('_')[0]
        
        alignment_pair  = (query_id, target_id) 
        
        if alignment_pair not in pairs and query_id not in ids:
            # format output order
            output = [query_id, target_id, row[1], row[2], row[3], row[4], row[5]]
            output_table.append(output)
            pairs.add(alignment_pair)
            ids.add(query_id)
            
    # Convert data_table to pandas DataFrame
    df = pd.DataFrame(output_table, columns=['query', 'target', 'score', 'tcov', 'qcov', 'fident', 'algn_fraction'])

    return df
           
def plot_freeliving_fraction_distribution(data_table, output_path):

    '''plots a simple histogram of the pct_freeliving column of a results datatable
    '''

    # create np array from column 4 of data_table 
    column_data = [row[4] for row in data_table if row[-1] != 'filtered']
    pct_freeliving_array = np.array(column_data, dtype=float)
   
    # create histogram 
    plt.hist(pct_freeliving_array, bins=30)
    plt.xlim(0.0,1.0)
    plt.xticks(np.arange(0.0,1.0,0.2))
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places

    # save plot to output path 
    plt.savefig(output_path)
    
def control_evorate_stats(data_frame):
    
    # coloring by significance based on BUSTED results
    higher_fg_rate = []
    higher_bg_rate = []
    no_difference = []
    for index, row in data_frame.iterrows():
        if row['test_shared_p_value'] < 0.05:
            if row['test_p_value'] > row['background_p_value'] and row['background_p_value'] < 0.05:
                higher_bg_rate.append(index)
            elif row['test_p_value'] < row['background_p_value'] and row['test_p_value'] < 0.05:
                higher_fg_rate.append(index) 
                print(row['query'], row['dnds_foreground']) 
            else:
                no_difference.append(index)
        if row['dnds_background'] > 1.0:
            print('DNDS',row['query'], row['dnds_background'], row['background_p_value'])
    
    print('fg_acc:',len(higher_fg_rate), 'bg_acc:',len(higher_bg_rate), 'no_rate:',len(no_difference))
    
def plot_evorate_stats(data_frame, output_path):
    # Get the columns for fraction free living aligned and evorate stat
    #fraction_freeliving = data_frame['algn_fraction']
    test_ratio = data_frame['test_ratio']
    #evorate_stats = data_frame[['selected_syn_per_site_avg','selected_ns_per_site_avg', 'branch_fraction']]
    #evorate_stats = data_frame[['branch_fraction', 'branch_fraction_full_norm']]
    
    # Code for exploring outlier values in evorate stats 
    
    '''
    # Outlier count 
    outliers = data_frame[data_frame['branch_fraction'] > 0.1]
    fraction_above_threshold = len(outliers) / len(data_frame)
    print(fraction_above_threshold, outliers.shape[0], data_frame.shape[0])
    
    # Outlier identity
    for index, row in outliers.iterrows():
        branch_fraction = row['branch_fraction']
        algn_fraction = row['algn_fraction']
        print(f"{row['query']}: branch_fraction={branch_fraction}, algn_fraction={algn_fraction}")
    '''
    
    # Filter the data_frame based on branch fraction
    #filtered_data = data_frame[data_frame['branch_fraction'] > 0]

    # Create a multipanel scatter plot
    #fig, axes = plt.subplots(nrows=1, ncols=len(evorate_stats.columns), figsize=(15, 5))
    
    
    # coloring by significance based on BUSTED results
    bg_enhanced = []
    fg_enchanced = []
    no_rate_diff = []
    no_rate_diff_no_rate = []
    for index, row in data_frame.iterrows():
        if row['test_shared_p_value'] < 0.05:
            if row['test_p_value'] > row['background_p_value'] and row['background_p_value'] < 0.05:
                bg_enhanced.append(index)
            elif row['test_p_value'] < row['background_p_value'] and row['test_p_value'] < 0.05:
                fg_enchanced.append(index)
            else:
                no_rate_diff_no_rate.append(index)
        else:
            if row['test_p_value'] < 0.05 and row['background_p_value'] < 0.05:
                no_rate_diff.append(index)
            else:
                no_rate_diff_no_rate.append(index)
           
       
       # Create a density plot
    plt.figure(figsize=(10, 6))
    
    # Plot histograms for algn_fraction
    sns.histplot(data_frame.loc[bg_enhanced, 'test_ratio'], color='red', label='background', kde=False, stat='count', bins=30, multiple="stack", zorder=3)
    sns.histplot(data_frame.loc[fg_enchanced, 'test_ratio'], color='blue', label='foreground', kde=False, stat='count', bins=30, multiple="stack", zorder=2)
    sns.histplot(data_frame.loc[no_rate_diff, 'test_ratio'], color='yellow', label='both', kde=False, stat='count', bins=30, multiple="stack", zorder=4)
    sns.histplot(data_frame.loc[no_rate_diff_no_rate, 'test_ratio'], color='grey', label='neither', kde=False, stat='count', bins=30, multiple="stack", zorder=1)

    plt.xlabel('Test branch fraction')
    plt.ylabel('Count')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    
    ''' 
    # Iterate over each evorate stat column
    for i, column in enumerate(evorate_stats.columns[:2]):
        ax = axes[i]
        # Plot untested points first 
        ax.scatter(fraction_freeliving[no_rate_diff_no_rate], evorate_stats[column][no_rate_diff_no_rate], c='grey', zorder=1)
        # Plot red points next 
        ax.scatter(fraction_freeliving[bg_enhanced], evorate_stats[column][bg_enhanced], c='red', zorder=3)
        # Plot blue points on top
        ax.scatter(fraction_freeliving[fg_enchanced], evorate_stats[column][fg_enchanced], c='blue', zorder=2)
        
        ax.set_xlabel('Fraction Freeliving')
        ax.set_ylabel(' '.join([word.capitalize() for word in column.split('_')]))
    '''
    
    # Save the plot to the output path
   # plt.savefig(output_path, dpi=600)

def plot_evorate_dnds(data_frame, output_path):
    
    # Create a box plot
    labels = ['symbiont nodes', 'non-symbiont nodes']
    filtered_fg = [dnds_foreground for dnds_foreground, test_p_value in zip(data_frame['dnds_foreground'], data_frame['test_p_value']) if test_p_value < 0.05]
    filtered_bg = [dnds_background for dnds_background, background_p_value in zip(data_frame['dnds_background'], data_frame['background_p_value']) if background_p_value < 0.05]
    data_to_plot = [filtered_fg, filtered_bg]
    
    plt.boxplot(data_to_plot, labels=labels, showfliers=False)
    plt.title('dN/dS values for foreground and background nodes in candidate trees')
    plt.ylabel('dn/ds values')
    plt.xlabel('group')
 
    plt.savefig(output_path)
    
def plot_evorate_stats_comp(data_frame, output_path):
    # Get the columns for fraction free living aligned and evorate stat
    fraction_freeliving = data_frame['branch_fraction']
    #evorate_stats = data_frame[['selected_syn_per_site_avg','selected_ns_per_site_avg', 'branch_fraction']]
    evorate_stats = data_frame[['avg_branch_length','total_branch_length']]
    
    # Filter the data_frame based on branch fraction
    filtered_data = data_frame[data_frame['branch_fraction'] > 0]

    # Create a multipanel scatter plot
    fig, axes = plt.subplots(nrows=1, ncols=len(evorate_stats.columns), figsize=(15, 5))
    
    # Iterate over each evorate stat column
    for i, column in enumerate(evorate_stats.columns):
        ax = axes[i]
        ax.scatter(fraction_freeliving, evorate_stats[column])
        ax.set_xlabel('branch_fraction')
        ax.set_ylabel(column)
    
    # Save the plot to the output path
    plt.savefig(output_path)

def plot_aBSREL_comparisons_candidates(data_frame, output_path, title):

    # Filter the data_frame based on algn_fraction
    no_outlier_data = data_frame[(data_frame['symbiont_branch_dnds_avg'] < 20) & (data_frame['non_symbiont_branch_dnds_avg'] < 20)]
    outliers = data_frame[(data_frame['symbiont_branch_dnds_avg'] > 20) | (data_frame['non_symbiont_branch_dnds_avg'] > 20)]
    num_rows_excluded = len(outliers)
    print("Number of rows excluded:", num_rows_excluded)
    print("High symbiont dn/ds:")
    ids = []
    for r in no_outlier_data[no_outlier_data['symbiont_branch_dnds_avg'] > 1].iterrows():
        print(r)
        ids.append(r[1]['query'])
    print(ids)
        
    
    filtered_data_low = no_outlier_data[no_outlier_data['algn_fraction'] <= 0.5]
    filtered_data_high = no_outlier_data[no_outlier_data['algn_fraction'] > 0.5]
    
    # Create scatter plots
    plt.figure(figsize=(10, 6))
    
    plt.scatter(filtered_data_low['symbiont_branch_dnds_avg'], filtered_data_low['non_symbiont_branch_dnds_avg'], color='red', label='algn_fraction <= 0.5', zorder=2)
    plt.scatter(filtered_data_high['symbiont_branch_dnds_avg'], filtered_data_high['non_symbiont_branch_dnds_avg'], color='blue', label='algn_fraction > 0.5', zorder=1)
    plt.xlabel('Symbiont Branch dN/dS Average')
    plt.ylabel('Non-Symbiont Branch dN/dS Average')
    plt.title(title)
    plt.legend()
    
    plt.savefig(output_path)
    
def plot_aBSREL_comparisons_noncandidates(data_frame, output_path):

    # Filter the data_frame based on algn_fraction
    no_outlier_data = data_frame[(data_frame['symbiont_branch_dnds_avg'] < 10) & (data_frame['non_symbiont_branch_dnds_avg'] < 10)]
    outliers = data_frame[(data_frame['symbiont_branch_dnds_avg'] > 10) | (data_frame['non_symbiont_branch_dnds_avg'] > 10)]
    num_rows_excluded = len(outliers)
    print("Number of rows excluded:", num_rows_excluded)
    print(outliers['symbiont_branch_dnds_avg'])
    
    # Create scatter plots
    plt.figure(figsize=(10, 6))
    
    plt.scatter(no_outlier_data['symbiont_branch_dnds_avg'], no_outlier_data['non_symbiont_branch_dnds_avg'])


    plt.xlabel('Symbiont Branch dN/dS Average')
    plt.ylabel('Non-Symbiont Branch dN/dS Average')
    plt.title('Symbiont vs Non-Symbiont Branch dN/dS Averages')
    
    plt.savefig(output_path)

def rate_distribution_comparison(data_frame1, data_frame2, output_path):
    
    '''
    Compares the distribution of dn/ds rates between two data frames 
    '''
    # Perform Mann-Whitney U test
    # Remove NaN values from the distributions
    data_frame1 = data_frame1.dropna(subset=['symbiont_branch_dnds_avg'])
    data_frame2 = data_frame2.dropna(subset=['symbiont_branch_dnds_avg'])
    no_outlier_data1 = data_frame1[(data_frame1['symbiont_branch_dnds_avg'] < 20)]
    no_outlier_data2 = data_frame2[(data_frame2['symbiont_branch_dnds_avg'] < 20)]
    statistic, p_value = mannwhitneyu(no_outlier_data1['symbiont_branch_dnds_avg'], no_outlier_data2['symbiont_branch_dnds_avg'])

    print(data_frame1['symbiont_branch_dnds_avg'])
    print(data_frame2['symbiont_branch_dnds_avg'])
    input()
    # Print the results
    print("Mann-Whitney U test results:")
    print("Statistic:", statistic)
    print("P-value:", p_value)
    
def relative_rate_comparison(data_frame1, data_frame2, output_path):
    
    '''
    Determines the relative rate differences between symbiont and non-symbiont branches within a dataframe
    using a PERMANOVA test
    '''
    print("Size of data_frame1:", data_frame1.shape)
    print("Size of data_frame2:", data_frame2.shape)
    print("Number of non-NA values in data_frame1:")
    print(data_frame1.notna().sum())

    print("Number of non-NA values in data_frame2:")
    print(data_frame2.notna().sum())
    
    data_frame1['Group'] = 0
    data_frame2['Group'] = 1
    
    data_frame1 = data_frame1.reset_index(drop=True)
    data_frame2 = data_frame2.reset_index(drop=True)
    
    combined_data = pd.concat([data_frame1, data_frame2], ignore_index=True)    
    
    combined_data.index = combined_data.index.astype(str)
    
    print(combined_data.shape)
    
     # Ensure the columns are numeric
    combined_data['symbiont_branch_dnds_avg'] = pd.to_numeric(combined_data['symbiont_branch_dnds_avg'], errors='coerce')
    combined_data['non_symbiont_branch_dnds_avg'] = pd.to_numeric(combined_data['non_symbiont_branch_dnds_avg'], errors='coerce')
    
    # Drop rows with missing values
    combined_data = combined_data.dropna(subset=['symbiont_branch_dnds_avg', 'non_symbiont_branch_dnds_avg'])
    
    print(combined_data.shape)
    # Check for duplicates
    duplicates = combined_data.index[combined_data.index.duplicated()]
    if not duplicates.empty:
        print(f"Duplicate labels found: {duplicates}")
    
    distance_matrix = DistanceMatrix(squareform(pdist(combined_data[['symbiont_branch_dnds_avg', 'non_symbiont_branch_dnds_avg']], metric='euclidean')), ids=combined_data.index.astype(str))
    
    
    results = permanova(distance_matrix, combined_data['Group'].astype(str), permutations=999)
    print(results)
    
def main():

    '''Script that determines the overall presence of wMel protein alignments with the free-living control dataset.

       alignment files should be generated with foldseek using:
        --format-output query,target,evalue,alntmscore,alnlen,qlen,tcov,qcov,tlen,u,t,lddt,fident,pident,prob
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-a','--alignment', type=str, help='path to a foldseek alignment result file')
    parser.add_argument('-c','--controls', type=str, help='paths to a directory of control alignments')
    parser.add_argument('-f','--fid_plot', type=str, help='output path for png of fraction of identical residues histogram')
    parser.add_argument('-j','--json_file', type=str, help='path to a pre-generated json of control alignment stats')
    parser.add_argument('-o','--csv_out',  type=str, help='output csv table of results to provided path')
    parser.add_argument('-d','--pdb_database', type=str, help='path to database of pdb files used in alignment')
    parser.add_argument('-v','--validation_ids', type=str, help='path to a txt file of structure IDs to pull from results')
    parser.add_argument('-e','--evorate_analysis', type=str, help='path to evorateworkflow results directory')
    parser.add_argument('-e2','--evorate_analysis2', type=str, help='path to second evorate results directory for distrubtion comparison')
    parser.add_argument('-p','--plot_evorate',  type=str, help='path to evorateworkflow results plot')
    parser.add_argument('-i','--id_map', type=str, help='path to id mapping file from uniprot')
    parser.add_argument('-s','--symbiont_ids', type=str, help='path to a txt file of symbiont IDs to pull from results')
    args = parser.parse_args()   

    # either generate the control dictionary or load it from previous run 
    if not args.json_file:
        control_dictionary = generate_control_dictionary(args.controls)
    else:
        with open(args.json_file, 'r') as json_f:
            control_dictionary = json.load(json_f)

    # generate alignment stats data table and then clean up the ids
    alignment_table = alignment_stats(args.alignment, control_dictionary)
    alignment_df = make_output_df(alignment_table)
    
    if args.evorate_analysis:
        
        absrel_df = parse_hyphy_output.parse_absrel_results(args.evorate_analysis, args.symbiont_ids, args.id_map)
        
        if args.evorate_analysis2:
            
            # statistical comparison of dn/ds rates between two evorate analyses
            absrel_df2 = parse_hyphy_output.parse_absrel_results(args.evorate_analysis2, args.symbiont_ids, args.id_map)
            rate_distribution_comparison(absrel_df, absrel_df2, args.plot_evorate)
            relative_rate_comparison(absrel_df, absrel_df2, args.plot_evorate)
            #alignment_stat_tests.LDA_QDA_analysis(absrel_df, absrel_df2)

        # aBSREL dnds comparison non candidates
        
        plot_aBSREL_comparisons_noncandidates(absrel_df, args.plot_evorate)   
        
        
        # aBSREL dnds comparison candidates
        print(absrel_df['symbiont_branch_dnds_avg'])
        
        evorate_alignment_df = pd.merge(alignment_df, absrel_df, on='query', how='left')
        
        
        #plot_aBSREL_comparisons_candidates(evorate_alignment_df, args.plot_evorate, 'Symbiont vs Non-Symbiont Branch dN/dS Averages wMel, small taxa range')  
    
        
    # save dataframe to csv 
    
    alignment_df.to_csv(args.csv_out, index=False)
        
    # plot histogram of freeliving fraction values for alignment
    if args.fid_plot:
        plot_freeliving_fraction_distribution(data_table, args.fid_plot)
    
    # parse list of validation or general IDs of interest to pull from the results of an alignment
    if args.validation_ids:
        ids_of_interest = set()
        with open(args.validation_ids, 'r') as txt_f:
            ids = txt_f.readlines()
            for struct_id in ids:
                # add structure IDs to id set 
                ids_of_interest.add(struct_id.strip())

        # find ids and print to std_out in csv format
        validation(alignment_df, ids_of_interest, args.pdb_database)

if __name__ == '__main__':
    main()