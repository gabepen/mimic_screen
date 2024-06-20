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

def plot_evorate_stats(data_frame, output_path):
    # Get the columns for fraction free living aligned and evorate stat
    fraction_freeliving = data_frame['algn_fraction']
    #evorate_stats = data_frame[['selected_syn_per_site_avg','selected_ns_per_site_avg', 'branch_fraction']]
    evorate_stats = data_frame[['branch_fraction', 'branch_fraction_full_norm']]
    
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
    filtered_data = data_frame[data_frame['branch_fraction'] > 0]

    # Create a multipanel scatter plot
    fig, axes = plt.subplots(nrows=1, ncols=len(evorate_stats.columns), figsize=(15, 5))
    
    # Iterate over each evorate stat column
    for i, column in enumerate(evorate_stats.columns):
        ax = axes[i]
        ax.scatter(fraction_freeliving, evorate_stats[column])
        ax.set_xlabel('Fraction Freeliving')
        ax.set_ylabel(' '.join([word.capitalize() for word in column.split('_')]))
    
    # Save the plot to the output path
    plt.savefig(output_path, dpi=600)

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
    parser.add_argument('-p','--plot_evorate',  type=str, help='path to evorateworkflow results plot')
    parser.add_argument('-i','--id_map', type=str, help='path to id mapping file from uniprot')
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
    
    # parse evorate analysis and add to current datatable
    if args.evorate_analysis:
        if not args.id_map:
            print('Please provide an id mapping file to parse evorate results')
            sys.exit(1)
        evorate_df = parse_hyphy_output.parse_absrel_results(args.evorate_analysis, args.id_map)
        
        # merge all evorate stats into alignment_df for plotting
        if args.plot_evorate:
            evorate_alignment_df = pd.merge(alignment_df, evorate_df, on='query', how='left')
            plot_evorate_stats(evorate_alignment_df, args.plot_evorate)
        
        # merge branch fraction into alignment_df for results table
        results_alignment_df = pd.merge(alignment_df, evorate_df[['query', 'branch_fraction']], on='query', how='left')
    # save dataframe to csv 
    results_alignment_df.to_csv(args.csv_out, index=False)
        
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