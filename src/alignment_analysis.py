import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pdb
import re
import seaborn as sns
import subprocess
import sys
from Bio.PDB import PDBParser
from glob import glob
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mannwhitneyu
from skbio.stats.distance import DistanceMatrix, permanova
from statistics import mean
from tqdm import tqdm
from utilities import parse_hyphy_output, uniprot_api_queries
from stats_modules import PCA_tests

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
            qlen = lp[5]
            tlen = lp[8]
            
            # alignment coverage windows 
            qstart = float(lp[15])
            qend = float(lp[16])
            tstart = float(lp[17])
            tend = float(lp[18])
            algn_stretch = (qstart, qend, tstart, tend)
                
            # alignment base statistic requirements
            if float(score) > 0.4 and  evalue < 0.01 and (tcov > 0.25 or qcov >= 0.5):
                
                # check presence in control proteomes
                if query in control_dict:
                    data_table.append([query, float(score), tcov, qcov, fident, control_dict[query]['algn_fraction'], target, algn_stretch, qlen, tlen, 'candidate'])
                else:
                    data_table.append([query, float(score), tcov, qcov, fident,0.0, target, algn_stretch, qlen, tlen, 'candidate'])      

            else:
                try:
                    data_table.append([query, float(score), tcov, qcov, fident, control_dict[query]['algn_fraction'], target, algn_stretch, qlen, tlen, 'filtered'])
                except KeyError:
                    data_table.append([query, float(score), tcov, qcov, fident,0.0, target, algn_stretch, qlen, tlen, 'filtered']) 

    
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

                row_string = row[['query', 'target', 'score', 'tcov', 'qcov', 'fident', 'algn_fraction', 'symbiont_branch_dnds_avg', 'non_symbiont_branch_dnds_avg']].astype(str).str.cat(sep=',') + ',' + str(average_pLDDTs[uni_id])
                print(row_string) 
                
def calculate_alignment_overlap(aln_span1, aln_span2):  
    
    overlap_start = max(aln_span1[2], aln_span2[2])
    overlap_end = min(aln_span1[3], aln_span2[3]) 
    overlap_len = max(0, overlap_end - overlap_start + 1)
    
    union_start = min(aln_span1[2], aln_span2[2])
    union_end = max(aln_span1[3], aln_span2[3])
    union_len = union_end - union_start + 1
    
    overlap_pct = (overlap_len / union_len)  
    return overlap_pct      

def create_host_self_alignment_table(alignment):
    
    # create a dictionary of host protein self alignments
    host_self_alignment_table = {}
    
    #parse host self alignment and create look up table
    with open(alignment, 'r') as tsv:
        lines = tsv.readlines()
        for l in lines:
            
            # collect foldseek output values
            parts = l.strip().split()
            score = float(parts[3])
            evalue = float(parts[2])
            tcov = float(parts[6])
            qcov = float(parts[7])
            fident = float(parts[12])
            
            # alignment base statistic requirements
            if score > 0.4 and  evalue < 0.01 and (tcov > 0.25 or qcov >= 0.5):
                
                # clean up file name to just have UNIProt ID
                start = 'AF-'
                end = '-F'
                target_id = re.search(f'{start}(.*?){end}', parts[1]).group(1) if re.search(f'{start}(.*?){end}', parts[1]) else parts[1]
                query_id = re.search(f'{start}(.*?){end}', parts[0]).group(1) if re.search(f'{start}(.*?){end}', parts[0]) else parts[0]
                
                # store the two aligned proteins and the tm-score in lookup table 
                if query_id in host_self_alignment_table:
                    host_self_alignment_table[query_id].append([target_id, score, fident])
                else:
                    host_self_alignment_table[query_id] = [[target_id, score, fident]]
    
    return host_self_alignment_table
                     
def plot_paralog_stat_distributions(structure_scores, sequence_scores, cohen_d_values, output_path):
    
    structure_scores = [score for score in structure_scores if score is not None]
    sequence_scores = [score for score in sequence_scores if score is not None]
   
    plt.figure(figsize=(10, 6))
    
    sns.histplot(structure_scores, kde=True, color='blue', label='Structure Scores', bins=30)
    sns.histplot(sequence_scores, kde=True, color='red', label='Sequence Scores', bins=30)
    
    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.title('Distribution of Structure and Sequence Average Scores Per Target Group')
    plt.legend()
    
    plt.savefig(output_path)
    plt.close()
    
def plot_paralog_rank_comparison(ranking_averages, output_path):
    
    # Extract the first and second values from the tuples
    first_values = [x[0] for x in ranking_averages]
    second_values = [x[1] for x in ranking_averages]

    # Determine the ranks of the first and second values
    first_ranks = pd.Series(first_values).rank().tolist()
    second_ranks = pd.Series(second_values).rank().tolist()

    # Create a scatter plot of the ranks
    plt.figure(figsize=(10, 6))
    plt.scatter(first_ranks, second_ranks, color='blue')

    plt.xlabel('Rank of TM-Score Value')
    plt.ylabel('Rank of SeqID Value')
    plt.title('Scatter Plot of Ranks')

    plt.savefig(output_path)
    plt.close()
    
def paralog_filter(target_group, host_self_alignment_table, query_tm_scores, query_seqid_scores):
    
    def cohen_d(data1, data2):
        """
        Calculates Cohen's d effect size for two samples.

        Args:
            data1: The first dataset (list or numpy array).
            data2: The second dataset (list or numpy array).

        Returns:
            Cohen's d effect size.
        """

        n1 = len(data1)
        n2 = len(data2)
        mean1 = np.mean(data1)
        mean2 = np.mean(data2)
        std1 = np.std(data1, ddof=1)  # Sample standard deviation
        std2 = np.std(data2, ddof=1)  # Sample standard deviation

        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        d = (mean1 - mean2) / pooled_std

        return d

    # iterate over target group and collect all alignment scores between each target in group
    alignment_scores = []
    sequence_identity_scores = []
    for target in target_group:
        for alignment in host_self_alignment_table[target]:
            if alignment[0] in target_group and alignment[0] != target:
                alignment_scores.append(alignment[1])
                sequence_identity_scores.append(alignment[2])
    

    '''
    # calculate the average alignment score between all targets in the group
    try:
        avg_alignment_score = sum(alignment_scores) / len(alignment_scores)
        avg_query_score = sum(query_tm_scores) / len(query_tm_scores)
    # the similarity between the host proteins are too low to have been aligned
    except ZeroDivisionError:
        return None
    '''
    
    
    # calculate the average seqid between all targets in the group
    try:
        avg_seqid_score = sum(sequence_identity_scores) / len(sequence_identity_scores)
        avg_alignment_score = sum(alignment_scores) / len(alignment_scores)
        avg_query_score = sum(query_seqid_scores) / len(query_seqid_scores)
    # the similarity between the host proteins are too low to have been aligned
    except ZeroDivisionError:
        return False, None, None
    
    '''
    # determine if the divergence in structure between the host proteins and the query is within the threshold
    if abs(avg_seqid_score - avg_query_score) < threshold:
        return True, avg_seqid_score, avg_alignment_score
    else:
        return False, avg_seqid_score, avg_alignment_score
    '''
    
    # statistical and magnitude test approach 
    
    #U_statistic, p_value = mannwhitneyu(alignment_scores, query_tm_scores)
    #cohen_d = cohen_d(alignment_scores, query_tm_scores)

    U_statistic, p_value = mannwhitneyu(sequence_identity_scores, query_seqid_scores)
    #cohen_d = cohen_d(sequence_identity_scores, query_seqid_scores)
    
    #cohen_d = abs(cohen_d)
    if p_value < 0.05:
        return False, avg_seqid_score, avg_alignment_score
    else:
        return True, avg_seqid_score, avg_alignment_score

def make_output_df_no_filters(data_table):
    
    output_table = []
    # get results 
    pairs = set()
    query_ids = set()
    host_ids = set()
    query_table = {}
    target_table = {}
    for row in tqdm(data_table):

        if row[-1] == 'filtered':
            continue
        
        # clean up file name to just have UNIProt ID
        start = 'AF-'
        end = '-F'
        target_id = re.search(f'{start}(.*?){end}', row[6]).group(1) if re.search(f'{start}(.*?){end}', row[6]) else row[6]
        query_id = re.search(f'{start}(.*?){end}', row[0]).group(1) if re.search(f'{start}(.*?){end}', row[0]) else row[0].split('_')[0]
        
        alignment_pair  = (query_id, target_id) 
        
        alignment_stats = [row[1], row[2], row[3], row[4], row[5], row[8], row[9], row[7]]
        output_fields = alignment_stats[0:7]
        output_fields.extend(alignment_stats[7])
        output_fields.insert(0, target_id)
        output_fields.insert(0, query_id)
        output_table.append(output_fields)

        
    df = pd.DataFrame(output_table, columns=['query', 'target', 'score', 'tcov', 'qcov', 'fident', 'algn_fraction', 'qlen', 'tlen', 'qstart', 'qend', 'tstart', 'tend'])

    return df
    
def make_output_df(data_table, host_self_alignment_table, top_hits_only, alignment_span_filter):
    
    '''cleans up the results table and returns as a pandas dataframe
    '''
    
    output_table = []
    # get results 
    pairs = set()
    query_ids = set()
    host_ids = set()
    query_table = {}
    target_table = {}
    for row in tqdm(data_table):

        if row[-1] == 'filtered':
            continue
        
        # clean up file name to just have UNIProt ID
        start = 'AF-'
        end = '-F'
        target_id = re.search(f'{start}(.*?){end}', row[6]).group(1) if re.search(f'{start}(.*?){end}', row[6]) else row[6]
        query_id = re.search(f'{start}(.*?){end}', row[0]).group(1) if re.search(f'{start}(.*?){end}', row[0]) else row[0].split('_')[0]
        
        alignment_pair  = (query_id, target_id) 
        
        ''' # determine top alignment for query id
        if top_hits_only and query_id in query_table:
            if row[1] + row[2] > query_table[query_id][2] + query_table[query_id][3]:
                query_table[query_id] = [target_id, row[1], row[2], row[3], row[4], row[5], row[8], row[9], row[7]]
        else:
        '''
        # add new query id to table
        if query_id not in query_table:
            query_table[query_id] = {target_id: [row[1], row[2], row[3], row[4], row[5], row[8], row[9], row[7]]}
        else:
            # query exists in table add new target id
            if target_id not in query_table[query_id]:
                query_table[query_id][target_id] = [row[1], row[2], row[3], row[4], row[5], row[8], row[9], row[7]]
            else:
                # query has been aligned to target at different positions, compare scores
                if row[1] > query_table[query_id][target_id][0]:
                    query_table[query_id][target_id] = [row[1], row[2], row[3], row[4], row[5], row[8], row[9], row[7]]
                else:
                    continue
                
        
        # track all alignments for target id
        if target_id in target_table:
            target_table[target_id].append([query_id, row[1], row[2], row[3], row[4], row[5], row[8], row[9], row[7]])
        else:
            target_table[target_id] = [[query_id, row[1], row[2], row[3], row[4], row[5], row[8], row[9], row[7]]]

    # iterate over all alignment pairs and add to output dataframe by each query ID then each target it aligns to 
    seen_target_ids = set()
    paralog_results = {}
    alternate_hits = {}
    for query_id in list(query_table.keys()):
        
        # check number of targets query aligns to 
        targets = list(query_table[query_id].keys())
        
        # this could indicate a group of host protein paralogs
        if len(targets) > 1 and host_self_alignment_table != None:
            
            #  get tm-score across all alignments for query id
            query_tm_scores = [query_table[query_id][target][1] for target in targets]
            query_seqid_scores = [query_table[query_id][target][4] for target in targets]
            
            # determine if the divergence in structure between the host proteins 
            # is the same as the divergence between the query and host proteins
            # (i.e. very likely houskeeping genes)
            result, seq_id_average, tm_score_avg = paralog_filter(targets, host_self_alignment_table, query_tm_scores, query_seqid_scores)

            if result:
                paralog_results[query_id] = 'yes'
            else:
                paralog_results[query_id] = 'no'
        
        # for top_hits_only output determine the best alignment for each query id
        if top_hits_only:
            
            # Rank the values in the dictionary for the query id based on the sum of the first two values in each sub dictionary
            ranked_targets = sorted(query_table[query_id].items(), key=lambda x: x[1][0] + x[1][1], reverse=True)

            # Update the query table with the best alignment for the query id
            query_table[query_id] = {ranked_targets[0][0]: ranked_targets[0][1]} 
            
            # store the alternate hits for the query id 
            alternate_hits[query_id] = [target[0] for target in ranked_targets[1:]]
                 
        # iterate over all alignments for query id
        for target_id in list(query_table[query_id].keys()) :
            
            # if the target id mapped to multiple queries determine percent overlaps of those alignments 
            if target_id not in seen_target_ids and len(target_table[target_id]) > 1 and not top_hits_only and alignment_span_filter:
                overlap_matrix = []
                for i in range(len(target_table[target_id])):
                    row = []
                    for j in range(len(target_table[target_id])):
                        aln_span1, aln_span2 = target_table[target_id][i][8], target_table[target_id][j][8] 
                        overlap = calculate_alignment_overlap(aln_span1, aln_span2)
                        row.append(overlap)
                    overlap_matrix.append(row)
                        
                # convert to distance matrix 
                distance_matrix = 1 - np.array(overlap_matrix)
                
                # hierachical clustering
                linked = linkage(distance_matrix, 'single')

                similarity_threshold = 0.75
                labels = fcluster(linked, similarity_threshold, criterion='distance')
                
                # Convert labels to standard ints
                labels = [label.item() for label in labels]

                # Move alignment groups into dictionary 
                grouped_alignments = {}
                for i, label in enumerate(labels):
                    # add alignments as tuple pairs of query ID target aligned to and TM-Score
                    if label not in grouped_alignments:
                        grouped_alignments[label] = [(target_table[target_id][i][0], target_table[target_id][i][1])]
                    else:
                        grouped_alignments[label].append((target_table[target_id][i][0], target_table[target_id][i][1]))
                
                # select the alignment with the highest TM-score from each group
                best_alignments = []
                for group in grouped_alignments:
                    
                    largest_value = max(grouped_alignments[group], key=lambda x: x[1])
                    best_alignments = [v for v in grouped_alignments[group] if abs(v[1] - largest_value[1]) <= 0.003]
                    
                    # remove other alignments in group from overall output
                    for alignment in grouped_alignments[group]:
                        if alignment not in best_alignments:
                            try:
                                del query_table[alignment[0]][target_id]
                            except KeyError:
                                continue
                
                seen_target_ids.add(target_id)
                
            try:
                output_fields = query_table[query_id][target_id][0:7]
                output_fields.extend(query_table[query_id][target_id][7])
                output_fields.insert(0, target_id)
                output_fields.insert(0, query_id)
                output_table.append(output_fields)
            # this means the alignemnt pair was not the best in the group and no longer exists for the query id
            except KeyError:
                continue
                
        
    '''if alignment_pair not in pairs:
        # format output order
        output = [query_id, target_id, row[1], row[2], row[3], row[4], row[5]]
        output_table.append(output)
        pairs.add(alignment_pair)
        ids.add(query_id)'''

    # Convert data_table to pandas DataFrame
    df = pd.DataFrame(output_table, columns=['query', 'target', 'score', 'tcov', 'qcov', 'fident', 'algn_fraction', 'qlen', 'tlen', 'qstart', 'qend', 'tstart', 'tend'])
    
    # Map on paralog check results 
    df['targets_likely_paralogs'] = df['query'].map(paralog_results)

    if top_hits_only:
        # Map on alternate hits results 
        df['alternate_hits'] = df['query'].map(alternate_hits)

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
    plt.close()
    
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
    plt.close()
    
def plot_test_fraction(data_frame, output_path, outlier_cutoff=10):
    
    # Filter the data_frame based on algn_fraction
    no_outlier_data = data_frame[(data_frame['symbiont_branch_dnds_avg'] < outlier_cutoff) & (data_frame['non_symbiont_branch_dnds_avg'] < outlier_cutoff)]
    outliers = data_frame[(data_frame['symbiont_branch_dnds_avg'] > outlier_cutoff) | (data_frame['non_symbiont_branch_dnds_avg'] > outlier_cutoff)]
    num_rows_excluded = len(outliers)
    
    # Create scatter plots
    plt.figure(figsize=(10, 6))

    plt.scatter(no_outlier_data['algn_fraction'], no_outlier_data['test_fraction'], color='blue', zorder=1)

    plt.xlabel('Alignment Fraction')
    plt.ylabel('Test Fraction')
    plt.title('FFPA vs Test Fraction')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
def plot_aBSREL_comparisons_candidates(data_frame, output_path, outlier_cutoff, title):

    # Filter the data_frame based on algn_fraction
    no_outlier_data = data_frame[(data_frame['symbiont_branch_dnds_avg'] < outlier_cutoff) & (data_frame['non_symbiont_branch_dnds_avg'] < outlier_cutoff)]
    outliers = data_frame[(data_frame['symbiont_branch_dnds_avg'] > outlier_cutoff) | (data_frame['non_symbiont_branch_dnds_avg'] > outlier_cutoff)]
    num_rows_excluded = len(outliers)
    '''
    print("Number of rows excluded:", num_rows_excluded)
    print("High symbiont dn/ds:")
    ids = []
    for r in no_outlier_data[no_outlier_data['symbiont_branch_dnds_avg'] > 1].iterrows():
        print(r)
        ids.append(r[1]['query'])
    print("High non-symbiont dn/ds:")
    for r in no_outlier_data[(no_outlier_data['non_symbiont_branch_dnds_avg'] > 1) & (no_outlier_data['algn_fraction'] < 0.5)].iterrows():
        print(r)
        ids.append(r[1]['query'])
    print(ids)
    '''
    
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
    plt.close()
    
def plot_aBSREL_comparisons_noncandidates(data_frame, output_path, outlier_cutoff, title):

    # Filter the data_frame based on algn_fraction
    no_outlier_data = data_frame[(data_frame['symbiont_branch_dnds_avg'] < outlier_cutoff) & (data_frame['non_symbiont_branch_dnds_avg'] < outlier_cutoff)]
    outliers = data_frame[(data_frame['symbiont_branch_dnds_avg'] > outlier_cutoff) | (data_frame['non_symbiont_branch_dnds_avg'] > outlier_cutoff)]
    num_rows_excluded = len(outliers)
    print("Number of rows excluded:", num_rows_excluded)
    print(outliers['symbiont_branch_dnds_avg'])
    
    # Create scatter plots
    plt.figure(figsize=(10, 6))
    
    plt.scatter(no_outlier_data['symbiont_branch_dnds_avg'], no_outlier_data['non_symbiont_branch_dnds_avg'])


    plt.xlabel('Symbiont Branch dN/dS Average')
    plt.ylabel('Non-Symbiont Branch dN/dS Average')
    plt.title(title)
    
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
    # Calculate and print the average of each data frame
    avg_df1 = no_outlier_data1['symbiont_branch_dnds_avg'].mean()
    avg_df2 = no_outlier_data2['symbiont_branch_dnds_avg'].mean()
    print("Average of symbiont_branch_dnds_avg in data_frame1:", avg_df1)
    print("Average of symbiont_branch_dnds_avg in data_frame2:", avg_df2)
    
    avg_df1 = no_outlier_data1['non_symbiont_branch_dnds_avg'].mean()
    avg_df2 = no_outlier_data2['non_symbiont_branch_dnds_avg'].mean()
    print("Average of non_symbiont_branch_dnds_avg in data_frame1:", avg_df1)
    print("Average of non_symbiont_branch_dnds_avg in data_frame2:", avg_df2)

    # Calculate and print the number of values above 1 in each data frame
    count_above_1_df1 = (no_outlier_data1['symbiont_branch_dnds_avg'] > 1).sum()
    count_above_1_df2 = (no_outlier_data2['symbiont_branch_dnds_avg'] > 1).sum()
    print("Number of values above 1 in data_frame1:", count_above_1_df1)
    print("Number of values above 1 in data_frame2:", count_above_1_df2)
    
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
    
def add_protein_description(data_frame, field_name):
    
    '''
    Adds a new column 'description' to the data frame by querying the UniProt API for each query ID.
    '''
    
    # Check for id dict for previous runs 
    if not os.path.exists('prot_descriptions.json'):
        id_descriptions = {}
    else:
        with open('prot_descriptions.json', 'r') as json_f:
            id_descriptions = json.load(json_f)
                    
    descriptions = []
    for query_id in data_frame[field_name]:
        if query_id in id_descriptions:
            descriptions.append(id_descriptions[query_id])
            continue
        entry = uniprot_api_queries.fetch_uniprot_entry(query_id)
        try:
            description = entry['proteinDescription']['recommendedName']['fullName']['value']
        except KeyError:
            try:
                description = entry['proteinDescription']['submissionNames'][0]['fullName']['value']
            except KeyError:
                description = 'NA'
        descriptions.append(description)
        id_descriptions[query_id] = description
    
    output_field = field_name + '_description'
    data_frame[output_field] = descriptions
    
    # save id descriptions to json
    with open('prot_descriptions.json', 'w') as json_f:
        json.dump(id_descriptions, json_f, indent=2)
        
    return data_frame

def add_go_terms(data_frame, field_name):  
    
    # Check for id dict for previous runs 
    if not os.path.exists('prot_descriptions.json'):
        id_descriptions = {}
    else:
        with open('prot_descriptions.json', 'r') as json_f:
            id_descriptions = json.load(json_f)
    
    # storing GO terms in lists for adding as columns to data frame
    cell_comps = []
    mol_func = []
    bio_proc = []
    
    # iterate through query ids and fetch GO terms from UniProt API
    for query_id in data_frame[field_name]:
        search_id = query_id
        query_id = query_id + '_GO'
        
        # check if query id has been queried before
        if query_id in id_descriptions:
            cell_comps.append(id_descriptions[query_id]['C'])
            mol_func.append(id_descriptions[query_id]['F'])
            bio_proc.append(id_descriptions[query_id]['P'])
            continue
        
        # fetch entry from UniProt API
        entry = uniprot_api_queries.fetch_uniprot_entry(search_id)
        try:
            refs = entry['uniProtKBCrossReferences']
        except KeyError:
            print(entry)  
            print(entry.keys()) 
            
        # parse extrernal references for GO terms
        term_dict = {'C': [], 'F': [], 'P': []}
        for ref in refs:
            
            # find GO entries 
            if ref['database'] == 'GO':
                
                # parse GO term and type
                go_id = ref['id']
                go_type = ref['properties'][0]['value'].split(':')[0]
                go_term = ref['properties'][0]['value'].split(':')[1]  
            
                # add GO term to appropriate list
                if go_type in term_dict:
                    term_dict[go_type].append(go_term)
                else:
                    term_dict[go_type] = [go_term]
                    
        # add GO terms to lists for columns 
        cell_comps.append(term_dict['C'])
        mol_func.append(term_dict['F'])
        bio_proc.append(term_dict['P'])
        
        # store terms for look up 
        id_descriptions[query_id] = term_dict
    
    output_field = field_name + '_cellular_components'
    data_frame[output_field] = cell_comps
    output_field = field_name + '_molecular_functions'
    data_frame[output_field] = mol_func
    output_field = field_name + '_biological_processes'
    data_frame[output_field] = bio_proc
    
     # save id GO terms to json
    with open('prot_descriptions.json', 'w') as json_f:
        json.dump(id_descriptions, json_f, indent=2)
        
    return data_frame

def collect_aa_sequences(data_frame, field_name, mt_predictions): 
    
    aa_sequences = []
    collected_ids = set()     
    for index, row in data_frame.iterrows():
        query_id = row[field_name]
        
        # prediction has already been made or id seq has been collected already
        if query_id in mt_predictions or query_id in collected_ids:
            continue
        
        # check if target protein is likely mitochondrial 
        #if 'mitoc' in row['target_description'] or 'mitoc' in ' '.join(row['target_cellular_components']).lower():
            
        # fetch entry from UniProt API
        entry = uniprot_api_queries.fetch_uniprot_entry(query_id)
        
        # collect amino acid sequence for uniprot entry
        try:
            aa_sequence = entry['sequence']['value']
        except KeyError:  
            aa_sequence = 'X'
        
        aa_sequences.append({'id': query_id, 'sequence': aa_sequence})
        
        collected_ids.add(query_id)
        
            
    return aa_sequences
    
def write_aa_sequences_to_fasta(aa_seq_list, output_path):
    
    with open(output_path, 'w+') as fasta_f:
        for seq in aa_seq_list:
            fasta_f.write(f'>{seq["id"]}\n{seq["sequence"]}\n')
    
    return output_path

def predict_mt_sequences(data_frame, fasta_path, mt_predictions_dict, field_name):
    
    # run targetp on fasta file
    results = subprocess.run(['targetp', '-fasta', fasta_path, '-format', 'short', '-stdout'], capture_output=True, text=True)
    
    probabilities = []
    for l in results.stdout.split('\n'):
        if l.startswith('#'):
            continue
        l = l.split('\t')
        
        try:
            query_id = l[0]
            prediction = l[1]
        except IndexError:
            continue
        
        # store prediction in dictionary
        if query_id not in mt_predictions_dict:
            mt_predictions_dict[query_id] = (l[2], l[3], l[4])
        
    output_field_mTP = field_name + '_mTP_probability'
    output_field_SP = field_name + '_SP_probability'
    data_frame[output_field_mTP] = data_frame[field_name].map(lambda x: mt_predictions_dict[x][2] if x in mt_predictions_dict else None)
    data_frame[output_field_SP] = data_frame[field_name].map(lambda x: mt_predictions_dict[x][1] if x in mt_predictions_dict else None)
    
    # remove tmp fasta file
    os.remove(fasta_path)
    
    # save id predictions to json
    with open('mt_sequence_predictions.json', 'w') as json_f:
        json.dump(mt_predictions_dict, json_f, indent=2)
        
    return data_frame
         
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
    parser.add_argument('-p2','--plot_evorate2',  type=str, help='path to evorateworkflow results plot (non-candidates)')
    parser.add_argument('-i','--id_map', type=str, help='path to id mapping file from uniprot')
    parser.add_argument('-s','--symbiont_ids', type=str, help='path to a txt file of symbiont IDs to pull from results')
    parser.add_argument('-pf','--paralog_filter', type=str, help='path to host self alignment file to filter out paralogs from the results')
    parser.add_argument('-mt','--mt_seq_predict', action='store_true', help='predict mitochondrial sequences')
    parser.add_argument('-t','--top_hits', action='store_true', help='only output top hits for each query id')
    parser.add_argument('-ap','--alignment_span_filter', action='store_true', help='filter out alignments where the target has aligned to multiple queries on the save span of target structure')
    parser.add_argument('-pca','--pca_test_dir', type=str, help='path to directory for running and storing PCA tests')
    args = parser.parse_args()   

    # either generate the control dictionary or load it from previous run 
    if not args.json_file:
        control_dictionary = generate_control_dictionary(args.controls)
    else:
        with open(args.json_file, 'r') as json_f:
            control_dictionary = json.load(json_f)

    # generate alignment stats data table
    alignment_table = alignment_stats(args.alignment, control_dictionary)
    
    # create host self alignment table if using paralog filter 
    if args.paralog_filter:
        
        # check if self alignment has been used before 
        self_alignment_name  = args.paralog_filter.split('/')[-1].split('.')[0]
        if not os.path.exists(f'{self_alignment_name}.json'):
            
            # create host self alignment table
            host_self_alignment_table = create_host_self_alignment_table(args.paralog_filter)
            
            # save host self alignment table to json
            with open(f'{self_alignment_name}.json', 'w') as json_f:
                json.dump(host_self_alignment_table, json_f, indent=2)
        
        else:
            with open(f'{self_alignment_name}.json', 'r') as json_f:
                host_self_alignment_table = json.load(json_f)
        
    else:
        host_self_alignment_table = None
        
    alignment_df = make_output_df(alignment_table, host_self_alignment_table, args.top_hits, args.alignment_span_filter)
    #alignment_df = make_output_df_no_filters(alignment_table)
    output_df = alignment_df.copy()
        
    if args.evorate_analysis:
        
        absrel_df = parse_hyphy_output.parse_absrel_results(args.evorate_analysis, args.symbiont_ids, args.id_map)
        if args.evorate_analysis2:
            
            # statistical comparison of dn/ds rates between two evorate analyses
            absrel_df2 = parse_hyphy_output.parse_absrel_results(args.evorate_analysis2, args.symbiont_ids, args.id_map)
            
        # aBSREL dnds comparison candidates
        evorate_alignment_df = pd.merge(alignment_df, absrel_df, on='query', how='left')
        plot_test_fraction(evorate_alignment_df,'/storage1/gabe/mimic_screen/main_paper/final_figs/evorate_figs/wmel_testfraction-FFPA.png', 10)
        plot_aBSREL_comparisons_candidates(evorate_alignment_df, args.plot_evorate, 10, 'Symbiont vs Non-Symbiont Branch dN/dS Averages wMelCandidates')  
        columns_to_drop = ['ns_per_site_avg', 'syn_per_site_avg', 'dnds_tree_avg', 'symbiont_tree_dnds_avg', 'non_symbiont_tree_dnds_avg', 'selection_branch_count', 'total_branch_length', 'avg_branch_length', 'selected_ns_per_site_avg', 'selected_syn_per_site_avg','branch_fraction','branch_fraction_full_norm']
        output_evorate_alignment_df = evorate_alignment_df.drop(columns=columns_to_drop)
        output_df = output_evorate_alignment_df.copy()
       
    # add protein descriptions to the output dataframe
    output_df = add_protein_description(output_df, 'query')  
    output_df = add_protein_description(output_df, 'target')
    output_df = add_go_terms(output_df, 'target')
    
    # add mitochondrial sequence prediction to the output dataframe
    if args.mt_seq_predict:
        
        # Check for id dict for previous runs 
        if not os.path.exists('mt_sequence_predictions.json'):
            mt_predictions = {}
        else:
            with open('mt_sequence_predictions.json', 'r') as json_f:
                mt_predictions = json.load(json_f)
       
        aa_sequences = collect_aa_sequences(output_df, 'query', mt_predictions)
        tmp_aa_seq_fasta = write_aa_sequences_to_fasta(aa_sequences, 'mt_prediciton_tmp.fasta')
        output_df = predict_mt_sequences(output_df, tmp_aa_seq_fasta, mt_predictions, 'query')
        
    if args.pca_test_dir:
        go_term_type = 'target_cellular_components'
        n_clusters = 10
        PCA_tests.lda_pca_analysis(args.pca_test_dir,output_df, 'wmel_parafilt_tm', go_term_type, n_clusters)

    # save dataframe to csv 
    output_df.to_csv(args.csv_out, index=False)
        
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
        validation(evorate_alignment_df, ids_of_interest, args.pdb_database)

if __name__ == '__main__':
    main()