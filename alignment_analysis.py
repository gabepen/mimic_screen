import argparse
import numpy as np
from glob import glob 
import os 
from tqdm import tqdm
import json
import re
from statistics import mean 
from Bio.PDB import PDBParser
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter, PercentFormatter

def calculate_average_pLDDT(pdb_file):
    
    '''Calculates the average pLDDT score from a pdb file 
       pLDDT is located in the b-factor feild of an atom entry
       in a standard format pdb
    '''

    # open and read the pdb file
    with open(pdb_file) as pf:
        lines = pf.readlines()

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

            # alignment base statistic requirements
            if float(score) > 0.4 and tcov > 0.25 and evalue < 0.01:

                # check presence in control proteomes
                if query in control_dict:
                    if control_dict[query]['algn_fraction'] > 0.8:
                        data_table.append([query, float(score), tcov, fident, control_dict[query]['algn_fraction'], target, 'controlled'])
                    else:
                        data_table.append([query, float(score), tcov, fident, control_dict[query]['algn_fraction'], target, 'candidate'])
                else:
                    data_table.append([query, float(score), tcov, fident,0.0, target,'candidate'])      

            else:
                try:
                    data_table.append([query, float(score), tcov, fident, control_dict[query]['algn_fraction'], target, 'filtered'])
                except KeyError:
                    data_table.append([query, float(score), tcov, fident,0.0, target,'filtered']) 

            
    return data_table

def validation(data_table, ids_of_interest):

    # find validation IDs in results
    average_pLDDTs = {}
    for row in data_table:
        for uni_id in ids_of_interest:
            if uni_id in row[0]:

                # want to know the correlation between free living presence and pLDDT
                if uni_id not in average_pLDDTs:
                    average_pLDDTs[uni_id] = calculate_average_pLDDT(legion_db + '/' + row[0])
            
                # clean up file name to just have UNIProt ID
                start = 'AF-'
                end = '-F1'
                query_id = re.search(f'{start}(.*?){end}', row[0]).group(1) if re.search(f'{start}(.*?){end}', row[0]) else row[0]
                target_id = re.search(f'{start}(.*?){end}', row[5]).group(1) if re.search(f'{start}(.*?){end}', row[5]) else row[5]

                # match order of existing table in paper 
                output = [query_id, target_id, row[1], row[2], row[3], row[4],  average_pLDDTs[uni_id]]
                output = map(str, output)
                print(','.join(output))

def results_to_stdout(data_table, query_db):
    
    '''pipe results dataframe to stdout in csv format
    '''

    # get results 
    average_pLDDTs = {}
    pairs = set()
    for row in data_table:

        if row[-1] == 'filtered':
            continue
        
        # store average pLDDTS
        if row[0] not in average_pLDDTs and query_db != None:
            average_pLDDTs[row[0]] = calculate_average_pLDDT(query_db + '/' + row[0])
        else:
            average_pLDDTs[row[0]] = 'NA'

        # clean up file name to just have UNIProt ID
        start = 'AF-'
        end = '-F1'
        target_id = re.search(f'{start}(.*?){end}', row[5]).group(1) if re.search(f'{start}(.*?){end}', row[5]) else row[5]
        query_id = re.search(f'{start}(.*?){end}', row[0]).group(1) if re.search(f'{start}(.*?){end}', row[0]) else row[0].split('_')[0]

        alignment_pair  = (query_id, target_id) 
        
        if alignment_pair not in pairs:
            # format output order
            output = [query_id, target_id, row[1], row[2], row[3], row[4],  average_pLDDTs[row[0]], row[-1]]
            output = map(str, output)
            pairs.add(alignment_pair)
            print(','.join(output))

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

def main():

    '''Script that determines the overall presence of wMel protein alignments with the free-living control dataset.

       alignment files should be generated with foldseek using:
        --format-output query,target,evalue,alntmscore,alnlen,qlen,tcov,qcov,tlen,u,t,lddt,fident,pident,prob
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument('-a','--alignment', type=str, help='path to an expiremental alignment file')
    parser.add_argument('-c','--controls', type=str, help='paths to a directory of control alignments')
    parser.add_argument('-f','--fid_plot', type=str, help='output path for png of fraction of identical residues histogram')
    parser.add_argument('-j','--json_file', type=str, help='path to a pre-generated json of control alignment stats')
    parser.add_argument('-o','--std_out',  action='store_true', help='output csv table of results to std_out')
    parser.add_argument('-p','--pdb_database', type=str, help='path to database of pdb files used in alignment')
    parser.add_argument('-v','--validation_ids', type=str, help='path to a txt file of structure IDs to pull from results')
    args = parser.parse_args()   

    # either generate the control dictionary or load it from previous run 
    if not args.json_file:
        control_dictionary = generate_control_dictionary(args.controls)
    else:
        with open(args.json_file, 'r') as json_f:
            control_dictionary = json.load(json_f)

    # generate alignment stats data table 
    data_table = alignment_stats(args.alignment, control_dictionary)
    
    # plot histogram of freeliving fraction values for alignment
    if args.fid_plot:
        plot_freeliving_fraction_distribution(data_table, args.fid_plot)
    
    # output csv to stdout
    if args.std_out and args.pdb_database:
        results_to_stdout(data_table, args.pdb_database)
    elif args.std_out:
        results_to_stdout(data_table, None)

    # parse list of validation or general IDs of interest to pull from the results of an alignment
    if args.validation_ids:
        ids_of_interest = set()
        with open(args.validation_ids, 'r') as txt_f:
            ids = txt_f.readlines()
            for struct_id in ids:
                # add structure IDs to id set 
                ids_of_interest.add(struct_id.strip())

        # find ids and print to std_out in csv format
        validation(data_table, ids_of_interest)

if __name__ == '__main__':
    main()