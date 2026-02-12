import argparse
import gzip
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
import tarfile
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

def calculate_average_pLDDT(pdb_file_or_handle):
    
    '''Calculates the average pLDDT score from a pdb file or file handle
       pLDDT is located in the b-factor feild of an atom entry
       in a standard format pdb
       
       Args:
           pdb_file_or_handle: Either a file path (string) or file handle (from tar archive)
    '''

    # open and read the pdb file
    try:
        if isinstance(pdb_file_or_handle, str):
            # It's a file path
            with open(pdb_file_or_handle) as pf:
                lines = pf.readlines()
        else:
            # It's a file handle (from tar archive)
            pdb_file_or_handle.seek(0)  # Reset to beginning
            lines = pdb_file_or_handle.readlines()
    except (FileNotFoundError, AttributeError):
        return 'NA'
    
    # parse
    pLDDTs = []
    for l in lines:
        flds = l.strip().split()

        # verify line in file represents an atom
        if len(flds) > 0 and flds[0] == 'ATOM':

            # append to pLDDT list catching issues with formatting of line
            try:
                pLDDTs.append(float(flds[-2]))
            except (IndexError, ValueError):
                print('pLDDT ERROR')
                print(pdb_file_or_handle if isinstance(pdb_file_or_handle, str) else 'file handle')
                return 'ERR' 
    
    if len(pLDDTs) == 0:
        return 'NA'
            
    # return average pDDT for structure
    return mean(pLDDTs)
    
def find_pdb_file(uniprot_id, pdb_source):
    '''
    Finds a PDB file matching the pattern AF-{uniprot_id}-F1-model_v{number}.pdb or .pdb.gz
    Supports version patterns v{number} (e.g., v4, v5)
    
    Args:
        uniprot_id: UniProt ID to search for
        pdb_source: Path to directory or tar archive containing PDB files
        
    Returns:
        Tuple of (file_handle_or_path, is_tar_handle) or (None, None) if not found
    '''
    # Try both .pdb and .pdb.gz patterns
    pattern_pdb = re.compile(rf'AF-{re.escape(uniprot_id)}-F1-model_v(\d+)\.pdb$')
    pattern_pdb_gz = re.compile(rf'AF-{re.escape(uniprot_id)}-F1-model_v(\d+)\.pdb\.gz$')
    
    # Check if source is a tar archive
    if os.path.isfile(pdb_source) and (pdb_source.endswith('.tar') or pdb_source.endswith('.tar.gz')):
        try:
            tar = tarfile.open(pdb_source, 'r:*')
            members = tar.getmembers()
            
            # Find matching files and get highest version (prefer .pdb over .pdb.gz)
            matches = []
            for member in members:
                match_pdb = pattern_pdb.search(member.name)
                match_pdb_gz = pattern_pdb_gz.search(member.name)
                if match_pdb:
                    version = int(match_pdb.group(1))
                    matches.append((version, member, False))  # False = not compressed
                elif match_pdb_gz:
                    version = int(match_pdb_gz.group(1))
                    matches.append((version, member, True))  # True = compressed
            
            if matches:
                # Sort by version and get highest
                matches.sort(key=lambda x: x[0], reverse=True)
                best_match = matches[0][1]
                is_compressed = matches[0][2]
                
                file_handle = tar.extractfile(best_match)
                # If compressed, wrap in gzip decompressor
                if is_compressed:
                    import io
                    file_handle = io.TextIOWrapper(gzip.GzipFile(fileobj=file_handle, mode='rb'))
                
                # Store tar reference in file handle for later closing if needed
                file_handle._tar_ref = tar
                return (file_handle, True)
            tar.close()
        except Exception as e:
            print(f'Error reading tar archive {pdb_source}: {e}')
            return (None, None)
    
    # Otherwise treat as directory
    elif os.path.isdir(pdb_source):
        # Search for matching files
        matches = []
        for filename in os.listdir(pdb_source):
            match_pdb = pattern_pdb.search(filename)
            match_pdb_gz = pattern_pdb_gz.search(filename)
            if match_pdb:
                version = int(match_pdb.group(1))
                matches.append((version, os.path.join(pdb_source, filename), False))
            elif match_pdb_gz:
                version = int(match_pdb_gz.group(1))
                matches.append((version, os.path.join(pdb_source, filename), True))
        
        if matches:
            # Sort by version and get highest
            matches.sort(key=lambda x: x[0], reverse=True)
            return (matches[0][1], False)
    
    return (None, None)

def calculate_region_pLDDT(pdb_file_or_handle, start_pos, end_pos):
    '''
    Calculates average pLDDT for a specific residue range in a PDB file
    Uses residue-based approach: identifies residues in range, then averages all atoms
    
    Args:
        pdb_file_or_handle: Either a file path (string) or file handle
        start_pos: Start residue position (1-indexed)
        end_pos: End residue position (1-indexed, inclusive)
        
    Returns:
        Average pLDDT for the region or 'NA' if error
    '''
    try:
        if isinstance(pdb_file_or_handle, str):
            # It's a file path
            with open(pdb_file_or_handle) as pf:
                lines = pf.readlines()
        else:
            # It's a file handle (from tar archive)
            pdb_file_or_handle.seek(0)  # Reset to beginning
            lines = pdb_file_or_handle.readlines()
    except (FileNotFoundError, AttributeError):
        return 'NA'
    
    # Parse PDB file and collect pLDDT values for residues in range
    # Residue number is in columns 23-26 (1-indexed) of ATOM line
    pLDDTs = []
    for l in lines:
        if len(l) < 26:
            continue
            
        if l.startswith('ATOM'):
            try:
                # Extract residue number (columns 23-26, 0-indexed: 22-25)
                residue_num_str = l[22:26].strip()
                if residue_num_str:
                    residue_num = int(residue_num_str)
                    
                    # Check if residue is in range
                    if start_pos <= residue_num <= end_pos:
                        # Extract pLDDT from b-factor field (second to last field)
                        flds = l.strip().split()
                        if len(flds) >= 2:
                            pLDDTs.append(float(flds[-2]))
            except (ValueError, IndexError):
                continue
    
    if len(pLDDTs) == 0:
        return 'NA'
    
    return mean(pLDDTs)

def add_plddt_columns(data_frame, query_pdb_source=None, target_pdb_source=None):
    '''
    Adds four pLDDT columns to the dataframe:
    - plddt_query_avg: Average pLDDT for entire query structure
    - plddt_target_avg: Average pLDDT for entire target structure
    - plddt_query_region: Average pLDDT for query alignment region (qstart to qend)
    - plddt_target_region: Average pLDDT for target alignment region (tstart to tend)
    
    Args:
        data_frame: DataFrame with columns 'query', 'target', 'qstart', 'qend', 'tstart', 'tend'
        query_pdb_source: Path to directory or tar archive containing query PDB files
        target_pdb_source: Path to directory or tar archive containing target PDB files
        
    Returns:
        DataFrame with four new pLDDT columns added
    '''
    # Initialize columns
    data_frame['plddt_query_avg'] = None
    data_frame['plddt_target_avg'] = None
    data_frame['plddt_query_region'] = None
    data_frame['plddt_target_region'] = None
    
    # Pre-load all unique query and target IDs to find PDB files upfront
    unique_query_ids = set(data_frame['query'].unique()) if query_pdb_source else set()
    unique_target_ids = set(data_frame['target'].unique()) if target_pdb_source else set()
    
    print(f"Pre-loading PDB files: {len(unique_query_ids)} unique queries, {len(unique_target_ids)} unique targets...")
    
    # Cache for PDB file handles and calculated values
    query_pdb_cache = {}
    target_pdb_cache = {}
    query_avg_cache = {}
    target_avg_cache = {}
    query_pdb_lines_cache = {}  # Cache file contents to avoid re-reading
    target_pdb_lines_cache = {}
    
    # Track open tar files to close them later (use set to avoid duplicates)
    open_tar_files = set()
    
    # Pre-load query PDB files
    if query_pdb_source:
        for query_id in tqdm(unique_query_ids, desc='Loading query PDB files'):
            pdb_file, is_tar = find_pdb_file(query_id, query_pdb_source)
            if pdb_file:
                query_pdb_cache[query_id] = (pdb_file, is_tar)
                if is_tar and hasattr(pdb_file, '_tar_ref'):
                    open_tar_files.add(pdb_file._tar_ref)
                # Read file contents once and cache
                try:
                    if isinstance(pdb_file, str):
                        # Check if it's a .gz file
                        if pdb_file.endswith('.gz'):
                            with gzip.open(pdb_file, 'rt') as pf:
                                query_pdb_lines_cache[query_id] = pf.readlines()
                        else:
                            with open(pdb_file) as pf:
                                query_pdb_lines_cache[query_id] = pf.readlines()
                    else:
                        pdb_file.seek(0)
                        query_pdb_lines_cache[query_id] = pdb_file.readlines()
                    # Calculate average pLDDT from cached lines
                    query_avg_cache[query_id] = calculate_average_pLDDT_from_lines(query_pdb_lines_cache[query_id])
                except Exception as e:
                    query_avg_cache[query_id] = 'NA'
            else:
                query_avg_cache[query_id] = 'NA'
    
    # Pre-load target PDB files
    if target_pdb_source:
        found_count = 0
        not_found_ids = []
        for target_id in tqdm(unique_target_ids, desc='Loading target PDB files'):
            pdb_file, is_tar = find_pdb_file(target_id, target_pdb_source)
            if pdb_file:
                found_count += 1
                target_pdb_cache[target_id] = (pdb_file, is_tar)
                if is_tar and hasattr(pdb_file, '_tar_ref'):
                    open_tar_files.add(pdb_file._tar_ref)
                # Read file contents once and cache
                try:
                    if isinstance(pdb_file, str):
                        # Check if it's a .gz file
                        if pdb_file.endswith('.gz'):
                            with gzip.open(pdb_file, 'rt') as pf:
                                target_pdb_lines_cache[target_id] = pf.readlines()
                        else:
                            with open(pdb_file) as pf:
                                target_pdb_lines_cache[target_id] = pf.readlines()
                    else:
                        pdb_file.seek(0)
                        target_pdb_lines_cache[target_id] = pdb_file.readlines()
                    # Calculate average pLDDT from cached lines
                    target_avg_cache[target_id] = calculate_average_pLDDT_from_lines(target_pdb_lines_cache[target_id])
                except Exception as e:
                    print(f"Error reading target PDB for {target_id}: {e}")
                    target_avg_cache[target_id] = 'NA'
            else:
                not_found_ids.append(target_id)
                target_avg_cache[target_id] = 'NA'
        
        print(f"Target PDB files: {found_count} found, {len(not_found_ids)} not found")
        if len(not_found_ids) > 0 and len(not_found_ids) <= 10:
            print(f"  Not found target IDs (first 10): {not_found_ids}")
        elif len(not_found_ids) > 10:
            print(f"  Not found target IDs (first 10): {not_found_ids[:10]}... ({len(not_found_ids)} total)")
    
    print(f"Calculating pLDDT values for {len(data_frame)} alignments...")
    
    # Now calculate pLDDT values using cached data
    for index, row in tqdm(data_frame.iterrows(), total=len(data_frame), desc='Calculating pLDDT'):
        query_id = row['query']
        target_id = row['target']
        
        # Calculate query pLDDT if source provided
        if query_pdb_source:
            data_frame.at[index, 'plddt_query_avg'] = query_avg_cache.get(query_id, 'NA')
            
            # Calculate query region pLDDT using cached lines
            if query_id in query_pdb_lines_cache and query_avg_cache.get(query_id, 'NA') != 'NA':
                try:
                    qstart = int(float(row['qstart']))
                    qend = int(float(row['qend']))
                    region_plddt = calculate_region_pLDDT_from_lines(query_pdb_lines_cache[query_id], qstart, qend)
                    data_frame.at[index, 'plddt_query_region'] = region_plddt
                except (ValueError, TypeError):
                    data_frame.at[index, 'plddt_query_region'] = 'NA'
            else:
                data_frame.at[index, 'plddt_query_region'] = 'NA'
        
        # Calculate target pLDDT if source provided
        if target_pdb_source:
            data_frame.at[index, 'plddt_target_avg'] = target_avg_cache.get(target_id, 'NA')
            
            # Calculate target region pLDDT using cached lines
            if target_id in target_pdb_lines_cache and target_avg_cache.get(target_id, 'NA') != 'NA':
                try:
                    tstart = int(float(row['tstart']))
                    tend = int(float(row['tend']))
                    region_plddt = calculate_region_pLDDT_from_lines(target_pdb_lines_cache[target_id], tstart, tend)
                    data_frame.at[index, 'plddt_target_region'] = region_plddt
                except (ValueError, TypeError):
                    data_frame.at[index, 'plddt_target_region'] = 'NA'
            else:
                data_frame.at[index, 'plddt_target_region'] = 'NA'
    
    # Close any open tar files
    for tar_ref in open_tar_files:
        if tar_ref:
            try:
                tar_ref.close()
            except:
                pass
    
    return data_frame

def calculate_average_pLDDT_from_lines(lines):
    '''Calculate average pLDDT from pre-read lines (for performance)'''
    pLDDTs = []
    for l in lines:
        flds = l.strip().split()
        if len(flds) > 0 and flds[0] == 'ATOM':
            try:
                pLDDTs.append(float(flds[-2]))
            except (IndexError, ValueError):
                continue
    
    if len(pLDDTs) == 0:
        return 'NA'
    
    return mean(pLDDTs)

def calculate_region_pLDDT_from_lines(lines, start_pos, end_pos):
    '''Calculate region pLDDT from pre-read lines (for performance)'''
    pLDDTs = []
    for l in lines:
        if len(l) < 26:
            continue
            
        if l.startswith('ATOM'):
            try:
                # Extract residue number (columns 23-26, 0-indexed: 22-25)
                residue_num_str = l[22:26].strip()
                if residue_num_str:
                    residue_num = int(residue_num_str)
                    
                    # Check if residue is in range
                    if start_pos <= residue_num <= end_pos:
                        # Extract pLDDT from b-factor field (second to last field)
                        flds = l.strip().split()
                        if len(flds) >= 2:
                            pLDDTs.append(float(flds[-2]))
            except (ValueError, IndexError):
                continue
    
    if len(pLDDTs) == 0:
        return 'NA'
    
    return mean(pLDDTs)

def _normalize_query_id(raw_id):
    """
    Normalize a query ID so that control dictionaries and alignment files
    use a consistent key format.
    
    Current issue:
    - Control dictionaries were built with keys like 'AF-Q5ZRZ3-F1-model_v4.pdb'
    - Alignment files use 'AF-Q5ZRZ3-F1-model_v4' (no extension)
    
    We strip any trailing '.pdb' or '.pdb.gz' so both sides can match on the
    extension-free ID.
    """
    if raw_id.endswith('.pdb'):
        return raw_id[:-4]
    if raw_id.endswith('.pdb.gz'):
        return raw_id[:-7]
    return raw_id


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
                # use a normalized query ID so keys are extension-independent
                query = _normalize_query_id(lp[0])
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
            # Normalize query ID to match control_dict keys (which are stored
            # without .pdb / .pdb.gz extensions by generate_control_dictionary)
            query = _normalize_query_id(lp[0])
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
    open_tar_files = []
    for index, row in df.iterrows():
        for uni_id in ids_of_interest:
            if uni_id in row['query']:
  
                # calculate structure prediction confidence
                if uni_id not in average_pLDDTs and structure_db != None:
                    pdb_file, is_tar = find_pdb_file(uni_id, structure_db)
                    if pdb_file:
                        if is_tar:
                            open_tar_files.append(pdb_file._tar_ref if hasattr(pdb_file, '_tar_ref') else None)
                        average_pLDDTs[uni_id] = calculate_average_pLDDT(pdb_file)
                    else:
                        average_pLDDTs[uni_id] = 'NA'

                # Get columns that exist in the dataframe
                available_cols = ['query', 'target', 'score', 'tcov', 'qcov', 'fident', 'algn_fraction']
                if 'symbiont_branch_dnds_avg' in df.columns:
                    available_cols.append('symbiont_branch_dnds_avg')
                if 'non_symbiont_branch_dnds_avg' in df.columns:
                    available_cols.append('non_symbiont_branch_dnds_avg')
                
                row_string = row[available_cols].astype(str).str.cat(sep=',') + ',' + str(average_pLDDTs[uni_id])
                print(row_string) 
    
    # Close any open tar files
    for tar_ref in open_tar_files:
        if tar_ref:
            try:
                tar_ref.close()
            except:
                pass 
                
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
    
def paralog_filter(target_group, host_self_alignment_table, query_tcov_scores, query_algn_fraction_scores):
    
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
        avg_query_score = sum(query_algn_fraction_scores) / len(query_algn_fraction_scores)
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

    U_statistic, p_value = mannwhitneyu(sequence_identity_scores, query_algn_fraction_scores)
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
            
            #  get tcov and algn_fraction across all alignments for query id
            query_tcov_scores = [query_table[query_id][target][1] for target in targets]
            query_algn_fraction_scores = [query_table[query_id][target][4] for target in targets]
            
            # determine if the divergence in structure between the host proteins 
            # is the same as the divergence between the query and host proteins
            # (i.e. very likely houskeeping genes)
            result, seq_id_average, tm_score_avg = paralog_filter(targets, host_self_alignment_table, query_tcov_scores, query_algn_fraction_scores)

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
    
def add_protein_description(data_frame, field_name, json_path='prot_descriptions.json'):
    
    '''
    Adds a new column 'description' to the data frame by querying the UniProt API for each query ID.
    
    Args:
        data_frame: DataFrame to add descriptions to
        field_name: Column name containing UniProt IDs
        json_path: Path to JSON file for caching/loading descriptions
    '''
    
    # Check for id dict for previous runs 
    if not os.path.exists(json_path):
        id_descriptions = {}
        print(f"Warning: JSON file not found at {json_path}, will create new cache")
    else:
        print(f"Loading cached descriptions from {json_path}...")
        with open(json_path, 'r') as json_f:
            id_descriptions = json.load(json_f)
        print(f"Loaded {len(id_descriptions)} cached entries")
    
    # Get unique IDs that need to be queried
    unique_ids = data_frame[field_name].unique()
    ids_to_query = [uid for uid in unique_ids if uid not in id_descriptions]
    ids_cached = len(unique_ids) - len(ids_to_query)
    
    if ids_cached > 0:
        print(f"Found {ids_cached} IDs in cache, {len(ids_to_query)} need to be queried")
    
    # Query UniProt API for missing IDs with progress bar
    if len(ids_to_query) > 0:
        print(f"Fetching {len(ids_to_query)} protein descriptions from UniProt API...")
        for query_id in tqdm(ids_to_query, desc=f'Fetching {field_name} descriptions'):
            try:
                entry = uniprot_api_queries.fetch_uniprot_entry(query_id)
                try:
                    description = entry['proteinDescription']['recommendedName']['fullName']['value']
                except KeyError:
                    try:
                        description = entry['proteinDescription']['submissionNames'][0]['fullName']['value']
                    except KeyError:
                        description = 'NA'
                id_descriptions[query_id] = description
            except Exception as e:
                print(f"Error fetching description for {query_id}: {e}")
                id_descriptions[query_id] = 'NA'
    
    # Map descriptions to dataframe
    descriptions = [id_descriptions.get(query_id, 'NA') for query_id in data_frame[field_name]]
    output_field = field_name + '_description'
    data_frame[output_field] = descriptions
    
    # save id descriptions to json
    with open(json_path, 'w') as json_f:
        json.dump(id_descriptions, json_f, indent=2)
        
    return data_frame

def add_go_terms(data_frame, field_name, json_path='prot_descriptions.json'):  
    
    '''
    Adds GO term columns to the data frame by querying the UniProt API.
    
    Args:
        data_frame: DataFrame to add GO terms to
        field_name: Column name containing UniProt IDs
        json_path: Path to JSON file for caching/loading GO terms
    '''
    
    # Check for id dict for previous runs 
    if not os.path.exists(json_path):
        id_descriptions = {}
    else:
        print(f"Loading cached GO terms from {json_path}...")
        with open(json_path, 'r') as json_f:
            id_descriptions = json.load(json_f)
        # Count GO term entries (those ending in '_GO')
        go_entries = sum(1 for k in id_descriptions.keys() if k.endswith('_GO'))
        print(f"Loaded {go_entries} cached GO term entries")
    
    # Get unique IDs that need to be queried
    unique_ids = data_frame[field_name].unique()
    ids_to_query = []
    for search_id in unique_ids:
        query_id_key = search_id + '_GO'
        if query_id_key not in id_descriptions:
            ids_to_query.append(search_id)
    
    ids_cached = len(unique_ids) - len(ids_to_query)
    if ids_cached > 0:
        print(f"Found {ids_cached} IDs in cache, {len(ids_to_query)} need to be queried")
    
    # Query UniProt API for missing IDs with progress bar
    if len(ids_to_query) > 0:
        print(f"Fetching {len(ids_to_query)} GO term annotations from UniProt API...")
        for search_id in tqdm(ids_to_query, desc=f'Fetching {field_name} GO terms'):
            query_id_key = search_id + '_GO'
            try:
                # fetch entry from UniProt API
                entry = uniprot_api_queries.fetch_uniprot_entry(search_id)
                try:
                    refs = entry['uniProtKBCrossReferences']
                except KeyError:
                    refs = []
                    
                # parse external references for GO terms
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
                
                # store terms for look up 
                id_descriptions[query_id_key] = term_dict
            except Exception as e:
                print(f"Error fetching GO terms for {search_id}: {e}")
                id_descriptions[query_id_key] = {'C': [], 'F': [], 'P': []}
    
    # Map GO terms to dataframe
    cell_comps = []
    mol_func = []
    bio_proc = []
    
    for search_id in data_frame[field_name]:
        query_id_key = search_id + '_GO'
        if query_id_key in id_descriptions:
            term_dict = id_descriptions[query_id_key]
            cell_comps.append(term_dict.get('C', []))
            mol_func.append(term_dict.get('F', []))
            bio_proc.append(term_dict.get('P', []))
        else:
            cell_comps.append([])
            mol_func.append([])
            bio_proc.append([])
    
    output_field = field_name + '_cellular_components'
    data_frame[output_field] = cell_comps
    output_field = field_name + '_molecular_functions'
    data_frame[output_field] = mol_func
    output_field = field_name + '_biological_processes'
    data_frame[output_field] = bio_proc
    
     # save id GO terms to json
    with open(json_path, 'w') as json_f:
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
    parser.add_argument('-qd','--pdb_database_query', type=str, help='path to directory or tar archive containing query PDB files')
    parser.add_argument('-td','--pdb_database_target', type=str, help='path to directory or tar archive containing target PDB files')
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
    parser.add_argument('--all_alignments', action='store_true', help='Output all alignments without deduplication')
    parser.add_argument('--skip_uniprot_metadata', action='store_true', help='Skip UniProt API calls for protein descriptions and GO terms (faster, but output will not include these columns)')
    parser.add_argument('--uniprot_metadata_json', type=str, default='prot_descriptions.json', help='Path to JSON file for caching/loading UniProt metadata (default: prot_descriptions.json)')
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
        
    if args.all_alignments:
        alignment_df = make_output_df_no_filters(alignment_table)
        # Add paralog filter column if paralog filter is provided
        if host_self_alignment_table is not None:
            paralog_results = {}
            for query_id in alignment_df['query'].unique():
                targets = alignment_df[alignment_df['query'] == query_id]['target'].unique().tolist()
                if len(targets) > 1:
                    query_tcov_scores = alignment_df[alignment_df['query'] == query_id]['tcov'].tolist()
                    query_algn_fraction_scores = alignment_df[alignment_df['query'] == query_id]['algn_fraction'].tolist()
                    result, _, _ = paralog_filter(targets, host_self_alignment_table, query_tcov_scores, query_algn_fraction_scores)
                    paralog_results[query_id] = 'yes' if result else 'no'
                else:
                    paralog_results[query_id] = None
            alignment_df['targets_likely_paralogs'] = alignment_df['query'].map(paralog_results)
        else:
            alignment_df['targets_likely_paralogs'] = None
    else:
        alignment_df = make_output_df(alignment_table, host_self_alignment_table, args.top_hits, args.alignment_span_filter)
    output_df = alignment_df.copy()
        
    if args.evorate_analysis:
        
        absrel_df = parse_hyphy_output.parse_absrel_results(args.evorate_analysis, args.symbiont_ids, args.id_map)
        if args.evorate_analysis2:
            
            # statistical comparison of dn/ds rates between two evorate analyses
            absrel_df2 = parse_hyphy_output.parse_absrel_results(args.evorate_analysis2, args.symbiont_ids, args.id_map)
            
        # aBSREL dnds comparison candidates
        evorate_alignment_df = pd.merge(alignment_df, absrel_df, on='query', how='left')
        if args.plot_evorate:
            plot_test_fraction(evorate_alignment_df,'/storage1/gabe/mimic_screen/main_paper/final_figs/evorate_figs/wmel_testfraction-FFPA.png', 10)
            plot_aBSREL_comparisons_candidates(evorate_alignment_df, args.plot_evorate, 10, 'Symbiont vs Non-Symbiont Branch dN/dS Averages wMelCandidates')  
        columns_to_drop = ['ns_per_site_avg', 'syn_per_site_avg', 'dnds_tree_avg', 'symbiont_tree_dnds_avg', 'non_symbiont_tree_dnds_avg', 'selection_branch_count', 'total_branch_length', 'avg_branch_length', 'selected_ns_per_site_avg', 'selected_syn_per_site_avg','branch_fraction','branch_fraction_full_norm']
        output_evorate_alignment_df = evorate_alignment_df.drop(columns=columns_to_drop)
        output_df = output_evorate_alignment_df.copy()
       
    # add pLDDT columns if PDB databases are provided (do this BEFORE UniProt API calls for faster feedback)
    if args.pdb_database_query or args.pdb_database_target:
        output_df = add_plddt_columns(output_df, args.pdb_database_query, args.pdb_database_target)
    
    # add protein descriptions and GO terms to the output dataframe (optional, can be slow)
    if not args.skip_uniprot_metadata:
        output_df = add_protein_description(output_df, 'query', args.uniprot_metadata_json)  
        output_df = add_protein_description(output_df, 'target', args.uniprot_metadata_json)
        output_df = add_go_terms(output_df, 'target', args.uniprot_metadata_json)
    
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

    # save dataframe to csv 
    output_df.to_csv(args.csv_out, index=False)
        
    # plot histogram of freeliving fraction values for alignment
    if args.fid_plot:
        plot_freeliving_fraction_distribution(alignment_table, args.fid_plot)
    
    # parse list of validation or general IDs of interest to pull from the results of an alignment
    if args.validation_ids:
        ids_of_interest = set()
        with open(args.validation_ids, 'r') as txt_f:
            ids = txt_f.readlines()
            for struct_id in ids:
                # add structure IDs to id set 
                ids_of_interest.add(struct_id.strip())

        # find ids and print to std_out in csv format
        # Use query database for validation (legacy behavior)
        validation_db = args.pdb_database_query if args.pdb_database_query else args.pdb_database_target
        if 'evorate_alignment_df' in locals():
            validation(evorate_alignment_df, ids_of_interest, validation_db)
        else:
            validation(output_df, ids_of_interest, validation_db)

if __name__ == '__main__':
    main()