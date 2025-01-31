import json 
import argparse
import random
import eutils_suite as es
import csv
from glob import glob
from datetime import datetime
from zoneinfo import ZoneInfo
import multiprocessing as mp
from tqdm import tqdm
from loguru import logger
import pdb
import subprocess

def dict_dfs_values(dictionary, rank):

    results = []

    if not isinstance(dictionary, dict):
        return results
    
    for key in dictionary:
        if rank in key:
            results.append((dictionary[key], key))
    
    for key, value in dictionary.items():
        if isinstance(value, dict):
            results.extend(dict_dfs_values(value, rank))
           
    return results

def dict_dfs_keys(dictionary, rank):

    results = []

    if not isinstance(dictionary, dict):
        return results
    
    for key in dictionary:
        if rank in key:
            results.append(key)
    
    for key, value in dictionary.items():
        if isinstance(value, dict):
            results.extend(dict_dfs_keys(value, rank))
           
    return results

def format_output(path):

    '''adds date time info and subcripts
    '''

    output_name = path.split('.')[0].split('/')[-1]
    outdir = '/'.join(path.split('.')[0].split('/')[:-1])

    current_datetime = datetime.now()
    output_name += f"_{current_datetime.day}{current_datetime.month}:{current_datetime.hour}"
    
    vers_num = len(glob(outdir + '/' + output_name +'*.csv'))
    if vers_num > 0:
        output_name += '~{}'.format(vers_num)

    return output_name, outdir

def generate_selection_dict(subtree, pool_level, select_level ):

     # pools : species dictionary 
    selection_dict = {}

    # preform recursive depth first search to identitfy values of specified rank
    results = dict_dfs_values(subtree, pool_level) # select for pools first 

    # now all subtrees for all bacteria poolss have been collected in results
    # next step is dfs the pools dictionaries to the species level returning keys this time
    for r, k in results:
        selections = dict_dfs_keys(r, select_level)
        # proces all species matches into new dicitonary for each pools
        for s in selections:
            if k not in selection_dict:
                selection_dict[k] = [s]
            else:
                selection_dict[k].append(s)
    
    return selection_dict

def collect_already_selected(previous_selection):

    '''Parses a prexisting randomly selected taxon list to avoid duplicate taxids
    '''
    taxids = set()
    with open(previous_selection, 'r') as csv_f:
        header = csv_f.readline()
        lines = csv_f.readlines()
        for l in lines:
            parts = l.split(',')

            # add tax id to set 
            taxids.add(parts[0])
    
    return taxids

def random_taxon_selection(selection_dict, previous_selection, order, dataset_tries=10):

    selected_species = []
    
    if 'Candidatus' not in order and 'unclassified' not in order:
        #  randomly select two species from the pool
       
        species_count = 0
        selection_opts = selection_dict[order]
        logger.info("Processing order: {}, {} options".format(order, len(selection_opts)))
        while species_count < 2:
            
            # randomly select a species from the pool
            opt = random.choice(selection_opts)
            selection_opts.remove(opt)

            # these terms are indicative of a likely host associated genome
            if 'Candidatus' not in opt and 'unclassified' not in opt:
                
                # get taxid of species option
                tax_id = opt.split()[0]

                # skip previously selected taxids 
                if tax_id in previous_selection:
                    continue 
        
                # verify a full genome with NCBI Datasets before continuing 
                # NCBI Datasets can be unreliable so we will retry a few times
                retries = dataset_tries
                result = None
                while retries > 0:
                    
                    # format datasets command
                    command = ["datasets", "summary", "genome", "taxon", str(tax_id)]
                    
                    # run command
                    try:
                        result = subprocess.run(command, capture_output=True, text=True)
                    except Exception as e:
                        logger.info('order: {} opt: {} | datasets subprocess error {} \n'.format(order,opt, e))

                    if result.returncode == 0:
                        break
                    else:
                        retries -= 1
                        continue
                
                # check if datasets command was successful
                if result:
                    if result.returncode == 0:
                        # parse output into dictionary 
                        genome_summary = json.loads(result.stdout)
                        
                        # check each genome report for completeness
                        genome_reports = genome_summary['reports']
                        for report in genome_reports:
                            if report['assembly_info']['assembly_level'] == 'Complete Genome':
                                species_count += 1
                                selected_species.append([tax_id, report['organism']['organism_name'], order, report['assembly_info']['assembly_level'], report['accession']])
                                logger.info('order: {} opt: {} | {} \n'.format(order,opt,report['assembly_info']['assembly_level']))
                                break
                    elif result.returncode == 1 and retries == 0:
                        logger.info('order: {} opt: {} | datasets retries exhausted \n'.format(order,opt))
                    else:
                        logger.info('order: {} opt: {} | datasets failed\n'.format(order,opt))
                else:
                    logger.info('order: {} opt: {} | datasets failed \n'.format(order,opt))

    return selected_species

def worker(selection_dict, previous_selection, order):
    try:
        result = random_taxon_selection(selection_dict, previous_selection, order)
        logger.info(f"Worker result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in worker: {e}")
        return None
  
def multi_process_selection(selection_dict, previously_selected_taxids, num_processes):
    
    pool_opts = list(selection_dict.keys())
    with mp.Pool(processes=num_processes) as pool:
        results = [pool.apply_async(worker, args=(selection_dict, previously_selected_taxids, pool_opt)) for pool_opt in pool_opts]
        output = [p.get() for p in tqdm(results, desc="Processing pools")]
    return output
    
    
def main():

    '''This program randomly selects taxons within two ranks within a taxonDB subtree
       First at a higher rank for pooling and then within a rank subtree below
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tree', type=str, help='path to taxon subtree json')
    parser.add_argument('-p', '--previous', type=str, help='path to directory of previous runs of the taxon_selector program to prevent taxid duplicates')
    parser.add_argument('-l', '--log', action='store_true',help='stores assembly record results for all species queries')
    args = parser.parse_args()

    # format output file 
    output_name, outdir = format_output(args.tree)

    # logging
    if args.log:
        logger.remove()  # Remove default handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = outdir + f"/free_living_check_{timestamp}.log"
        logger.add(log_file, enqueue=True, rotation="10 MB")
    else: 
        logger.remove()

    # load subtree json
    with open(args.tree) as json_f:
        subtree = json.load(json_f)

    # create lookup table for two levels of subtree 
    rank1, rank2 = '[order]', '[species]'
    selection_dict = generate_selection_dict(subtree, rank1, rank2)

    # collect previously selected taxids 
    if args.previous: 
        previously_selected_taxids = set()
        previous_csvs = glob(args.previous + '/*.csv')
        for taxon_csv in previous_csvs:
            taxid_set = collect_already_selected(taxon_csv)
            previously_selected_taxids.update(taxid_set)
    else: previously_selected_taxids = set()

    # generate new results 
    results = multi_process_selection(selection_dict, previously_selected_taxids, 8)
   

    # write out results to tsv
    with open(outdir + '/' + output_name + '.csv', '+w') as csv_f:
        header = ['taxid','organism',rank1,'assembly_status','AssemblyAccession']
        writer = csv.writer(csv_f)
        writer.writerow(header)
        for pool_selection in results:
            if not pool_selection:
                continue 
            else:
                for species in pool_selection:
                    writer.writerow(species)

if __name__ == '__main__':
    main()