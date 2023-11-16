import json 
import argparse
import random
import eutils_suite as es
import csv
from glob import glob
from datetime import datetime
from zoneinfo import ZoneInfo

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
        log_file = open(outdir + '/' + output_name + '.log', 'w+')
    else: log_file = None

    # load subtree json
    with open(args.tree) as json_f:
        subtree = json.load(json_f)

    # create lookup table for two levels of subtree 
    rank1, rank2 = '[order]', '[species]'
    selection_dict = generate_selection_dict(subtree, rank1, rank2)

    # collect previously selected taxids 
    previously_selected_taxids = set()
    previous_csvs = glob(args.previous + '/*.csv')
    for taxon_csv in previous_csvs:
        taxid_set = collect_already_selected(taxon_csv)
        previously_selected_taxids.update(taxid_set)

    # initialize entrez database toolkit objects
    genome_lib = es.Librarian('genome')
    assembly_lib = es.Librarian('assembly')

   
    # write out results to tsv
    with open(outdir + '/' + output_name + '.csv', '+w') as csv_f:
        header = ['taxid','organism',rank1,'assembly_status','RsUid','GbUid','AssemblyAccession']
        writer = csv.writer(csv_f)
        writer.writerow(header)

        # randomly select pools that are not unclassifed or Candidatus
        o_count = 0
        pool_opts = list(selection_dict.keys())
        while o_count < 100:
            pool = random.choice(pool_opts)
            pool_opts.remove(pool)
            if 'Candidatus' not in pool and 'unclassified' not in pool:
                #  randomly select two species from the pools 

                '''break after two sepcies are selected or if all species opts 
                   are exhausted
                '''
                try:
                    s_count = 0
                    selection_opts = selection_dict[pool]
                    while s_count < 4:
                        # monitoring
                        opt = random.choice(selection_opts)
                        selection_opts.remove(opt)

                        print('pools remaining:{} options remaining:{} o_count:{} s_count:{}'.format(
                            len(pool_opts), len(selection_opts), o_count, s_count), end="\r" )

                        if 'Candidatus' not in opt and 'unclassified' not in opt:
                            tax_id = opt.split()[0]

                            # skip previously selected taxids 
                            if tax_id in previously_selected_taxids:
                                continue 
                    
                            # verify a full genome with entrezpy before continuing 
                            record = genome_lib.find_gid_from_taxid(tax_id)
                            
                            # check if there is a genome ID associated with taxID
                            try: 
                                genome_record = genome_lib.find_id_summary(record['IdList'][0])

                                # check if genome has an assembly 
                                if genome_record[0]['AssemblyID'] != '':
                                    assembly_record = assembly_lib.find_id_summary(genome_record[0]['AssemblyID'])
                                    assembly_status = assembly_record['DocumentSummarySet']['DocumentSummary'][0]['AssemblyStatus']  
                                    
                                    # check if assembly is complete then collect supporting information
                                    if 'Complete' in assembly_status:
                                        organism = assembly_record['DocumentSummarySet']['DocumentSummary'][0]['Organism']
                                        rs_uid = assembly_record['DocumentSummarySet']['DocumentSummary'][0]['RsUid']
                                        gb_uid = assembly_record['DocumentSummarySet']['DocumentSummary'][0]['GbUid']
                                        asscension = assembly_record['DocumentSummarySet']['DocumentSummary'][0]['AssemblyAccession']
                                        
                                        # option found increment count and write entry out 
                                        s_count += 1
                                        writer.writerow([tax_id, organism, pool, assembly_status, rs_uid, gb_uid, asscension])
                                
                                # logging 
                                if log_file:
                                    log_file.write('pool: {} opt: {} | {} \n'.format(pool,opt,assembly_status))

                            except IndexError:
                                if log_file:
                                    log_file.write('pool: {} opt: {} | no genome record found \n'.format(pool,opt))
                                continue
                            except TypeError:
                                continue
                    # s_count reached before opts exhausted 
                    o_count += 1

                except IndexError:
                    continue
                # both species found in pools increment
    log_file.close()

if __name__ == '__main__':
    main()