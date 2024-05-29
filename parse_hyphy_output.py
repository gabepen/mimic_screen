import argparse
import os
import json
import pandas as pd

def load_id_map(id_map_file):
    
    '''
    load the tsv file from uniprot id mapping and return a dictionary
    '''
    
    id_map = {}
    with open(id_map_file) as file:
        next(file)  # skip the header line

        for l in file:
            l = l.strip().split('\t')
            if l[1] not in id_map:
                id_map[l[1]] = l[0]

    return id_map
    
def parse_absrel_results(directory, id_map_file):
    
    id_map = load_id_map(id_map_file)
    data = []
    
    # parse directory of absrel result json files
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            
            with open(filepath) as file:
                try:
                    json_data = json.load(file)
                # catching empty json files from halted or failed snakemake runs
                except json.JSONDecodeError:
                    continue
                
                # metrics to collect from the json file
                corrected_pvalues = []
                full_model_nonsyn_branch_lens = [] # Full adaptive model, the branch lengths under this model for nonsynonymous sites
                full_model_syn_branch_lens = [] # Full adaptive model, the branch lengths under this model for synonymous sites
                sig_full_model_nonsyn_branch_lens = []
                sig_full_model_syn_branch_lens = []
                selection_pvalues = 0
                
                
            
                # parse each branch in the json file for appropriate info 
                for branch in json_data['branch attributes']['0']:
                    
                    # values to store for all branches
                    corrected_pvalues.append(json_data['branch attributes']['0'][branch]['Corrected P-value'])
                    full_model_nonsyn_branch_lens.append(json_data['branch attributes']['0'][branch]['Full adaptive model (non-synonymous subs/site)'])
                    full_model_syn_branch_lens.append(json_data['branch attributes']['0'][branch]['Full adaptive model (synonymous subs/site)'])
                                                      
                    # values to store for branches with evidence of selection
                    if json_data['branch attributes']['0'][branch]['Corrected P-value'] < 0.05:
                        selection_pvalues += 1
                        sig_full_model_nonsyn_branch_lens.append(json_data['branch attributes']['0'][branch]['Full adaptive model (non-synonymous subs/site)'])
                        sig_full_model_syn_branch_lens.append(json_data['branch attributes']['0'][branch]['Full adaptive model (synonymous subs/site)'])
                
                # convert file name to samplename if id_map is provided
                if id_map:
                    refseq_accession = os.path.splitext(filename)[0].replace('_absrel', '')
                    try:
                        filename = id_map[refseq_accession]
                    except KeyError:
                        filename = refseq_accession
                    
                
                # add to data frame 
                data.append({
                    'query': filename,
                    'ns_per_site_avg': sum(full_model_nonsyn_branch_lens) / len(full_model_nonsyn_branch_lens) if full_model_nonsyn_branch_lens else None,
                    'syn_per_site_avg': sum(full_model_syn_branch_lens) / len(full_model_syn_branch_lens) if full_model_syn_branch_lens else None,
                    'selection_branch_count': selection_pvalues,
                    'selected_ns_per_site_avg': sum(sig_full_model_nonsyn_branch_lens) / len(sig_full_model_nonsyn_branch_lens) if sig_full_model_nonsyn_branch_lens else None,
                    'selected_syn_per_site_avg': sum(sig_full_model_syn_branch_lens) / len(sig_full_model_syn_branch_lens) if sig_full_model_syn_branch_lens else None,
                    'branch_fraction': selection_pvalues / len(json_data['branch attributes']['0']),
                })
    
    datatable = pd.DataFrame(data)
    return datatable

def main():
    parser = argparse.ArgumentParser(description='Parse JSON files')
    parser.add_argument('-d','--directory', help='Path to the directory containing JSON files')
    parser.add_argument('-m','--id_map', help='Path to the id mapping file from uniprot')
    parser.add_argument('-t','--test_type', help='Type of test')
    args = parser.parse_args()
    
        
    if args.test_type == 'absrel':
        parse_absrel_results(args.directory, args.id_map)

if __name__ == '__main__':
    main()