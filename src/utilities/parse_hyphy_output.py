import argparse
import os
import json
import pandas as pd
import pdb

def load_id_map(id_map_file):
    
    '''
    load the tsv file from uniprot id mapping and return a dictionary
    '''
    
    id_map = {}
    with open(id_map_file) as file:

        for l in file:
            l = l.strip().split('\t')
            if l[1] not in id_map:
                id_map[l[1]] = l[0]

    return id_map
    
def parse_absrel_results(directory, symbiont_ids, id_map_file):
    
    if id_map_file:
        id_map = load_id_map(id_map_file)
    data = []
    
    if symbiont_ids:
        with open(symbiont_ids) as file:
            symbiont_ids = set([l.strip() for l in file.readlines()])
    
    # debug stat tracking
    tree_lengths = []
    
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
                
                # In the parse_absrel_results function, after loading the JSON (around line 46),
# add this check before processing branch attributes:

            with open(filepath) as file:
                try:
                    json_data = json.load(file)
                # catching empty json files from halted or failed snakemake runs
                except json.JSONDecodeError:
                    continue
                
                # Check if this is a "no_orthologs_found" error file
                if 'error' in json_data and json_data['error'] == 'no_orthologs_found':
                    # Extract sample name from filename or JSON
                    refseq_accession = os.path.splitext(filename)[0].replace('_absrel', '')
                    if 'sample' in json_data:
                        refseq_accession = json_data['sample']
                    
                    # Convert to prot_id if id_map is provided
                    if id_map_file:
                        try:
                            prot_id = id_map[refseq_accession]
                        except KeyError:
                            prot_id = refseq_accession
                    else:
                        prot_id = refseq_accession
                    
                    # Add entry with all None values
                    data.append({
                        'query': prot_id,
                        'accession': refseq_accession,
                        'ns_per_site_avg': None,
                        'syn_per_site_avg': None,
                        'dnds_tree_avg': None,
                        'symbiont_branch_dnds_avg': None,
                        'symbiont_tree_dnds_avg': None,
                        'non_symbiont_branch_dnds_avg': None,
                        'non_symbiont_tree_dnds_avg': None,
                        'selection_branch_count': None,
                        'total_branch_length': None,
                        'avg_branch_length': None,
                        'selected_ns_per_site_avg': None,
                        'selected_syn_per_site_avg': None,
                        'branch_fraction': None,
                        'branch_fraction_full_norm': None,
                        'test_fraction': None
                    })
                    continue
                
                # metrics to collect from the json file
                corrected_pvalues = []
                # metrics to collect from the json file
                corrected_pvalues = []
                full_model_nonsyn_branch_lens = [] # Full adaptive model, the branch lengths under this model for nonsynonymous sites
                full_model_syn_branch_lens = [] # Full adaptive model, the branch lengths under this model for synonymous sites
                sig_full_model_nonsyn_branch_lens = []
                sig_full_model_syn_branch_lens = []
                branch_lengths = []
                branch_dnds = []
                symbiont_branch_dnds = []
                symbiont_branch_dn = []
                symbiont_branch_ds = []
                non_symbiont_branch_dnds = []
                non_symbiont_branch_dn = []
                non_symbiont_branch_ds = []
                selection_pvalues = 0
                foreground_branches = 0
                background_branches = 0
                total_branches = 0
                
                # parse each branch in the json file for appropriate info 
                for branch in json_data['branch attributes']['0']:
                    
                    # values to store for all branches
                    corrected_pvalues.append(json_data['branch attributes']['0'][branch]['Corrected P-value'])
                    full_model_nonsyn_branch_lens.append(json_data['branch attributes']['0'][branch]['Full adaptive model (non-synonymous subs/site)'])
                    full_model_syn_branch_lens.append(json_data['branch attributes']['0'][branch]['Full adaptive model (synonymous subs/site)'])
                    branch_lengths.append(json_data['branch attributes']['0'][branch]['Full adaptive model'])  
                    branch_dnds.append(full_model_nonsyn_branch_lens[-1] / full_model_syn_branch_lens[-1])
                        
                    try:
                        branch_taxid = branch.split('_')[-1]
                        # skip internal nodes
                        if branch_taxid.startswith('Node'):
                            continue
                        # Handle MIMIC_CANDIDATE suffix - taxon ID is 2 positions before the end
                        if branch_taxid == 'CANDIDATE' or branch_taxid.endswith('CANDIDATE'):
                            branch_taxid = branch.split('_')[-3]
                    
                        if symbiont_ids and branch_taxid in symbiont_ids:
                            foreground_branches += 1
                            total_branches += 1
                            symbiont_branch_dnds.append(full_model_nonsyn_branch_lens[-1] / full_model_syn_branch_lens[-1])
                            symbiont_branch_dn.append(json_data['branch attributes']['0'][branch]['Full adaptive model (non-synonymous subs/site)'])
                            symbiont_branch_ds.append(json_data['branch attributes']['0'][branch]['Full adaptive model (synonymous subs/site)'])
                        else:
                            background_branches += 1
                            total_branches += 1
                            non_symbiont_branch_dnds.append(full_model_nonsyn_branch_lens[-1] / full_model_syn_branch_lens[-1])
                            non_symbiont_branch_dn.append(json_data['branch attributes']['0'][branch]['Full adaptive model (non-synonymous subs/site)'])
                            non_symbiont_branch_ds.append(json_data['branch attributes']['0'][branch]['Full adaptive model (synonymous subs/site)'])
                    except IndexError:
                        # internal node dont inlcude in the dnds calculation
                        pass
                                          
                    # values to store for branches with evidence of selection
                    if json_data['branch attributes']['0'][branch]['Corrected P-value'] < 0.05:
                        selection_pvalues += 1
                        sig_full_model_nonsyn_branch_lens.append(json_data['branch attributes']['0'][branch]['Full adaptive model (non-synonymous subs/site)'])
                        sig_full_model_syn_branch_lens.append(json_data['branch attributes']['0'][branch]['Full adaptive model (synonymous subs/site)'])

                # convert file name to samplename if id_map is provided
                if id_map_file:
                    refseq_accession = os.path.splitext(filename)[0].replace('_absrel', '')
                    try:
                        prot_id = id_map[refseq_accession]
                    except KeyError:
                        prot_id = refseq_accession
                
                # calculate fraction of test_branches in tree 
                if foreground_branches > 0:
                    test_fraction = foreground_branches / total_branches
                else:
                    test_fraction = 0
                    
                
                # add to data frame     
                data.append({
                    'query': prot_id,
                    'accession': refseq_accession,
                    'ns_per_site_avg': sum(full_model_nonsyn_branch_lens) / len(full_model_nonsyn_branch_lens) if full_model_nonsyn_branch_lens else None,
                    'syn_per_site_avg': sum(full_model_syn_branch_lens) / len(full_model_syn_branch_lens) if full_model_syn_branch_lens else None,
                    'dnds_tree_avg': sum(branch_dnds) / len(branch_dnds) if branch_dnds else None,
                    'symbiont_branch_dnds_avg': (sum(symbiont_branch_dn) / sum(symbiont_branch_ds)) if symbiont_branch_dnds else 0,
                    'symbiont_tree_dnds_avg': sum(symbiont_branch_dnds) / len(symbiont_branch_dnds) if symbiont_branch_dnds else None,
                    'non_symbiont_branch_dnds_avg': (sum(non_symbiont_branch_dn) / sum(non_symbiont_branch_ds)) / len(non_symbiont_branch_dnds) if non_symbiont_branch_dnds else 0,
                    'non_symbiont_tree_dnds_avg': sum(non_symbiont_branch_dnds) / len(non_symbiont_branch_dnds) if non_symbiont_branch_dnds else None,
                    'selection_branch_count': selection_pvalues,
                    'total_branch_length': sum(branch_lengths),
                    'avg_branch_length': sum(branch_lengths) / len(branch_lengths),
                    'selected_ns_per_site_avg': (sum(sig_full_model_nonsyn_branch_lens) / len(sig_full_model_nonsyn_branch_lens)) / len(json_data['branch attributes']['0']) if sig_full_model_nonsyn_branch_lens else None,
                    'selected_syn_per_site_avg': (sum(sig_full_model_syn_branch_lens) / len(sig_full_model_syn_branch_lens)) / len(json_data['branch attributes']['0']) if sig_full_model_syn_branch_lens else None,
                    'branch_fraction': selection_pvalues / total_branches if selection_pvalues > 1 else 0,
                    'branch_fraction_full_norm': (selection_pvalues / total_branches) / (sum(branch_lengths) / len(branch_lengths)) if sum(branch_lengths) > 0 else 0,
                    'test_fraction': test_fraction
                })
    
    # map accessions to queries where no succesful aBSREL run was generated      
    if id_map_file:
        all_prot_ids = set(id_map.values())
        found_prot_ids = set(item['query'] for item in data)
        missing_prot_ids = all_prot_ids - found_prot_ids
        for prot_id in missing_prot_ids:
            refseq_accession = [key for key, value in id_map.items() if value == prot_id][0]
            
            data.append({
                'query': prot_id,
                'accession': refseq_accession,
                'ns_per_site_avg': None,
                'syn_per_site_avg': None,
                'dnds_tree_avg': None,
                'symbiont_branch_dnds_avg': None,
                'symbiont_tree_dnds_avg': None,
                'non_symbiont_branch_dnds_avg': None,
                'non_symbiont_tree_dnds_avg': None,
                'selection_branch_count': None,
                'total_branch_length': None,
                'avg_branch_length': None,
                'selected_ns_per_site_avg': None,
                'selected_syn_per_site_avg': None,
                'branch_fraction': None,
                'branch_fraction_full_norm': None,
                'test_fraction': None
            })
            
    datatable = pd.DataFrame(data)
    return datatable

def parse_busted_results(directory, id_map_file):
    
    if id_map_file:
        id_map = load_id_map(id_map_file)
    
    data = []
    
     # parse directory of busted result json files
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            
            with open(filepath) as file:
                try:
                    json_data = json.load(file)
                # catching empty json files from halted or failed snakemake runs
                except json.JSONDecodeError:
                    continue
                
                # counter number of branches included in the foreground set
                number_of_foregound_branches = 0 
                for tested_node in json_data['tested']['0']:
                    if json_data['tested']['0'][tested_node] == 'test':
                        number_of_foregound_branches += 1
                    
                # convert seq id to uniprot id if id_map is provided
                refseq_accession = os.path.splitext(filename)[0].replace('_flagged.treefile_busted', '')
                try:
                    prot_id = id_map[refseq_accession]
                except KeyError:
                    prot_id = refseq_accession
                    
                # determine significance
                data.append({
                    'query': prot_id,
                    'foreground_branches': number_of_foregound_branches,
                    'p_value': json_data['test results']['p-value'],
                    'LRT': json_data['test results']['LRT']
                })
            
            
    datatable = pd.DataFrame(data)
    return datatable      

def parse_busted_ph_results(directory: str, id_map_file: str=None) -> pd.DataFrame:
    
    if id_map_file:
        id_map = load_id_map(id_map_file)
    
    data = []
    
     # parse directory of busted result json files
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            
            with open(filepath) as file:
                try:
                    json_data = json.load(file)
                # catching empty json files from halted or failed snakemake runs
                except json.JSONDecodeError:
                    continue
                
                # counter number of branches included in the foreground set
                number_of_foregound_branches = 0 
                number_of_backgound_branches = 0
                for tested_node in json_data['tested']['0']:
                    if json_data['tested']['0'][tested_node] == 'test':
                        number_of_foregound_branches += 1
                    elif json_data['tested']['0'][tested_node] == 'background':
                        number_of_backgound_branches += 1
                try:
                    test_ratio = number_of_foregound_branches / (number_of_backgound_branches + number_of_foregound_branches)
                    if test_ratio == 0:
                        print(f'No foreground: {filename}')
   
                except ZeroDivisionError:
                    print(f'No background: {filename}')
                    test_ratio = -1
                
                
                # convert seq id to uniprot id if id_map is provided
                refseq_accession = os.path.splitext(filename)[0].replace('_flagged.treefile_busted', '')
                if id_map_file:
                    try:
                        prot_id = id_map[refseq_accession]
                    except KeyError:
                        prot_id = refseq_accession
                else:
                    prot_id = refseq_accession
                        
                
                # determine significance
                data.append({
                    'query': prot_id,
                    'foreground_branches': number_of_foregound_branches,
                    'background_branches': number_of_backgound_branches,
                    'test_ratio': test_ratio,
                    'test_p_value': json_data['test results']['p-value'],
                    'test_LRT': json_data['test results']['LRT'],
                    'background_p_value': json_data['test results background']['p-value'],
                    'background_LRT': json_data['test results background']['LRT'],
                    'test_shared_p_value': json_data['test results shared distributions']['p-value'],
                    'test_shared_LRT': json_data['test results shared distributions']['LRT'],
                    'dnds_background': json_data['fits']['MG94xREV with separate rates for branch sets']['Rate Distributions']['non-synonymous/synonymous rate ratio for *background*'][0][0],
                    'dnds_foreground': json_data['fits']['MG94xREV with separate rates for branch sets']['Rate Distributions']['non-synonymous/synonymous rate ratio for *test*'][0][0]
                })
                
    datatable = pd.DataFrame(data)
    return datatable      

def main():
    parser = argparse.ArgumentParser(description='Parse hyphy JSON files')
    parser.add_argument('-d','--directory', help='Path to the directory containing JSON files')
    parser.add_argument('-m','--id_map', default=None, help='Path to the id mapping file from uniprot')
    parser.add_argument('-t','--test_type', help='Type of test')
    parser.add_argument('-s','--symbiont_ids', default=None, help='Path to the symbiont ids file')
    parser.add_argument('-o','--output', default='', help='Path to generate the output file')
    args = parser.parse_args()
    
    if args.symbiont_ids:
        with open(args.symbiont_ids) as file:
            symbiont_ids = file.read().splitlines()
        
    if args.test_type == 'absrel':
        parse_absrel_results(args.directory, args.id_map).to_csv(args.output + '/absrel_results.csv', index=False)

    if args.test_type == 'busted':
        parse_busted_results(args.directory, args.id_map).to_csv(args.output + '/busted_results.csv', index=False)
    
    if args.test_type == 'busted-ph':
        parse_busted_ph_results(args.directory, args.id_map).to_csv(args.output + '/busted-ph_results.csv', index=False)
    
    
if __name__ == '__main__':
    main()