from ete3 import Tree
import argparse
import os
import json
from tqdm import tqdm
import sys

def flag_nodes(treefile, fg_id_list):
    
    # load tree
    tree = Tree(treefile)
    # print tree nodes
    for node in tree.traverse():
        if node.is_leaf():
            try:
                tax_id = node.name.split('_')[-2]
                if tax_id in fg_id_list or tax_id == 'CANDIDATE':
                # flag the node with the FG flag for BUSTED 
                    node.name += '{FG}'
                else:
                    node.name += '{BG}'

            except IndexError:
                print(node.name)
                print('Error in parsing tax id')
            
            
        
    # save the modified tree to a new file
    output_treefile = os.path.splitext(treefile)[0] + '_flagged.treefile'
    tree.write(format=1, outfile=output_treefile)

def generate_id_list(genome_selected_file: str, results_dir: str) -> (list, list):
    
    fg_ids = []
    bg_ids = []
    # read the genome selected file to get each tax id
    with open(genome_selected_file) as file:
        for line in file:
        # find the corresponding results folder for macsyfinder
            tax_id = line.strip()
            results_folder = os.path.join(results_dir, tax_id)   
    
            # check the best solution file for systems
            best_solution_tsv = results_folder + '/best_solution.tsv'
            try:
                with open(best_solution_tsv, 'r') as f:
                    lines = f.readlines()
                    if "# No Systems found" in lines[3]:
                        bg_ids.append(tax_id)
                    else:
                        fg_ids.append(tax_id)
            except FileNotFoundError:
                continue
    
    return fg_ids, bg_ids
   
    

def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument('-i','--id_file', help='Path to the symbiont id file')
    parser.add_argument('-t','--tree_file_dir', default=None, help='Path to the newick tree')
    parser.add_argument('-g','--genome_selected_file', default=None, help='Path to the genome selected file')
    parser.add_argument('-r','--results_dir', default=None, help='Path to the results directory')
    parser.add_argument('-f','--foreground_ids', default=None, help='Path to the foreground ids file, all other ids will be considered background')
    args = parser.parse_args()

    if args.id_file :
        with open(args.id_file) as file:
            symbiont_ids = file.read().splitlines()
    
    if args.genome_selected_file and not args.foreground_ids:
        fg_ids, bg_ids = generate_id_list(args.genome_selected_file, args.results_dir)
    elif args.foreground_ids:
        with open(args.foreground_ids) as file:
            fg_ids = file.read().splitlines()

    for sample_dir in os.listdir(args.tree_file_dir):
        tree_file_path = os.path.join(args.tree_file_dir, sample_dir, sample_dir + '.treefile')
        if os.path.exists(tree_file_path):
            flag_nodes(tree_file_path, fg_ids)

if __name__ == '__main__':
    main()
    
    
        