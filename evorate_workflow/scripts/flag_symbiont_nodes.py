from ete3 import Tree
import argparse
import os
import json
from tqdm import tqdm

def flag_nodes(treefile, id_list):

    # load tree
    tree = Tree(treefile)
    # print tree nodes
    for node in tree.traverse():
        if node.is_leaf():
            try:
                tax_id = node.name.split('_')[-2]
                if tax_id in id_list:
                # flag the node with the FG flag for BUSTED 
                    node.name += '{FG}'

            except IndexError:
                print(node.name)
                input()
            
            
        
    # save the modified tree to a new file
    output_treefile = os.path.splitext(treefile)[0] + '_flagged.treefile'
    tree.write(format=1, outfile=output_treefile)
    

def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument('-i','--id_file', help='Path to the symbiont id file')
    parser.add_argument('-t','--tree_file_dir', default=None, help='Path to the newick tree')
    args = parser.parse_args()

    with open(args.id_file) as file:
        symbiont_ids = file.read().splitlines()
    
    for sample_dir in os.listdir(args.tree_file_dir):
        tree_file_path = os.path.join(args.tree_file_dir, sample_dir, sample_dir + '.treefile')
        if os.path.exists(tree_file_path):
            flag_nodes(tree_file_path, symbiont_ids)

if __name__ == '__main__':
    main()
    
    
        