from ete3 import Tree
import argparse
import subprocess
import json
import os

def get_scientific_name(taxid: str, retries: int) -> str: 
        
    """
    Retrieve the scientific name for a given taxonomy ID using the NCBI datasets tool.
    Args:
        taxid (str): The taxonomy ID for which to retrieve the scientific name.
        retries (int): The number of times to retry the query in case of transient errors.
    Returns:
        str: The scientific name corresponding to the given taxonomy ID, or None if the query fails after the specified number of retries.
    """
      
    retries = retries
    while retries > 0:
        result = subprocess.run(f"datasets summary taxonomy taxon {taxid}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        reports = result.stdout
        try:
            # process stdout into dictionary 
            reports = reports.replace('true', 'True')
            report_dict = eval(reports)
    
            # obtain scientific name from result dict 
            sci_name = report_dict["reports"][0]["taxonomy"]["current_scientific_name"]["name"]
            return sci_name
        # this will occur if the taxid does not return a valid result
        # which should only occur with transient errors in the dataset query 
        # so logging the error and retrying is appropriate
        except:
            retries -= 1
    return None
            
def rename_leaves(tree_file: str, msa_file: str, foreground_genomes: list, taxon_dict: dict, output_dir: str) -> None:
    
    # Load the tree
    try:
        tree = Tree(tree_file)
    except Exception as e:
        print(f"Error loading tree: {e}")
        return

    # Read the MSA file into a list of lines
    with open(msa_file, 'r') as msa:
        msa_lines = msa.readlines()

    # Rename the leaves
    for leaf in tree:
        
        # Get the taxid from the leaf name
        taxid = leaf.name.split("_")[-1]
        
        # Check if the taxid has been processed before
        if taxid in taxon_dict:
            sci_name = taxon_dict[taxid]
        else:  
            # Get the scientific name from the taxid
            sci_name = get_scientific_name(taxid, 3)
            taxon_dict[taxid] = sci_name
            
        # Check if the taxid is in the list of free-living tax IDs
        if taxid in foreground_genomes:
            sci_name = f"{sci_name}_HA"
        
        sci_name = sci_name.replace("(", "_")
        sci_name = sci_name.replace(")", "_")
        # Set the leaf name to the scientific name
        if sci_name:
            # Find the corresponding line in the MSA file and rename it
            for i, line in enumerate(msa_lines):
                if line.startswith(f">{leaf.name}"):
                    msa_lines[i] = f">{sci_name}\n"
                    break
            
            leaf.name = sci_name

    # Write the modified MSA file to a new file
    output_msa = output_dir + msa_file.split('/')[-1]
    with open(output_msa, 'w') as msa:
        msa.writelines(msa_lines)

    # Write the modified tree to a new file
    output_tree = output_dir + tree_file.split('/')[-1]
    tree.write(outfile=output_tree)

def count_leaves(tree_file: str) -> int:
    
    # Load the tree
    tree = Tree(tree_file)
    return len(tree)

def remove_leafs(tree_file: str, output_file: str, leafs: list) -> None:
    
    tree = Tree(tree_file)
    # Search for the leaf by name and branch length
    for node in tree.traverse():
        if node.is_leaf() and node.name in leafs:
            print("removing:", node.name)
            node.detach()
    
    # Prune the tree to remove any internal nodes that no longer have any leaves
    leaf_names = [leaf.name for leaf in tree if leaf.is_leaf() and leaf.name]
    tree.prune(leaf_names)

    # Write the modified tree to a new file
    tree.write(outfile=output_file)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Rename leaves of a phylogenetic tree.")
    parser.add_argument("-t", "--tree_file", type=str, help="Path to single input tree file")
    parser.add_argument("-o", "--output_dir", type=str, help="Path to the output dir")
    parser.add_argument("-m", "--msa_file", type=str, help="Path to the input multiple sequence alignment file")
    parser.add_argument("-f", "--foreground_tax_ids", type=str, help="Path to the file containing the free-living tax IDs")
    parser.add_argument("-l", "--accession_list", type=str, help="Path to the file containing the accession list for multiple tree files")
    parser.add_argument("-p", "--prefix", type=str, help="Path to the evorate workdlow output dir")
    parser.add_argument("-d", "--taxon_dict", type=str, help="json file for storing taxid to scientific name mapping")
    
    # Parse arguments
    args = parser.parse_args()
    expirement_prefix = args.prefix

    # Load the list of foreground tax IDs
    with open(args.foreground_tax_ids, 'r') as f:
        foreground_tax_ids = f.read().splitlines()
       
    # Check if taxon dict exists 
    if not os.path.exists(args.taxon_dict):
        taxon_dict = {}
    else:
        # Load taxon dict
        with open(args.taxon_dict, 'r') as f:
            taxon_dict = json.load(f)
            
    # Rename the leaves of the tree
    with open(args.accession_list, 'r') as f:
        for line in f:
            line = line.strip()
            tree_file = f"{expirement_prefix}/tree_files/{line}/{line}.treefile"
            msa_file = f"{expirement_prefix}/msa_files/{line}/{line}_final_align_AA.aln"
            rename_leaves(tree_file, msa_file, foreground_tax_ids, taxon_dict, args.output_dir)
    
    # Save the taxon dict for future use
    with open(args.taxon_dict, 'w') as f:
        json.dump(taxon_dict, f)
        
if __name__ == "__main__":
    main()