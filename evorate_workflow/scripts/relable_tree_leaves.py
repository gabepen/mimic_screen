from ete3 import Tree
import argparse
import subprocess

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
            
def rename_leaves(tree_file: str, msa_file: str, output_file: str) -> None:
    # Load the tree
    tree = Tree(tree_file)

    # Read the MSA file into a list of lines
    with open(msa_file, 'r') as msa:
        msa_lines = msa.readlines()

    # Rename the leaves
    for leaf in tree:
        # Get the taxid from the leaf name
        taxid = leaf.name.split("_")[-1]
        # Get the scientific name from the taxid
        sci_name = get_scientific_name(taxid, 3)
        # Set the leaf name to the scientific name
        if sci_name:
           
            # Find the corresponding line in the MSA file and rename it
            for i, line in enumerate(msa_lines):
                if line.startswith(f">{leaf.name}"):
                    msa_lines[i] = f">{sci_name}\n"
                    break
            
            leaf.name = sci_name

    # Write the modified MSA file to a new file
    output_msa = output_file.replace(".treefile", ".fasta")
    with open(output_msa, 'w') as msa:
        msa.writelines(msa_lines)

    # Write the modified tree to a new file
    tree.write(outfile=output_file)

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
    parser.add_argument("tree_file", type=str, help="Path to the input tree file")
    parser.add_argument("output_file", type=str, help="Path to the output tree file")
   # parser.add_argument("msa_file", type=str, help="Path to the input multiple sequence alignment file")
    # Parse arguments
    args = parser.parse_args()
    leafs_to_remove = ['Gallaecimonas mangrovi', 'Vibrio campbellii','Vibrio harveyi', 'Oceanimonas pelagia']
    remove_leafs(args.tree_file, args.output_file, leafs_to_remove)
    
if __name__ == "__main__":
    main()
