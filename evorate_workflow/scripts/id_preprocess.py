import argparse
import subprocess
from tqdm import tqdm

def match_ncbi_ids(np_id):
    
    '''
    Retrieves the WP ID from the NCBI RefSeq database for a given NP ID.
    
    Parameters:
        np_id (str): The NP ID to match.
    
    Returns:
        str: The WP ID corresponding to the given NP ID.
    '''
    
    eutil_command = f"esummary -db protein -id {np_id} | grep WP_"
    output = subprocess.run(eutil_command, shell=True, capture_output=True, text=True)
    
    # split xml summary lines into list
    lines = output.stdout.split('\n')
    for l in lines:
        if 'WP_' in l:
            
            # this collects the wp_id that had been marked as ReplacedBy in the xml 
            wp_id = l.split('>')[1].split('<')[0]
            if wp_id:
                return wp_id

    # no wp_id was found return None
    return None
    
    
    
def np_to_wp_accessions(file_path, output_path):
    
    '''
    Identify uniprot accessions with only a NP_# accession mapped.
    
    Args:
        file_path (str): Path to the TSV file.
        output_path (str): Path to the output file.
        
    Returns:
        None

    '''
    unique_lines = {}
    duplicate_lines = {}

    with open(file_path, 'r') as file:
        
        # skip header 
        file.readline()
        
        # parse each line
        for line in file:
            line = line.strip()
            if line:
                columns = line.split('\t')
                uniprot_id = columns[0]
                refseq_acc_num = columns[1]
                
                # check uniprot ID for duplicates
                if uniprot_id in unique_lines:
                    duplicate_lines[uniprot_id] = [refseq_acc_num]
                    duplicate_lines[uniprot_id].append(unique_lines[uniprot_id])
                    del unique_lines[uniprot_id]
                else:
                    unique_lines[uniprot_id] = refseq_acc_num

    # write out to new file
    with open(output_path, 'w') as output:
         
        # for unique uniprot ids try to map NP_#s to identical WP_# record 
        for prot_id in tqdm(unique_lines):
            unique_accession = unique_lines[prot_id]
            if unique_accession.startswith('NP_'): 
                wp_match = match_ncbi_ids(unique_accession)
                
                # write out successful matches
                if wp_match:
                    output.write(prot_id + '\t' + wp_match + '\n')
            
            # write out unique WP_# records
            elif unique_accession.startswith('WP_'):
                output.write(prot_id + '\t' + unique_accession + '\n') 
            
        # write out all other lines that have WP_# accessions 
        for prot_id in duplicate_lines:
            accessions = duplicate_lines[prot_id]
            for acc_id in accessions:
                if acc_id.startswith('WP_'):
                    output.write(prot_id + '\t' + acc_id + '\n')
                
            
                

def main():
    '''
    This script is used to parse the output of a Uniprot ID mapping to the RefSeq protein database.
    It will identify UniProt accessions that are only mapped to a NP_ accession and determine the identical
    WP_ record. NP_ records cannot be used in the workflow as they are not used in reference proteomes files 
    found in NCBI Datasets.
    
    Leave the header added to the file by the Uniprot ID mapping.
    
    Returns a new id_map file for later use with alignment_analysis.py script
    '''
    
    parser = argparse.ArgumentParser(description='Identify lines with unique first column values in a TSV file')
    parser.add_argument('file_path', type=str, help='Path to the TSV file')
    args = parser.parse_args()

    # output file path 
    output_path = args.file_path.split('.')[0] + '_NPtoWP.tsv'
    np_to_wp_accessions(args.file_path, output_path)

if __name__ == '__main__':
    main()
    