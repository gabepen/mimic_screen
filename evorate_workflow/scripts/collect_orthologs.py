import argparse
import os
from tqdm import tqdm 
import zipfile 
import shutil
import subprocess
from loguru import logger
from Bio import SeqIO
import json


def shorten_sequence_names(fasta_file, max_length):
    
    # Parse the FASTA file
    sequences = SeqIO.parse(fasta_file, "fasta")

    # Create a new list to store the modified sequences
    modified_sequences = []

    # Iterate over each sequence in the file shorten descriptions
    for seq in sequences:
        seq.description = seq.description[:max_length]

        # Add the modified sequence to the list
        modified_sequences.append(seq)

    # Write the modified sequences back to the FASTA file
    SeqIO.write(modified_sequences, fasta_file, "fasta")

def download_gene_data_package(query_dict, taxon_dict, accession_dict, workdir, retries=3):
    
    # parse each taxon key to generate the download command 
    issue_taxa = []
    for taxid in taxon_dict:
        
        if taxid != '2774015':
            continue
        
        # get the query accessions for the taxon
        accessions = taxon_dict[taxid]
        
        # log
        logger.info(f"Downloading ortholog sequences for taxon: {taxid}")
        logger.info(f"Orthologs: {accessions}")
        
        # format download command 
        output_path = f"{workdir}/orthologs/{taxid}_orthologs_dataset.zip"
        ncbi_datasets_command = f"datasets download gene accession {accessions} --include gene --filename {output_path} --taxon-filter {taxid}"
        
        # retry loop for downloading ortholog sequences
        valid_zip = False
        while retries > 0:
            try:
            
                # run downlaod command
                result = subprocess.run(ncbi_datasets_command, shell=True, capture_output=True, text=True)
                error_output = result.stderr
                
                # extract protein sequence fasta  
                zip_path = os.path.join(output_path)
                extract_dir = os.path.splitext(zip_path)[0]
                packet_fasta = f"{workdir}/orthologs/{taxid}"
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extract('ncbi_dataset/data/gene.fna',path=packet_fasta)
                    
                # remove packet archive 
                os.remove(os.path.join(output_path))
                
                # if no errors break out of retry loop
                valid_zip = True
                break 
            
            except FileNotFoundError:
                
                retries -= 1
                
                # no gene data for taxon, no point in retrying
                if error_output.startswith("Error: The taxonomy ID"):
                    logger.error(f"\t{error_output.split('\n')[0]}")
                    break
                
                # other error, log and retry 
                logger.error(f"\t{error_output}")
                logger.info(f"\tRetrying...")
        
        # check if a valid zip archive was downloaded for the packet      
        if valid_zip:
            
            # move packet multiseq file up two directories
            shutil.move(os.path.join(packet_fasta, "ncbi_dataset/data/gene.fna"), packet_fasta)
            
            # downloading by WP accession obtains all gene records annotated across all genomes
            # there could be multiple chromosome accessions for a single genome
            refseq_chr_accessions = accession_dict[taxid]['refseq_chr_accessions'] 
          
            # append sequences to appropriate fasta file
            gene_fa_file = os.path.join(packet_fasta, "gene.fna")
            with open(gene_fa_file, 'r') as f:
                matched = False
                for line in f:
                    
                    # identify the accession number connected to the nucleotide sequence
                    if line.startswith('>'):
                        
                        # check for annotation match 
                        annotation_accession = line.split('>')[-1].split(':')[0]
                        if annotation_accession in refseq_chr_accessions:
                            
                            # valid annotation accession now get the WP accession to append to the correct fasta file
                            wp_accession = line.split('protein_accession=')[1].split(']')[0]
                            target_fasta = f"{workdir}/msa_files/{wp_accession}/ortholog_seqs.fna"

                            # append sequences to multiseq fasta file  
                            with open(target_fasta, 'a+') as m:
                                m.write(line)
                                m.write(next(f))
                                matched = True
                            break
                        
                # thanks ncbi... (this shouldnt happen)
                if not matched:
                    scratch_path = f"{workdir}/orthologs/{taxid}_debug_orthologs_dataset.zip"
                    logger.error(f"Error: No matching accession found for taxid: {taxid}")
                    logger.info(f"Testing single accession error message:")
                    debug_datasets_command = f"datasets download gene accession {accessions[0]} --include gene --filename {output_path} --taxon-filter {taxid}"
                     # run downlaod command
                    result = subprocess.run(debug_datasets_command, shell=True, capture_output=True, text=True)
                    error_output = result.stderr
                    logger.error(f"\t{error_output}")
                    issue_taxa.append(taxid)
                    
                
            # delete the empty ncbi_dataset directory
            shutil.rmtree(os.path.join(target_fasta, "ncbi_dataset"))
    
    logger.info(f"Taxa issues ({len(issue_taxa)}): {issue_taxa}")
    
def collect_ortholog_accesssions(query_id, candidate_list, workdir):
    
    os.makedirs(workdir + '/orthologs', exist_ok=True)
    os.makedirs(workdir + '/msa_files', exist_ok=True)
    
    # dictionaries for sorting accessions 
    query_dict = {}
    taxon_dict = {}
    
    for file in os.listdir(workdir + '/rbh_results'):
        if file.endswith(".tsv"):
            
            # get target organism tax id
            taxid = file.split('_')[0]
            
            # prep taxon dict 
            taxon_dict[taxid] = []
            
            # Open the rbh file for the query sequence
            with open(os.path.join(workdir, 'rbh_results', file), 'r') as f:
                
                # Parse the file line by line
                for line in f:
                    
                    # Split the line into fields
                    fields = line.strip().split()
                    
                    # Write out rbh hits to the query accession number
                    if fields[0] in candidate_list:
                        
                        # map candidate accession to its orthologs 
                        if fields[0] not in query_dict:
                            query_dict[fields[0]] = [fields[1]]
                            
                            # create candidate folder for storing gene seqs 
                            os.makedirs(workdir + '/msa_files/' + fields[0], exist_ok=True)
                                                    
                        else:
                            query_dict[fields[0]].append(fields[1])
                        
                        # map taxid to its orthologs 
                        taxon_dict[taxid].append(fields[1])

    return query_dict, taxon_dict
   

def count_orthologs(workdir):
    
    # Create a dictionary to store the counts of orthologs for each genome
    ortholog_counts = {}

    # Iterate over the RBH result files
    for file in os.listdir(workdir + '/rbh_results'):
        if file.endswith(".tsv"):
            
            # Extract the taxid and genome accession from the file name
            taxid = file.split('_')[1]
            genome_accession = file.split('_')[3].split('.')[0]
            
            # Read the RBH result file and count the number of lines
            with open(os.path.join(workdir, 'rbh_results', file), 'r') as f:
                ortholog_count = sum(1 for line in f)
            
            # Store the count in the dictionary
            ortholog_counts[genome_accession] = ortholog_count

    # Create the completion marking file
    with open(os.path.join(workdir, 'collected_orthologs.txt'), 'w') as f:
        # Write the counts to the file
        for genome_accession, count in ortholog_counts.items():
            f.write(f"{taxid}\t{genome_accession}\t{count}\n")
            
def main():
    
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", "--query_id", help="ID of the query microbe")
    parser.add_argument("-s", "--query_sequence", help="Specific query sequence from the proteome")
    parser.add_argument("-c", "--candidate_list", help="List of candidate accessions to search for orthologs")
    parser.add_argument("-w", "--workdir", help="Path to the working directory")
    parser.add_argument("-j", "--genome_record_json", help="Path to the genome record JSON file for mapping taxids to GCF accessions")
    parser.add_argument("-l", "--log", default='logs/', help="Path to the log file")
    args = parser.parse_args()
    
    # initialize logging 
    logger.remove()  # Remove default handler
    logger.add(f"{args.log}/orthologs_collection_debug.log", rotation="500 MB") # Add a log file handler
    
    # load candidate accessions
    with open(args.candidate_list, 'r') as f:
        candidate_list = f.read().splitlines() 
    
    logger.info(f"Collecting {len(candidate_list)} ortholog datasets for {args.query_id}...")
    
    # collect ortholog accessions for the query sequence
    query_dict, taxon_dict = collect_ortholog_accesssions(args.query_id, candidate_list, args.workdir)
   
    # load genome record json
    with open(args.genome_record_json, 'r') as f:
        genome_accession_dict = json.load(f)
        
    # download
    download_gene_data_package(query_dict, taxon_dict, genome_accession_dict, args.workdir)
              
    # shorten sequence names to prevent hyphy post-msa.bf errors when seq names are too long
    for root, dirs, files in os.walk(args.workdir + '/msa_files'):
        for file in files:
            if file.endswith(".fna"):
                shorten_sequence_names(os.path.join(root, file), 20)
    
    # collect ortholog sequences into a single fasta file 
    #ortholog_seq_fasta = collect_ortholog_seqs(args.workdir, args.query_sequence)
    

if __name__ == "__main__":
    main()