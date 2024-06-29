import subprocess
import argparse
from tqdm import tqdm
import os
import zipfile
import shutil
import random
from loguru import logger
import datetime

def taxonkit_get_subtree(taxid):

    '''Uses taxonkit to generate a list of subspecies for a given taxid'''
    
    taxonkit_command = f"taxonkit list --ids {taxid} --indent ''"
    result = subprocess.run(taxonkit_command, shell=True, capture_output=True)
    output = result.stdout.decode().splitlines()
    logger.info(f"Subtree taxid: {taxid} Size: {len(output)}")
    
    return output[1:]

def download_genomes(id_list, max_genome_count, workdir):
    
    '''uses ncbi datasets to download genomes for a list of taxids'''
    
    # Create a directory to store the genomes
    os.makedirs(workdir + '/genomes', exist_ok=True)
    
    # shuffle taxid list 
    random.shuffle(id_list)
    
    # Log download command 
    download_command = [
        "datasets", "download", "genome", "taxon", "taxid",
        "--filename", f"workdir/genomes/taxid_dataset.zip",
        "--include", "gff3,protein", "--annotated",
        "--assembly-source", "RefSeq", "--assembly-level", "complete",
        "--assembly-version", "latest", "--released-after", "01/01/2016",
    ]
    download_command = ' '.join(download_command)
    logger.info("ncbi datasets download command:" + download_command)
        
    
    # Download the genomes
    potential_dl_count = min(max_genome_count, len(id_list))
    max_genome_count = min(max_genome_count, len(id_list))
    genomes_selected = []
    for taxid in tqdm(id_list):
        
        # Break if we have reached the maximum number of genomes
        if max_genome_count == 0:
            break
        
        if not taxid:
            continue
        
        # Set output file path
        o_file = f"{workdir}/{taxid}_dataset.zip"
        
        # skip already downloaded genomes
        if os.path.exists(o_file):
            max_genome_count += 1
            continue
        
        # run dataset download command 
        download_command = [
            "datasets", "download", "genome", "taxon", str(taxid),
            "--filename", f"{workdir}/genomes/{taxid}_dataset.zip",
            "--include", "gff3,protein", "--annotated",
            "--assembly-source", "RefSeq", "--assembly-level", "complete",
            "--assembly-version", "latest", "--released-after", "01/01/2016",
        ]
        download_command = ' '.join(download_command)
        
        # Run the download command and capture the output
        dl_output = subprocess.run(download_command, shell=True, capture_output=True)
        
        # Check the return code of the subprocess to determine if a download was successful
        if dl_output.returncode == 0:
            
            # logging output 
            dl_error_message = dl_output.stderr.decode().split('\n')
            dl_error_message = dl_error_message[1].strip().split('[')[0]
            logger.info(f"{dl_error_message} for taxid {taxid}")
            
            # mark a succesful genome download 
            max_genome_count -= 1
            
            # save taxid 
            genomes_selected.append(taxid)
            
        else:
            dl_error_message = dl_output.stderr.decode().split('\n')
            dl_error_message = dl_error_message[1].strip().replace('Error:', '')
            logger.warning(f"Failed to download genome for taxid {taxid}, {dl_error_message}")
            continue

    # Log the number of genomes downloaded
    logger.info(f"Downloaded {potential_dl_count - max_genome_count} genomes of {potential_dl_count} requested")
    
    return genomes_selected

def main():

    '''This script preforms an RBH search of a set of taxa to identify orthologs
       of a query sequence from each proteome in the taxa group.
    '''
    
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="RBH blastp analysis for identifying orthologs")
    parser.add_argument("-p", "--query_proteome", help="Path to the query proteome file")
    parser.add_argument("-i", "--query_id", help="ID of the query microbe")
    parser.add_argument("-x", "--taxid", type=int, help="Taxonomy ID for the species of interest")
    parser.add_argument("-w", "--workdir", default="work", help="Path to the working directory")
    parser.add_argument("-t", "--threads", type=int, help="Number of threads for mmseqs to use")
    parser.add_argument("-m", "--max_genome_count", default=300, type=int, help="max number of genomess to download")
    parser.add_argument("-l", "--log", default="logs/", help="log file name")
    args = parser.parse_args()

    # initialize logging 
    logger.remove()  # Remove default handler
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = args.log + f"ncbi_genome_download_{timestamp}.log"
    logger.add(log_file, rotation="500 MB") # Add a log file handler
    
    # make workdir
    os.makedirs(args.workdir, exist_ok=True)
    
    # Collect genome fastas for the taxid group
    id_list = taxonkit_get_subtree(args.taxid)
    genomes_selected = download_genomes(id_list, args.max_genome_count, args.workdir)
    
    # Save list of genomes used for downstream analysis 
    with(open(f"{args.log}/genomes_selected.txt", 'w')) as f:
        for taxid in genomes_selected:
            f.write(f"{taxid}\n")
    
if __name__ == "__main__":
    main()