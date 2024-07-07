import subprocess
import argparse
from tqdm import tqdm
import os
import zipfile
import shutil
import random
from loguru import logger
import datetime
import json

def taxonkit_get_subtree(taxid):

    '''Uses taxonkit to generate a list of subspecies for a given taxid'''
    
    taxonkit_command = f"taxonkit list --ids {taxid} --indent '' -r"
    result = subprocess.run(taxonkit_command, shell=True, capture_output=True)
    output = result.stdout.decode().splitlines()
    
    
    # collect only taxids of rank = species
    species_taxids = []
    for tid in output:
        if tid != '' and tid.split()[1] == '[species]':
            species_taxids.append(tid.split()[0])
        

    # log number collected
    logger.info(f"Subtree taxid: {taxid} Size: {len(species_taxids)}")    
    return species_taxids

def download_genomes(id_list, max_genome_count, workdir):
    
    '''uses ncbi datasets to download genomes for a list of taxids'''
    
    # Create a directory to store the genomes
    os.makedirs(workdir + '/genomes', exist_ok=True)
    
    # dictionary for mapping taxids to genome records
    taxid_genome_map = {}
    def map_id_to_genome(taxid, archive_path):
        
        ''' nested to be called for both existing archives and new archives'''
        # Open the zip archive
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            
            # Get the list of files in the archive
            file_list = zip_ref.namelist()
            
            # add first GCF accession to the map
            for file_name in file_list:
                if file_name.endswith('protein.faa'):
                    gcf_accession = file_name.split('/')[-2]
                    break                 
            
            # parse seq_report
            refseq_chr_accessions = []
            
            # get the sequence report file
            seq_report_path = [file_name for file_name in file_list if file_name.endswith('sequence_report.jsonl')][0]
            with zip_ref.open(seq_report_path) as seq_report_file:
                
                # its a jsonl file load each into a dict
                for line in seq_report_file:
                    record = json.loads(line)
                    
                    # need to accomadate for the fact that some records may have multiple chromosomes
                    if record['assignedMoleculeLocationType'] == 'Chromosome':
                        refseq_chr_accessions.append(record['refseqAccession'])
                        
            # add to the map
            taxid_genome_map[taxid] = {'gcf_accession':gcf_accession, 'refseq_chr_accessions':refseq_chr_accessions}    
            
    # shuffle taxid list 
    random.shuffle(id_list)
    
    # Log download command 
    download_command = [
        "datasets", "download", "genome", "taxon", "taxid",
        "--filename", f"workdir/genomes/taxid_dataset.zip",
        "--include", "seq-report,protein", "--annotated",
        "--assembly-source", "RefSeq", "--assembly-level", "complete",
        "--assembly-version", "latest", "--released-after", "01/01/2016",
        "--reference"
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
        o_file = f"{workdir}/genomes/{taxid}_dataset.zip"
        
        # skip already downloaded genomes
        if os.path.exists(o_file):
            max_genome_count += 1
            map_id_to_genome(taxid, o_file)
            continue
        
        # run dataset download command 
        download_command = [
            "datasets", "download", "genome", "taxon", str(taxid),
            "--filename", f"{workdir}/genomes/{taxid}_dataset.zip",
            "--include", "seq-report,protein", "--annotated",
            "--assembly-source", "RefSeq", "--assembly-level", "complete",
            "--assembly-version", "latest", "--released-after", "01/01/2016",
            "--reference"
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
            
            # save taxid - genome record map
            map_id_to_genome(taxid, o_file) 
            
        else:
            dl_error_message = dl_output.stderr.decode().split('\n')
            #dl_error_message = dl_error_message[1].strip().replace('Error:', '')
            logger.warning(f"Failed to download genome for taxid {taxid}, {dl_error_message}")
            continue

    # Log the number of genomes downloaded
    logger.info(f"Downloaded {potential_dl_count - max_genome_count} genomes of {potential_dl_count} requested")
    
    return genomes_selected, taxid_genome_map

def main():

    '''This script preforms an RBH search of a set of taxa to identify orthologs
       of a query sequence from each proteome in the taxa group.
    '''
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Taxon kit subtree download of random species RefSeq genome records")
    parser.add_argument("-p", "--query_proteome", help="Path to the query proteome file")
    parser.add_argument("-i", "--query_id", help="ID of the query microbe")
    parser.add_argument("-x", "--taxid", type=int, help="Taxonomy ID for the species of interest")
    parser.add_argument("-w", "--workdir", default="work", help="Path to the working directory")
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
    genomes_selected, genome_accession_map = download_genomes(id_list, args.max_genome_count, args.workdir)
    
    # Save genome_accession_map to a json file 
    with open(f"{args.workdir}/genomes/genome_accession_map.json", 'w') as f:
        json.dump(genome_accession_map, f)
    
    # Save list of genomes used for downstream analysis 
    with(open(f"{args.log}/genomes_selected.txt", 'w')) as f:
        for taxid in genomes_selected:
            f.write(f"{taxid}\n")
    
if __name__ == "__main__":
    main()