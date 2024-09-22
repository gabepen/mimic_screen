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
import multiprocessing as mp
from queue import Queue
from functools import partial
import time
import urllib.request
import globi_db_queries

def taxonkit_get_subtree(taxid: str) -> list:

    '''Uses taxonkit to generate a list of subspecies for a given taxid'''
    taxonkit_command = f"taxonkit list --ids {taxid} --indent '' -r"
    result = subprocess.run(taxonkit_command, shell=True, capture_output=True)
    output = result.stdout.decode().splitlines()
    
    # collect only taxids of rank = species
    species_taxids = []
    for tid in output:
        if tid != '' and tid.split()[1] == '[species]':
            species_taxids.append(tid.split()[0])
    
    logger.info(f"Number of species in subtree for taxid {taxid} is {len(species_taxids)}")
        
    return species_taxids 

def get_scientific_name(taxid: str, retries: int) -> str:     
    
        
    retries = retries
    while retries > 0:
        result = subprocess.run(f"datasets summary taxonomy taxon {taxid}", shell=True, capture_output=True, text=True)
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
            logger.warning(f"Symbiosis check for {taxid} error, {result}")
            retries -= 1

def check_sciname_for_symbiosis(sci_name: str) -> bool:
    
    # both conditions are clear indications of a symbiotic taxa
    if 'symbiont' in sci_name.lower():
        return True 
    elif 'candidatus' in sci_name.lower():
        return True 
    else:
        return False

def validate_globi_results(taxid: str, rows: list):
    
    '''Parses the rows returned in from a GloBI query to determine if a valid interaction has been observed'''
    
    # initialize citation logger 
    logger_citations = logger.bind(task="citations")
    
    # no interaction results found
    if len(rows) == 0:
        return False

    # interactions found, check citations 
    found_interactions = set()
    for r in rows:
        
        # only save citations for unique interactions
        if (r[38],r[42]) not in target_organisms:
            ref_citation = r[84]
            source_citation = r[85]
            logger_citations.info(f"{taxid} | {r[38]} | {r[42]} | {ref_citation} | {source_citation}")
            found_interactions.add((r[38], r[42]))
    
    return True 
    
    
def dataset_download(dl_command, taxid, taxon_genome_map,
                     success_queue, fg_genomes_selected, 
                     bg_genomes_selected, max_genome_count, 
                     max_genome_event, globi_db_path):
    
    
    def add_taxa_to_selected_group(taxid: str, selected_list: list, list_name: str) -> bool:  
        
        """
        Add a taxid to the selected group if the number of selected genomes is less than half of the max genome count.
        Parameters:
        - taxid (str): The taxid to be added to the selected group.
        - selected_list (list): The list of selected taxids.
        - list_name (str): The name of the list for logging purposes.
        Returns:
        - bool: True if the taxid was successfully added, False otherwise.
        """
        
        # save taxid if forground genomes are not over half of the max genome count
        if len(selected_list) < int(max_genome_count / 2):
            logger.info(f"Adding taxid {taxid} to {list_name} genomes")
            selected_list.append(taxid)
            # increment the success count
            success_queue.put(1)
            return True 
        else:
            logger.info(f"Max {list_name} genomes reached skipping taxid {taxid}")
            return False
      
    def map_id_to_genome(taxid: str, archive_path: str) -> dict:
        
        """
        Maps a taxid to the corresponding GCF accession and refseq chromosome accessions.
        Args:
            taxid (str): The taxid to map.
            archive_path (str): The path to the zip archive containing the genome files.
        Returns:
            dict: A dictionary containing the mapped GCF accession and refseq chromosome accessions.
        Raises:
            FileNotFoundError: If the zip archive specified by archive_path does not exist.
            IndexError: If the zip archive does not contain a sequence_report.jsonl file.
        """
        
        
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
            logger.info(f"taxid {taxid} was mapped to GCF accession {gcf_accession} and refseq chromosome accessions {refseq_chr_accessions}") 
            return {'gcf_accession':gcf_accession, 'refseq_chr_accessions':refseq_chr_accessions}    
    
    # set up the dl command 
    o_file = dl_command[dl_command.index('--filename') + 1]
    dl_command = ' '.join(dl_command)
    
    # Run the download command and capture the output
    dl_output = subprocess.run(dl_command, shell=True, capture_output=True)
    
    # Check the return code of the subprocess to determine if a download was successful
    if dl_output.returncode == 0:
        
        # logging output 
        dl_error_message = dl_output.stderr.decode().split('\n')
        dl_error_message = dl_error_message[1].strip().split('[')[0]
        logger.info(f"{dl_error_message} for taxid {taxid}")
            
        # check for biological interactions using GloBI database 
        
        # GloBI does not seem to include 'X endosymbiont of Y' or 'Candidatus X' in their taxa names
        # so we will check the taxid name for these terms which indicate expiremental confidence of symbiosis 
        
        sci_name = get_scientific_name(taxid, 3)
        
        logger.info(f"Taxid {taxid} scientific name is {sci_name}")
        if check_sciname_for_symbiosis(sci_name):
            logger.info(f"Taxid {taxid} is a symbiont based on scientific name")
            
            # add to the foreground genomes if possible
            if not add_taxa_to_selected_group(taxid, fg_genomes_selected, 'foreground'):
                # if the max fg genomes have been reached delete the archive and return
                os.remove(o_file)
                return

        # If the scientific name does not indicate symbiosis, check GloBI database for interactions
        query = {
            'sourceTaxonName': sci_name, 
            'interactionTypeName': ['parasiteOf','hasHost','pathogenOf']
        }
        results = globi_db_queries.multi_column_search('interactions', query, globi_db_path)
        if validate_globi_results(taxid, results):
            
            logger.info(f"Taxid {taxid} is a symbiont/pathogen based on GloBI results")
            # add to the foreground genomes if possible
            if not add_taxa_to_selected_group(taxid, fg_genomes_selected, 'foreground'):
                # if the max fg genomes have been reached delete the archive and return
                os.remove(o_file)
                return
        
        # Need to verify freeliving through isolation source 
        
        # scientific name and GloBI results do not indicate symbiosis add to background genomes with confidence
        else:  
            
            logger.info(f"Taxid {taxid} is not a symbiont based on GloBI results")
            if not add_taxa_to_selected_group(taxid, bg_genomes_selected, 'background'):
                # if the max bg genomes have been reached delete the archive and return
                os.remove(o_file)
                return
        
        # taxid was succesfully added to a selected list, add to the taxon_genome_map
        taxon_genome_map[taxid] = map_id_to_genome(taxid, o_file) 
        
        # monitor success queue to check if max_genome_count has been reached
        if success_queue.qsize() >= max_genome_count:
            max_genome_event.set()
            logger.info('Max genome number reached')
        
        
    else:
        dl_error_message = dl_output.stderr.decode().split('\n')
        dl_error_message = dl_error_message[1].strip().replace('Error:', '')
        #logger.warning(f"Failed to download genome for taxid {taxid}, {dl_error_message}")
        
def monitor_downloads(max_genome_event, terminate_event, pool): 
    '''Monitor the download processes and increment the success count'''
    logger.info('Monitor initiated')
     # check for max_genome_count reached event 
    while True:
        if max_genome_event.is_set():
            terminate_event.set()
            logger.info('Terminate initiated')
            break
        time.sleep(2)  

def download_genomes(id_list: list, max_genome_count: int, workdir: str, log_file: str) -> (list, dict):
    
    '''uses ncbi datasets to download genomes for a list of taxids'''
    
    # Create a directory to store the genomes
    os.makedirs(workdir + '/genomes', exist_ok=True)
      
    # shuffle taxid list 
    random.shuffle(id_list)
    
    # Log download command 
    download_command = [
        "datasets", "download", "genome", "taxon", "taxid",
        "--filename", f"workdir/genomes/taxid_dataset.zip",
        "--include", "seq-report,protein,gff3", "--annotated",
        "--assembly-source", "RefSeq", "--assembly-level", "complete",
        "--assembly-version", "latest", "--released-after", "01/01/2012"
    ]
    download_command = ' '.join(download_command)
    logger.info("ncbi datasets download command:" + download_command)
        
    
    # Download the genomes
    potential_dl_count = min(max_genome_count, len(id_list))
    max_genome_count = min(max_genome_count, len(id_list))
 
    # generate all potential download commands and store in list 
    download_commands = []
    taxids_used = []
    for taxid in id_list:
        
        # Set output file path
        o_file = f"{workdir}/genomes/{taxid}_dataset.zip"
        
        # run dataset download command 
        download_command = [
            "datasets", "download", "genome", "taxon", str(taxid),
            "--filename", f"{workdir}/genomes/{taxid}_dataset.zip",
            "--include", "seq-report,protein,gff3", "--annotated",
            "--assembly-source", "RefSeq", "--assembly-level", "complete",
            "--assembly-version", "latest", "--released-after", "01/01/2016",
            "--reference"
        ]
        
        # add to command lists  
        download_commands.append(download_command)
        taxids_used.append(taxid)

    # multiprocess dataset downloads
    # control a server process with a Manager to track succes_queue changes 
    with mp.Manager() as manager:
        success_queue = manager.Queue() 
        fg_genomes_selected = manager.list()
        bg_genomes_selected = manager.list()
        taxid_genome_map = manager.dict()
         # for tracking downloads 
        max_genome_event = manager.Event()
        log_queue = manager.Queue()
        terminate_event = manager.Event()  
        
        # HARD CODING FOR TESTING REMOVE AND MOVE TO CONFIG FILE 
        globi_db_path = '/private/groups/corbettlab/gabe/evorate_debug_clutter/GloBI_tests/globi.db'
        
        # create processes
        with mp.Pool(processes=100) as pool:
            
            # Start the monitor_downloads function in a separate thread
            monitor_process = mp.Process(target=monitor_downloads, args=(max_genome_event, terminate_event, pool))
            monitor_process.start()
            
            # multi-threaded download ansyn 
            results = [pool.apply_async(dataset_download, args=(download_command, taxid, taxid_genome_map, 
                                                                success_queue, fg_genomes_selected, bg_genomes_selected, 
                                                                max_genome_count, max_genome_event, globi_db_path)) 
                       for download_command, taxid in zip(download_commands, taxids_used)]
            
            # moinitor monitoring loop 
            while True:
                if terminate_event.is_set():
                    logger.info("Terminating download pool")
                    pool.terminate()
                    pool.join()
                    break
                elif all(result.ready() for result in results):
                    logger.info("All possible downloads attempted")
                    terminate_event.set()
                    max_genome_event.set()
                    pool.close()
                    logger.info("Pool Closed")
                    pool.terminate()
                    logger.info("Pool Terminated")
                    pool.join()
                    logger.info("Pool Joined")
                    break
                time.sleep(2) # check every two seconds
            
            logger.info("Terminating monitor process")
            # stop monitor 
            monitor_process.join()
            logger.info("Terminated monitor process")
            
            
            
            logger.info(f"Downloaded {len(fg_genomes_selected)} foreground genomes and {len(bg_genomes_selected)} background genomes of {max_genome_count} requested")
        # return first taxids selected up until max_genome_count in case there were more downloaded after the pool termination 
        
        return list(fg_genomes_selected)+list(bg_genomes_selected), dict(taxid_genome_map) 

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

   
    
    # old logging config
    logger.remove()  # Remove default handler
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dl_log_file = args.log + f"ncbi_genome_download_{timestamp}.log"
    citation_log_file = args.log + f"GloBI_citations_{timestamp}.log"
    logger.add(dl_log_file, enqueue=True, rotation="10 MB")
    logger.add(citation_log_file, enqueue=True, rotation="10 MB", filter=lambda record: record["extra"].get("task")== "citations") # Add a log file handler
    
    
    # make workdir
    os.makedirs(args.workdir, exist_ok=True)
    
    # Collect genome fastas for the taxid group
    id_list = taxonkit_get_subtree(args.taxid)
    genomes_selected, genome_accession_map = download_genomes(id_list, args.max_genome_count, args.workdir, dl_log_file)
    
    # Save genome_accession_map to a json file 
    with open(f"{args.workdir}/genomes/genome_accession_map.json", 'w') as f:
        json.dump(genome_accession_map, f)
    
    # Save list of genomes used for downstream analysis 
    with(open(f"{args.log}/genomes_selected.txt", 'w')) as f:
        # Write all file names of zip files in the genomes directory
        zip_files = [file for file in os.listdir(f"{args.workdir}/genomes/") if file.endswith(".zip")]
        for zip_file in zip_files:
            taxid = zip_file.split('_')[0]
            f.write(f"{taxid}\n")
            
if __name__ == "__main__":
    main()