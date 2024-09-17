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
import ordered_replicon
from io import TextIOWrapper

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

def dataset_download(dl_command, taxid, taxon_genome_map,
                     success_queue, fg_genomes_selected, 
                     bg_genomes_selected, max_genome_count, 
                     max_genome_event, macsyfinder_workdir):
    
    def check_symbiosis(taxid: str) -> bool:
        
        try:
            retries = 3
            while retries > 0:
                logger.info(f"Checking for symbiosis in taxid {taxid}")
                result = subprocess.run(f"datasets summary taxonomy taxon {taxid}", shell=True, capture_output=True, text=True)
                reports = result.stdout
                try:
                    # process stdout into dictionary 
                    reports = reports.replace('true', 'True')
                    report_dict = eval(reports)
                    
                    # obtain scientific name from result dict 
                    sci_name = report_dict["reports"][0]["taxonomy"]["current_scientific_name"]["name"]
                    
                    # both conditions are clear indications of a symbiotic taxa
                    if 'symbiont' in sci_name.lower():
                        return True 
                    elif 'candidatus' in sci_name.lower():
                        return True 
                    else:
                        return False

                # this will occur if the taxid does not return a valid result
                # which should only occur with transient errors in the dataset query 
                # so logging the error and retrying is appropriate
                except:
                    logger.warning(f"Symbiosis check for {taxid} error, {result}")
                    retries -= 1
                
            return False
        except Exception as e:
            logger.warning(f"Failed to check for symbiosis in taxid {taxid}, {e}")
            return False
        
    def run_macsyfinder_TXSScan(proteome_file: str, output_dir: str) -> bool:
        
        macsyfinder_argument = ["macsyfinder", "--sequence-db", proteome_file, "-o", output_dir,
                                "--models", "TXSScan", "all", "--db-type", "ordered_replicon",
                                "-w", "4", "--models-dir", "macsyfinder_models",  "--mute"]
        
        # Run macsyfinder subprocess
        subprocess.run(macsyfinder_argument)
        
        # check for output file 
        if not os.path.exists(output_dir + '/best_solution.tsv'):
            return False
        
        # check output for identified secretion systems
        best_solution_tsv = output_dir + '/best_solution.tsv'
        with open(best_solution_tsv, 'r') as f:
            lines = f.readlines()
            if "# No Systems found" in lines[3]:
                return False
            else:
                return True
        
    def map_id_to_genome(taxid: str, archive_path: str) -> dict:
        
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
        
        # extract gff and proteome files for macsyfinder analysis 
       
        with zipfile.ZipFile(o_file, 'r') as zip_ref:
            logger.info(o_file)
            
            
            # Get the list of files in the archive
            file_list = zip_ref.namelist()
            
            # find the protein and gff files
            for file_name in file_list:
                if file_name.endswith('protein.faa'):
                    protein_file = file_name
                elif file_name.endswith('genomic.gff'):
                    gff_file = file_name
           
            # check if the genome contains secretion systems
            # first order the proteome based on GFF coords 
            ordered_replicon_fasta_output = macsyfinder_workdir + f"/{taxid}_ordered_replicon.fasta"
            try:
                with zip_ref.open(protein_file) as protein_fasta, zip_ref.open(gff_file) as gff3:
                    logger.info('running ordered replicon')
                    ordered_replicon.order_proteins_by_gene_annotation(
                        TextIOWrapper(gff3, encoding='utf-8'),
                        TextIOWrapper(protein_fasta, encoding='utf-8'), 
                        ordered_replicon_fasta_output
                    )
            except Exception as e:
                logger.info(f"Failed to extract protein and gff files for taxid {taxid}, {e}")
                return
      
            # run macsyfinder on the ordered replicon
            try:
                has_phenotype = run_macsyfinder_TXSScan(ordered_replicon_fasta_output, macsyfinder_workdir+f"/{taxid}")
            except Exception as e:
                logger.warning(f"Failed to run macsyfinder on taxid {taxid}, {e}")
                has_phenotype = False
            logger.info(f"Taxid {taxid} has phenotype: {has_phenotype}")  
            
            # the taxid has a secretion system, check if it is a symbiont
            if check_symbiosis(taxid):
                # save taxid if forground genomes are not over half of the max genome count
                if len(fg_genomes_selected) < int(max_genome_count / 2):
                    fg_genomes_selected.append(taxid)
                    logger.info(f"Taxid {taxid} is a fg genome")
                    # increment the success count
                    success_queue.put(1)
                else:
                    logger.info(f"Max foreground genomes reached skipping taxid {taxid}")
                    os.remove(o_file)
                    return

            elif has_phenotype == False:   
                # if its not add it to bg genome list
                if len(bg_genomes_selected) < int(max_genome_count / 2):
                    bg_genomes_selected.append(taxid)
                    logger.info(f"Taxid {taxid} is a bg genome")
                    # increment the success count
                    success_queue.put(1)
                else:
                    logger.info(f"Max background genomes reached skipping taxid {taxid}")
                    os.remove(o_file)
                    return
            else:
                logger.info(f"Taxid {taxid} is a not explicity a symbiont, but has a secretion system")
                os.remove(o_file)
                return
                
        # save taxid - genome record map
        taxon_genome_map[taxid] = map_id_to_genome(taxid, o_file) 
        
        # monitor success queue to check if max_genome_count has been reached
        if success_queue.qsize() >= max_genome_count:
            max_genome_event.set()
            logger.info('Max genome number reached')
        
        
    else:
        dl_error_message = dl_output.stderr.decode().split('\n')
        dl_error_message = dl_error_message[1].strip().replace('Error:', '')
        logger.warning(f"Failed to download genome for taxid {taxid}, {dl_error_message}")
        
    

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
    
    # Create temporary directory to store macsyfinder results
    macsyfinder_workdir = workdir + '/tmp_macsyfinder'
    os.makedirs(macsyfinder_workdir, exist_ok=True)
    
    
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
        
        # create processes
        with mp.Pool(processes=100) as pool:
            
            # Start the monitor_downloads function in a separate thread
            monitor_process = mp.Process(target=monitor_downloads, args=(max_genome_event, terminate_event, pool))
            monitor_process.start()
            
            # multi-threaded download ansyn 
            results = [pool.apply_async(dataset_download, args=(download_command, taxid, taxid_genome_map, 
                                                                success_queue, fg_genomes_selected, bg_genomes_selected, 
                                                                max_genome_count, max_genome_event, macsyfinder_workdir)) 
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
    log_file = args.log + f"ncbi_genome_download_{timestamp}.log"
    logger.add(log_file, enqueue=True, rotation="10 MB") # Add a log file handler
    
    
    # make workdir
    os.makedirs(args.workdir, exist_ok=True)
    
    # Collect genome fastas for the taxid group
    id_list = taxonkit_get_subtree(args.taxid)
    genomes_selected, genome_accession_map = download_genomes(id_list, args.max_genome_count, args.workdir, log_file)
    
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