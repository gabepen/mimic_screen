import argparse
import os
from tqdm import tqdm 
import zipfile 
import shutil
import subprocess
from loguru import logger
from Bio import SeqIO
import json
import multiprocessing as mp
import time
import traceback


def shorten_sequence_names(fasta_file):
    
    # Parse the FASTA file
    sequences = SeqIO.parse(fasta_file, "fasta")

    # Create a new list to store the modified sequences
    modified_sequences = []

    # Iterate over each sequence in the file shorten descriptions
    for seq in sequences:
        seq.description = ']'.join(seq.description.split(']')[:2]) + ']'

        # Add the modified sequence to the list
        modified_sequences.append(seq)

    # Write the modified sequences back to the FASTA file
    SeqIO.write(modified_sequences, fasta_file, "fasta")

def download_gene_data(taxid, taxon_dict, accession_dict, seq_dict, workdir):
    
    # manage race conditions with lock 
    # need multiple locks for each potential target fasta 
    locks = {}
    
    def write_to_target_fasta(target_fasta, header, gene_seq):
        
        if target_fasta not in locks:
            locks[target_fasta] = mp.Lock()
        
        with locks[target_fasta]:
            with open(target_fasta, 'a+') as m:
                m.write(header)
                m.write(gene_seq)
    
    # timing download       
    dl_start_time = time.time()
    
    # get the query accessions for the taxon
    accessions = taxon_dict[taxid]
    
    # log
    logger.info(f"Downloading {len(accessions)} WP accession ortholog sequences for taxon: {taxid}")
    
    # format download command 
    output_path = f"{workdir}/orthologs/{taxid}_orthologs_dataset.zip"
    ortho_accessions = [t[1] for t in accessions]
    ncbi_datasets_command = f"datasets download gene accession {','.join(ortho_accessions)} --include gene --filename {output_path} --taxon-filter {taxid}"
       
    # run downlaod command
    result = subprocess.run(ncbi_datasets_command, shell=True, capture_output=True, text=True)
    error_output = result.stderr
    
    # extract protein sequence fasta to taxid folder in orthologs folder 
    zip_path = os.path.join(output_path)
    extract_dir = os.path.splitext(zip_path)[0]
    packet_fasta = f"{workdir}/orthologs/{taxid}"
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extract('ncbi_dataset/data/gene.fna',path=packet_fasta)
    
    try:
        # move packet multiseq file up two directories
        shutil.move(os.path.join(packet_fasta, "ncbi_dataset/data/gene.fna"), packet_fasta)
    except shutil.Error:
        # file exists, thats ok dont panic and keep going 
        pass
    
    dl_end_time = time.time() 
    # downloading by WP accession obtains all gene records annotated across all genomes
    # there could be multiple chromosome accessions for a single genome
    refseq_chr_accessions = accession_dict[taxid]['refseq_chr_accessions'] 
    
    # store
    # Iterate over each sequence in the file
    gene_fa_file = os.path.join(packet_fasta, "gene.fna")
    acc_matched = False
    logger.info("Parsing gene dataset file for taxon {}".format(taxid))
    parse_start_time = time.time()
 
    for seq_record in SeqIO.parse(gene_fa_file, "fasta"):
        
        # Get the accession number connected to the nucleotide sequence
        annotation_accession = seq_record.id.split(':')[0]
        
        # Check for annotation match
       
        if annotation_accession in refseq_chr_accessions:
            acc_matched = True
            
            try:
                # Get the WP accession to append to the correct fasta file
                ortho_wp_accession = seq_record.description.split('protein_accession=')[1].split(']')[0]
            
            except IndexError:
                logger.info(f"Error: No WP accessions found in fasta for taxid: {taxid}")
                break

            # Find the source WP accession that the ortho accession is associated with
            source_wp_accession = [t[0] for t in accessions if t[1] == ortho_wp_accession][0]
            
            # Set output fasta file path
            target_fasta = f"{workdir}/msa_files/{source_wp_accession}/{source_wp_accession}.fna"

            # Create candidate folder for storing gene seqs 
            os.makedirs(workdir + '/msa_files/' + source_wp_accession, exist_ok=True)

            # sequence information stored in tuple
            seq_record.description =  seq_record.description = ']'.join(seq_record.description.split(']')[:1]) + '] ' + taxid
            seq_content = (f">{seq_record.description}\n", str(seq_record.seq) + "\n")

            # Append sequences to multiseq fasta file 
            if target_fasta not in seq_dict:
                seq_dict[target_fasta] = [seq_content]
            else:
                current_list = seq_dict.get(target_fasta, [])
                current_list.append(seq_content)
                seq_dict[target_fasta] = current_list
            
            
                   
    if not acc_matched:
        logger.info(f"Error: No matching assembly accession found for taxid: {taxid}")   
    
    parse_end_time = time.time()
    logger.info(f"Taxid {taxid} download time: {dl_end_time - dl_start_time}, parse time: {parse_end_time - parse_start_time}")
    # remove gene dataset file 
    shutil.rmtree(packet_fasta)

def download_gene_data_packages(taxon_dict, accession_dict, workdir):
    
    with mp.Manager() as manager:
        seq_dict = manager.dict() 
        with mp.Pool(processes=1) as pool:
            logger.info("Started multiprocessing ortholog collection with {} processes".format(mp.cpu_count()))
            results = [pool.apply_async(download_gene_data, args=(taxid, taxon_dict, accession_dict, seq_dict, workdir))
                                                for taxid in taxon_dict.keys()]
            pool.close()
            pool.join()
                
        logger.info("Pool closed")
        
        
        # seq_dict should contain all sequences to be written to all target fasta files in the form of tuples
        for target_fasta in seq_dict:
            with open(target_fasta, 'a+') as m:
                for seq_content in seq_dict[target_fasta]:
                    m.write(seq_content[0])
                    m.write(seq_content[1])
    
            # log fasta size
            logger.info(f"Wrote {len(seq_dict[target_fasta])} sequences to {target_fasta}")
        
    
def collect_ortholog_accesssions(query_id, candidate_list, workdir):
    
    os.makedirs(workdir + '/orthologs', exist_ok=True)
    os.makedirs(workdir + '/msa_files', exist_ok=True)
    
    # dictionaries for sorting accessions 
    query_dict = {}
    taxon_dict = {}
    
    # set for storing candidate IDs where at least one ortholog seq was identified
    candidate_sources = set()
    
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
                        
                        # candidate has at least one ortholog seq
                        candidate_sources.add(fields[0]) 
                          
                        # map candidate accession to its orthologs 
                        if fields[0] not in query_dict:
                            query_dict[fields[0]] = [fields[1]]
                                               
                        else:
                            query_dict[fields[0]].append(fields[1])
                        
                        # map taxid to its orthologs 
                        taxon_dict[taxid].append((fields[0],fields[1]))

    return query_dict, taxon_dict, candidate_sources
   

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
    logger.add(f"{args.log}/orthologs_collection_debug.log", enqueue=True, rotation="10 MB") # Add a log file handler
    
    # load candidate accessions
    with open(args.candidate_list, 'r') as f:
        candidate_list = f.read().splitlines() 
    
    logger.info(f"Collecting {len(candidate_list)} ortholog datasets for {args.query_id}...")
    
    # collect ortholog accessions for the query sequence
    query_dict, taxon_dict, candidate_sources = collect_ortholog_accesssions(args.query_id, candidate_list, args.workdir)
   
    # load genome record json
    with open(args.genome_record_json, 'r') as f:
        genome_accession_dict = json.load(f)
            
    # download
    download_gene_data_packages(taxon_dict, genome_accession_dict, args.workdir)
    
    # Check for candidate IDs not in the candidate source list 
    missing_ids = set(candidate_list) - set(candidate_sources)
    
    # Create dummy files for candidate IDs that had no orthologs within any of the selected taxon
    # These will fail at the next rule but will not stop the workflow
    logger.info(f"Creating dummy files for candidate ID where no orthologs could be collected...")
    for candidate_id in candidate_list:
        expected_fasta_path = f"{args.workdir}/msa_files/{candidate_id}/{candidate_id}.fna"
        if not os.path.exists(expected_fasta_path):
            logger.info(f"{candidate_id}")
            with open(expected_fasta_path, 'w') as f:
                f.write('>no_orthologs_found\n')
                f.write('X\n')
       
    # there can be cases where a accession with only a couple orthologs fails to find a match in the gene data sets
    # leaving the above dummy file creation and adding this one to have a form of logging that the sequence had too few orthologs
    
              
    # shorten sequence names to prevent hyphy post-msa.bf errors when seq names are too long
    '''
    for root, dirs, files in os.walk(args.workdir + '/msa_files'):
        for file in files:
            if file.endswith(".fna"):
                shorten_sequence_names(os.path.join(root, file))
    '''
    # collect ortholog sequences into a single fasta file 
    #ortholog_seq_fasta = collect_ortholog_seqs(args.workdir, args.query_sequence)
    

if __name__ == "__main__":
    main()