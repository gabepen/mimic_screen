import subprocess
import argparse
from tqdm import tqdm
import os
import zipfile
import shutil

def taxonkit_get_subtree(taxid):

    '''Uses taxonkit to generate a list of subspecies for a given taxid'''
    
    print(f"Getting subtree ids for taxid: {taxid}")
    taxonkit_command = f"taxonkit list --ids {taxid} --indent ''"
    result = subprocess.run(taxonkit_command, shell=True, capture_output=True)
    output = result.stdout.decode().splitlines()
    
    return output[1:]

def download_genomes(id_list, workdir):
    
    '''uses ncbi datasets to download genomes for a list of taxids'''
    
    # Create a directory to store the genomes
    os.makedirs(workdir + '/genomes', exist_ok=True)
    
    # Download the genomes
    for taxid in tqdm(id_list):
        
        if not taxid:
            continue
        
        # Set output file path
        o_file = f"{workdir}/{taxid}_dataset.zip"
        
        # skip already downloaded genomes
        if os.path.exists(o_file):
            continue
        
        # run dataset download command 
        download_command = [
            "datasets", "download", "genome", "taxon", str(taxid),
            "--filename", f"{workdir}/genomes/{taxid}_dataset.zip",
            "--include", "gff3,protein", "--annotated",
            "--assembly-source", "RefSeq", "--assembly-level", "complete",
            "--assembly-version", "latest", "--released-after", "01/01/2022",
        ]
        download_command = ' '.join(download_command)
        
        # Run the download command retrying upon errors up to 10 times 
        subprocess.run(download_command, shell=True)

def mmseq2_RBH(query_proteome, query_id, workdir, threads=1):
    
    # create results directory 
    os.makedirs(workdir + '/rbh_results', exist_ok=True)
    
    # generate list of orthologs between the query proteome and all proteomes in a workdir 
    for file in os.listdir(workdir + '/genomes'):
        if file.endswith(".zip"):
            
            # Extract the contents of the zip file to a directory with the same name
            zip_path = os.path.join(workdir, 'genomes', file)
            extract_dir = os.path.splitext(zip_path)[0]
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            taxid = extract_dir.split('/')[-1].split('_')[0]
         
            # select target proteome fasta from extracted files 
            data_dir = os.path.join(extract_dir, 'ncbi_dataset', 'data')
            subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
            subdir = subdirs[0]
            genome_accession = subdir
            t_fasta = os.path.join(data_dir, subdir, 'protein.faa')
            
            # Run mmseqs easy-rbh to identify orthologs
            mmseqs_command = f"mmseqs easy-rbh {query_proteome} {t_fasta} {workdir}/rbh_results/{query_id}_{taxid}_{genome_accession}.tsv {workdir}/tmp --threads {threads}"
            subprocess.run(mmseqs_command, shell=True)
            # Clean up the extracted files
            shutil.rmtree(extract_dir)
    shutil.rmtree(f"{workdir}/tmp")
            
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

    '''This script preforms an RBH search of a set of taxa to identify orthologs
       of a query sequence from each proteome in the taxa group.
    '''
    
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="RBH blastp analysis for identifying orthologs")
    parser.add_argument("-p", "--query_proteome", help="Path to the query proteome file")
    parser.add_argument("-i", "--query_id", help="ID of the query microbe")
    parser.add_argument("-s", "--query_sequence", help="Specific query sequence from the proteome")
    parser.add_argument("-x", "--taxid", type=int, help="Taxonomy ID for the species of interest")
    parser.add_argument("-n", "--nrdb_path", help="Path to the nr database")
    parser.add_argument("-w", "--workdir", default="work", help="Path to the working directory")
    parser.add_argument("-t", "--threads", type=int, help="Number of threads for mmseqs to use")
    args = parser.parse_args()

    # make workdir
    os.makedirs(args.workdir, exist_ok=True)
    
    # Collect genome fastas for the taxid group
    id_list = taxonkit_get_subtree(args.taxid)
    download_genomes(id_list, args.workdir)
    
    # Run mmseq2 analysis 
    mmseq2_RBH(args.query_proteome, args.query_id, args.workdir)

if __name__ == "__main__":
    main()