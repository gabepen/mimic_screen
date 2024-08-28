import argparse
import Bio
from Bio import SeqIO
import pandas as pd
import pdb
from typing import TextIO
import zipfile
from io import TextIOWrapper

def order_proteins_by_gene_annotation(gff3_file, proteome_fasta, output):

    gene_info = pd.DataFrame(columns=["chromosome", "start", "end", "protein_id"])
    # Load and parse GFF3 file
    features = [l for l in gff3_file.readlines() if not l.startswith("#")]
    for feature in features:
        fields = feature.split("\t")
        if fields[1] == "Protein Homology" or fields[1] == "GeneMarkS-2+":
            protein_id = fields[8].split(";")[0].split("=")[1].split('-')[1]
            gene_info = pd.concat([gene_info, pd.DataFrame({"chromosome": [fields[0]], "start": [int(fields[3])], "end": [fields[4]], "protein_id": [protein_id]})], ignore_index=True)
    
    # Load the multi-protein FASTA file
    protein_sequences = SeqIO.to_dict(SeqIO.parse(proteome_fasta, "fasta"))
    # Sort genes by chromosome and start position
    gene_info = gene_info.sort_values(by=["chromosome", "start"])
    
    # Create a list of ordered protein sequences
    ordered_proteins = []
    for index, row in gene_info.iterrows():
        protein_id = row["protein_id"]
        if protein_id in protein_sequences:
            ordered_proteins.append(protein_sequences[protein_id])
        '''
        else:
            print(f"Warning: Protein {protein_id} not found in FASTA file")
        '''
        
    # Write ordered protein sequences to a new FASTA file
    with open(output, "w+") as output:
        SeqIO.write(ordered_proteins, output, "fasta")


def main():
    parser = argparse.ArgumentParser(description='Process GFF3 file and proteome FASTA.')
    parser.add_argument('-g', '--gff3_file', help='Path to the GFF3 file')
    parser.add_argument('-p', '--proteome_fasta', help='Path to the proteome FASTA file')
    parser.add_argument('-o', '--output', help='Path to the output file')
    parser.add_argument('-z', '--zip_file', help='Path to the input zipfile')
    args = parser.parse_args()
    
    with zipfile.ZipFile(args.zip_file, 'r') as zip_ref:
        # Get the list of files in the archive
        file_list = zip_ref.namelist()
        
        # find the protein and gff files
        for file_name in file_list:
            if file_name.endswith('protein.faa'):
                protein_file = file_name
            elif file_name.endswith('genomic.gff'):
                gff_file = file_name
           
        with zip_ref.open(gff_file) as gff3, zip_ref.open(protein_file) as protein_fasta:
            order_proteins_by_gene_annotation(
                TextIOWrapper(gff3, encoding='utf-8'),
                TextIOWrapper(protein_fasta, encoding='utf-8'),
                args.output
            )

if __name__ == '__main__':
    main()