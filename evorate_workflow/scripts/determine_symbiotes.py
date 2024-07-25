from tqdm import tqdm
import sys
import os
import subprocess
import pdb

def check_for_symbiosis(tax_id_list):

    symbiotes = []
    for tax_id in tqdm(tax_id_list):
        result = subprocess.run(f"datasets summary taxonomy taxon {tax_id}", shell=True, capture_output=True, text=True)
        reports = result.stdout
        try:
            reports = reports.replace('true', 'True')
            report_dict = eval(reports)
            sci_name = report_dict["reports"][0]["taxonomy"]["current_scientific_name"]["name"]
            if "symbiont" in sci_name:
                symbiotes.append(tax_id)
        except:
            print(f"Error with {tax_id}")
            print(result.stdout)
            continue
       


    return symbiotes

'''
Usage: python snakemake_log_parser.py log_file output_file rule_spec
    tax_ids: path to genomes_selected file
    output_file: path to output file for sample
'''

tax_ids = sys.argv[1]
output_file = sys.argv[2]

with open(tax_ids) as f:
    tax_id_list = f.readlines()

symbiotes = check_for_symbiosis(tax_id_list)

with open(output_file, 'w+') as f:
    for tax_id in symbiotes:
        f.write(tax_id + '\n')