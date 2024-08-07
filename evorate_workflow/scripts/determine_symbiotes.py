from tqdm import tqdm
import sys
import os
import subprocess
import urllib.request
import json

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

def check_for_infectious(tax_id_list):
    
    infectious = []
    total_non_infectious = 0
    for tax_id in tax_id_list:
        tax_id = tax_id.strip()
        response = urllib.request.urlopen(f'https://www.bv-brc.org/api/taxonomy/?eq(taxon_id,{tax_id})&http_accept=application/json')
        html = response.read()
        json_string = html.decode("utf-8")
        json_array = json.loads(json_string)
        try:
            if 'symbiont' not in json_array[0]['taxon_name']:
                if json_array[0]['genomes'] == 0:
                    total_non_infectious += 1
                    print(f"{tax_id} is not infectious")
        except IndexError:
            total_non_infectious += 1
    print(f"Total non-infectious: {total_non_infectious}")   
'''
Usage: python determine_symbiotes.py tax_ids output_file
    tax_ids: path to genomes_selected file
    output_file: path to output file for sample
'''

tax_ids = sys.argv[1]
output_file = sys.argv[2]

with open(tax_ids) as f:
    tax_id_list = f.readlines()

#symbiotes = check_for_symbiosis(tax_id_list)
infectious_bacs = check_for_infectious(tax_id_list)

with open(output_file, 'w+') as f:
    for tax_id in symbiotes:
        f.write(tax_id)