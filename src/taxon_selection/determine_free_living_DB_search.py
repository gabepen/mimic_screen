import argparse
import datetime
import csv
import sys
import os
from loguru import logger

# Add utilities to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../utilities'))

# import utilities module
import globi_db_queries
import free_living_check

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='path to file output of taxon_selector.py')
    parser.add_argument('-o', '--output', type=str, help='path to output file for freeliving determinations')
    parser.add_argument('-d', '--globi_db_path', type=str, help='path to globi database')
    parser.add_argument('-l', '--log', type=str, help='path to log file')
    args = parser.parse_args()
    
    # init loguru logger
    logger.remove()  # Remove default handler
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dl_log_file = f"{args.log}/determine_free_living_{timestamp}.log"
    logger.add(dl_log_file, enqueue=True, rotation="10 MB")
    
    globi_db_path = args.globi_db_path
    
    
    with open(args.output, 'w') as output:
        with open (args.file, 'r') as csv_f:
            header = csv_f.readline()
            lines = csv_f.readlines()
            
            for line in lines:
                try:
                    taxid = line.split(',')[0]
                    organism_name = line.split(',')[1]

                    # check if the scientific name indicates symbiosis
                    if free_living_check.check_sciname_for_symbiosis(organism_name):
                        logger.info(f"{taxid} | {organism_name} | symbiont | sciname")
                        continue
                    
                    # If the scientific name does not indicate symbiosis, check GloBI database for interactions
                    query = {
                        'sourceTaxonName': organism_name, 
                        'interactionTypeName': ['parasiteOf','hasHost','pathogenOf']
                    }
                    results = globi_db_queries.multi_column_search('interactions', query, globi_db_path)
                    if free_living_check.validate_globi_results(taxid, results):
                        logger.info(f"{taxid} | {organism_name} | symbiont | globi")
                        continue 
                    
                    # If no interactions are found, check the biosample isolation sources
                    biosample_uids = free_living_check.ncbi_taxid_to_biosample_uids(taxid)
                    if free_living_check.check_biosample_isolation_source(biosample_uids):
                        logger.info(f"{taxid} | {organism_name} | freeliving ")
                        output.write(f"{taxid}\n")
                        output.flush()
                    else:
                        logger.info(f"{taxid} | {organism_name} | no confidence")
                        continue
                except Exception as e:
                    logger.error(f"Error processing {line}: {e}")
                    continue
                
        

if __name__ == '__main__':
    main()