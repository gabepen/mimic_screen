from loguru import logger
import argparse
import urllib.request
import json
import xmltodict
import datetime
import csv
import sys
import os

def is_source_freeliving(isolation_source: str) -> tuple:
    
    """
    Determines if the given isolation source is associated with a free-living environment or a host-associated environment.
    Args:
        isolation_source (str): The source from which the sample was isolated.
    Returns:
        tuple: A tuple containing a boolean and a string. The boolean indicates whether the source is free-living (True) or not (False). 
               The string provides additional information about the determination, such as the specific term that led to the conclusion.
    """
    
    
    isolation_source = isolation_source.lower()
    
    if isolation_source == 'missing':
        return (False, 'no isolation source')
    
    if 'culture' in isolation_source:
        return (False, 'from culture')
    
    # define core terms to check for freeliving status
    core_free_living_attributes = ['soil', 'sediment', 'water', 'air', 'dust', 'marsh', 'saltern', 
                                   'spring', 'vent', 'river', 'wasterwater', 'freshwater', 'seawater', 
                                   'bay']
    core_host_associated_attributes = ['human', 'animal', 'infected', 'plant', 'insect', 'fungal', 
                                       'protozoan', 'algal', 'microbial', 'culture', 'strain']
    
    
    living_entity_syns = {'living_thing', 'organism', 'animal', 'plant', 'body_part', 'organ', 'culture', 
                               'biological_entity', 'biological_object', 'biological_attribute',
                               'plant_organ', 'biological_system'}
    
    # determine word association with NLP model to determine if the isolation source is associated with a living thing 
    for word in isolation_source.split():
        synsets = wn.synsets(isolation_source)
        for synset in synsets:
            # Check if the synset's hypernyms contain any living categories
            hypernyms = synset.hypernyms()
            while hypernyms:
                for hypernym in hypernyms:
                    if hypernym.name().split('.')[0] in living_entity_syns:
                        print(hypernym)
                        return (False, hypernym)
                hypernyms = hypernyms[0].hypernyms() 
    
    # check for host associations specific isolation source words
    for word in isolation_source.split():
        if word in core_host_associated_attributes:
            return (False, word)
        
    # no living entity associations found in isolation source words confirm with core terms
    for word in isolation_source.split():
        if word in core_free_living_attributes:
            return (True, word)
    
    # no core terms found in isolation source words 
    return (False, 'unknown')

def validate_globi_results(taxid: str, rows: list):
    
    """
    Parses the rows returned from a GloBI query to determine if a valid interaction has been observed.
    Args:
        taxid (str): The taxonomic ID of the organism being queried.
        rows (list): A list of rows returned from the GloBI query, where each row is expected to be a list containing interaction data.
    Returns:
        bool: True if valid interactions are found, False otherwise.
    Logs:
        Logs citation information for unique interactions found.
    """
        
    
    # no interaction results found
    if len(rows) == 0:
        return False

    # interactions found, check citations 
    found_interactions = set()
    for r in rows:
        
        # only save citations for unique interactions
        if (r[38],r[42]) not in found_interactions:
            ref_citation = r[84]
            source_citation = r[85]
            found_interactions.add((r[38], r[42]))
    
    return True 

def ncbi_taxid_to_biosample_uids(taxid: str) -> list:
    
    entrez_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=biosample&term=txid{taxid}&retmax=10&retmode=json"
    results_json = json.loads(urllib.request.urlopen(entrez_url).read())
    return results_json['esearchresult']['idlist']

def check_sciname_for_symbiosis(sci_name: str) -> bool:

    """
    Check if the scientific name indicates a symbiotic relationship.
    This function examines the provided scientific name to determine if it 
    suggests a symbiotic taxa. It checks for the presence of specific keywords 
    that are commonly associated with symbiosis.
    Args:
        sci_name (str): The scientific name to be checked.
    Returns:
        bool: True if the scientific name indicates a symbiotic relationship, 
              False otherwise.
    """
    
    # conditions are clear indications of a symbiotic taxa
    return 'symbiont' in sci_name.lower()

def check_biosample_isolation_source(biosample_uids: list) -> bool:
    
    """
    Checks if the isolation sources of given biosample UIDs are associated with free-living environments.
    This function queries the NCBI Entrez API to retrieve metadata for each biosample UID provided. It then
    parses the metadata to find the 'isolation_source' attribute and checks if it is associated with a 
    free-living environment using the `is_source_freeliving` function.
    Args:
        biosample_uids (list): A list of biosample UIDs to be checked.
    Returns:
        bool: Returns True if all biosample isolation sources are confirmed with free-living environments,
              otherwise returns False.
    """
    
    
    # initialize tasked logger
    logger_biosamples = logger.bind(task="biosamples")
    
    
    # check each biosample uid for freeliving sources
    for uid in biosample_uids:
        
        # format entrez API url
        entrez_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=biosample&id={uid}&retmode=json"
        
        # call entrez API and parse results
        results_json = json.loads(urllib.request.urlopen(entrez_url).read())
        sample_data = xmltodict.parse(results_json['result'][uid]['sampledata'])
        
        # check for isolation source field in the attributes section of the biosample metadata
        for meta_data_field in sample_data['BioSample']['Attributes']['Attribute']:
            if meta_data_field['@attribute_name'] == 'isolation_source':
                isolation_source = meta_data_field['#text']
                
                # check if the isolation source is associated with a free-living environment
                freeliving_status = is_source_freeliving(isolation_source)
                logger_biosamples.info(f"{uid} | {isolation_source} | {freeliving_status[0]} | {freeliving_status[1]}")
                
                # if the any of the biosample isolation sources are associated with a host, return False
                if not freeliving_status[0]:
                    return False 
            elif meta_data_field['@attribute_name'] == 'host':
                host = meta_data_field['#text']
                if host.lower() != 'enviorment':
                    logger_biosamples.info(f"{uid} | {host} | host")
                    return False
    return True
