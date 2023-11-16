import argparse
from bardapi import BardCookies
import csv 
import requests
import urllib
import http.cookiejar
import bacdive
import re
import pdb

def get_paper_titles_from_bacdive_entry(bacdive_entry):
  """Extracts the paper titles from the literature section of a BacDive database entry.

  Args:
    bacdive_entry: A BacDive database entry.

  Returns:
    A list of paper titles.
  """

  # Get the literature section of the BacDive database entry.
  literature_section = bacdive_entry["Literature"]

  # Split the literature section into individual references.
  references = re.split(r"[;]", literature_section)

  # Extract the paper title from each reference.
  paper_titles = []
  for reference in references:
    # Split the reference into individual fields.
    fields = reference.split(r"\t")

    # The paper title is the first field in the reference.
    paper_title = fields[0]

    # Remove any leading or trailing whitespace from the paper title.
    paper_title = paper_title.strip()

    # Add the paper title to the list of paper titles.
    paper_titles.append(paper_title)

  return paper_titles


def teach_bard(bard):

    '''Bard is not able to obtain accurate sources and reach correct conclusions
       without being taught basic definitions and guidance.

       This function pre-constrains bard on a couple basic principles for the bacteria
       classification task presented in ask_bard()
    '''
    print('introducing concept')
    concept = (
        "Your objective is a research automation task in which you will be given scientific literature on a bacteria strain "
        "and you will determine if the bacteria is free-living or host-associated."
       
    )
    bard.get_answer(concept)

    print('providing definitions')

    free_living_defintion = ( 
        "For free-living bacteria refer to this definiton: "
        "Free-living bacteria, also known as non-symbiotic bacteria, are microorganisms that exist and thrive independently "
        "in various environments, such as soil, water, or the human body, without forming specific mutualistic or parasitic "
        "relationships with other organisms. These bacteria are not bound to a host or a particular location, and they obtain "
        "their nutrients and energy from the surrounding environment. Free-living bacteria can carry out various metabolic processes, "
        "including nitrogen fixation, decomposition of organic matter, and many other functions, contributing to ecosystem processes and the recycling of nutrients."
        
    )
    bard.get_answer(free_living_defintion)

    host_associated_definition = (
        "Host-associated bacteria are microorganisms that have a close and often symbiotic relationship with a host organism, "
        "which can be a plant, animal, or even a human. These bacteria inhabit the host's body or specific organs or tissues and "
        "may provide various benefits or cause harm to the host. "
    )
    bard.get_answer(host_associated_definition)


    guidance = (
        "Certain bacteria might appear to have conflicting information about whether or not it is free-living or host-associated. "
        "In this case default to the bacteria being host-associated and provide the source that led you to that conclusion. "
        "Also double check that source for specific reference to the bacteria's scientific name, if you cannot find a specific reference "
        "than discard the source from your reasoning and make a decision without it."

    )
    #bard.get_answer(guidance)

    citation_instruction = (
        "When selecting a citation for your decision, look for the orginal publication decribing the discovery of the bacteria. "
        "This is likely to be the oldest paper containing the exact scientific name given. If you have found a more recent paper "
        "that discovered novel traits of the bacteria that have determined it to be host-associated use that, but ensure there is "
        "specific reference to the exact scientific name of the bacteria"
    )
    #bard.get_answer(citation_instruction)

    print('constraining output')

    reinforcement_mean = ( 
        "Here are your output constraints, when I say to refer to your output constraints you will review them thouroughly: "
        "You are being way to chatty in your responses, remember this is a research automation task, you are to read the papers and make "
        " a ONE word response to the prompt, there will be no more 'Based on the paper titles' or 'Therefor, my respones to your query is' "
        "DO NOT, under ANY circumstances output more than a single word response. This means no explanation, no preamble to the result, "
        "literally just free-living, host-associated, or unsure if it could be either. The output should be one of these three strings: "
        " ['free-living', 'host-associated', 'unsure']"
    )

    reinforcement_nice = (
        "Please give your results as a one word answer. I do not need any explanation of your findings and anything other than a single word "
        "makes things much more difficult. Just simply select from one of these: ['free-living', 'host-associated', 'unsure']"
    )
    bard.get_answer(reinforcement_mean)

def ask_bard(bard, species_name, literature):

    # first prompt on nature of bacterium 
    answer = bard.get_answer("From the content of these papers determine if the bacteria {} is free-living or host-associated, refer to your output restains and keep your response to one word: {}".format(species_name, str(literature)))['content'] 

    return answer

def get_cookies(url):
    try:
        # Send an HTTP GET request to the provided URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Access and search for the desired cookie
            cookies = response.cookies
            return cookies
        else:
            print(f"Failed to retrieve data. Status code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return None

def main():

    '''This script uses a pypi bard api project https://pypi.org/project/bardapi/ 
       to determine if a given scientific name is a free living bacteria or host 
       associated.

       Currently configured to work with files generated by taxon_selector.py 
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='path to file output of taxon_selector.py')
    parser.add_argument('-o', '--output', type=str, help='path to output file for bard determinations ')
    parser.add_argument('-r', '--resume', type=str, help='tax id to resume calling from')
    args = parser.parse_args()

    # __Secure-1PSID value 

    cookie_dict = {
        "__Secure-1PSID": "cQjPQw7Fp6XQCKZyF6azwVWITjD0s3s1qy-z9s3GK7iDfej3935oaIgdvxQKnXm0L1yw5w.",
        "__Secure-1PSIDTS": "sidts-CjEB3e41hdVqZyp3zzry-xNeC5ThZrrvkAo6LlJOg-q8nOK_TXj93_PDBcXCIYNO1xxLEAA",
        "__Secure-1PSIDCC": "ACA-OxPxnBGmAetuDyhqyArowiNiqHhYQwBoUyvSJGuXTdKT4au-7DBjiiiYzacgOIKshHYt"
        # Any cookie values you want to pass session object.
    }

    # loging to bacdive client 
    client = bacdive.BacdiveClient('gapenunu@ucsc.edu', 'rmz0anz-FCX0dbf1fjy')

    # init bard 
    bard = BardCookies(cookie_dict=cookie_dict)
    
    # pre constrain bard session 
    teach_bard(bard)

    # black listed scientific names
    black_list = ['Mycolicibacter', 'Listeria', 'Pseudomonas', 'Pseudoalteromonas', 'Streptomyces']

    # parse taxon_selector output
    with open(args.file, 'r') as csv_f:
        header = csv_f.readline()
        lines = csv_f.readlines()

        # resuming 
        if args.resume:
            for i in range(len(lines)):
                l = lines[i].split(',')
                if int(args.resume) == int(l[0]):
                    resume_point = i+1
                    break
        else:
            resume_point = 0
            
        # get each species name
        c = 0
        for i in range(resume_point,len(lines)):
            l = lines[i].split(',')
            species_name = l[1].split('(')[0]
            
            # skip black listed names 
            for name in black_list:
                if name in species_name:
                    continue
            
            # clean up scientific name 
            if len(species_name.split()) > 2:
                species_name = ' '.join(species_name.split()[:2])
            taxid = l[0]
            print(species_name)
            
            # collect literature from bacdive
            count = client.search(taxonomy=species_name)
            
            # check if the database contained any hits 
            if count == 0:
                continue
            
            # retrieve papers and store in a list 
            papers = []

            # obtain strain/species from search results with retrieve function 
            for strain in client.retrieve():

                # determine BacDive entry NCBI tax id
                try:
                    ncbi_tax_id = -1

                    # some entries contain multiple taxids for different strains
                    if type(strain['General']['NCBI tax id']) == list:
                        for tax_id_entry in strain['General']['NCBI tax id']:
                            
                            # the one for species is more likely to match the tax ids sleected by taxon_selector.py
                            if tax_id_entry['Matching level'] == 'species':
                                ncbi_tax_id = tax_id_entry['NCBI tax id']
                    
                    # if there is only one tax id for the entry it can be assigned like this 
                    else:
                        ncbi_tax_id = strain['General']['NCBI tax id']['NCBI tax id']

                    # check that BacDive taxid matches the strain selected by taxon_selector.py
                    if ncbi_tax_id == int(taxid):

                        # collect relevant paper titles from the BacDive entry
                        try:
                            if type(strain['External links']['literature']) != list:
                                literature = [strain['External links']['literature']]
                            else:
                                literature = strain['External links']['literature']
                                
                            for paper in literature:
                                papers.append(paper['title'])                 
                        except KeyError:
                            continue

                # catches issues with NCBI tax id entry, shouldnt be a problem anymore hence the breakpoint
                except TypeError as e:
                    print(e)
                    print(strain['General'])
                    print(species_name)
                    breakpoint()
                # entry has a no NCBI tax id (specific strain)
                except KeyError:
                    continue

            # if no papers were found skip 
            if len(papers) == 0:
                print('No papers found for: ' + species_name)
                continue
                
            # ask bard about the bacteria
            answer = ask_bard(bard, species_name, papers)
            c += 1
            
            if 'Response Error' in answer:
                print(answer)
                print('ERROR: connection lost at taxid: ' + taxid + ' species: ' + species_name)
                exit()
            
            # write output to file
            with open(args.output, 'a+') as o_file:
                if len(answer.split('\n')) > 1:
                    parts = answer.split('\n')
                    call = parts[0].strip().replace(',','')
                    explanation = ' '.join(parts[1:]).replace('\n','').replace('\r','').replace(',','')
                else:
                    call = answer.strip().replace(',','')
                    explanation = ''
                o_file.write(taxid + '\t' + species_name + '\t' + call + '\t' + str(papers) + '\n')
            
            # review research task goals 
            if c > 10:
                print('refreshing research task')
                teach_bard(bard)
                c = 0

if __name__ == '__main__':
    main()