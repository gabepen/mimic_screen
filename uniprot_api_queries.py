import requests
import pprint

def fetch_uniprot_entry(uniprot_id):
    """
    Fetches a UniProt entry by its ID.
    
    Parameters:
    uniprot_id (str): The UniProt ID of the entry to fetch.
    
    Returns:
    dict: The UniProt entry data in JSON format.
    """
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}?format=json"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()

def search_uniprot(query, format='json'):
    """
    Searches the UniProt database with a given query.
    
    Parameters:
    query (str): The search query.
    format (str): The format of the results (default is 'json').
    
    Returns:
    dict: The search results in the specified format.
    """
    url = "https://www.uniprot.org/uniprot/"
    params = {
        'query': query,
        'format': format
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()

def main():
    # Example usage
    uniprot_id = "Q73FR6"
    entry = fetch_uniprot_entry(uniprot_id)
    pp = pprint.PrettyPrinter(indent=5)
    pp.pprint(entry)
    print(entry.keys())
    print(entry['proteinDescription']['recommendedName']['fullName']['value'])
    
   

if __name__ == "__main__":
    main()