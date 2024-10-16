#!/bin/bash

# Define the BioSample UID
UID="$1"

# Step 1: Search for the BioSample using esearch
echo "Searching for BioSample $UID..."
SEARCH_RESULT=$(curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=biosample&term=${UID}")

# Debugging: Print the search result
echo "Search result: $SEARCH_RESULT"

# Extract the ID from the search result using xmllint
ID=$(echo $SEARCH_RESULT | xmllint --xpath 'string(//IdList/Id)' -)

# Debugging: Print the extracted ID
echo "Extracted ID: $ID"

# Check if ID is found
if [ -z "$ID" ]; then
    echo "Error: No results found for BioSample $UID."
    exit 1
fi

# Step 2: Fetch the BioSample data using efetch
echo "Fetching BioSample data for ID $ID..."
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=biosample&id=$ID&retmode=xml" -o biosample.xml

if [ $? -eq 0 ]; then
    # Check if the file is not empty
    if [ -s biosample.xml ]; then
        echo "BioSample data for $UID saved to biosample.xml"
    else
        echo "Error: BioSample data for $UID is empty."
        exit 1
    fi
else
    echo "Error: Failed to fetch BioSample data for $UID."
    exit 1
fi