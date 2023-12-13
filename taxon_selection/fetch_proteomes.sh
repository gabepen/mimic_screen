#!/bin/bash
set -o pipefail
# cause a bash script to exit immediately when a command fails
set -e
# cause the bash shell to treat unset variables as an error and exit immediately
set -u

# File containing the list of tax IDs and species name, one per line
tax_id_file=$1

# Define the destination directory
destination_dir=$2

# set up logging to catch taxids not found in alphafold DBs 
log_file=$3
missing_file=$4

# function for handling gsutils errors
log_missing() {
    echo "$1" >> "$missing_file"
    echo "$2" >> "$log_file"
}

# Check if the destination directory exists, and create it if not
if [ ! -d "$destination_dir" ]; then
  mkdir -p "$destination_dir"
fi

# Check if the tax ID file exists
if [ ! -f "$tax_id_file" ]; then
  echo "Tax ID file not found: $tax_id_file"
  exit 1
fi

# Loop through each tax ID in the file
while IFS= read -r line; do

  # Split the line into tax_id and species_name
  tax_id=$(echo "$line" | cut -f 1)
  species_name=$(echo "$line" | cut -f 2)

  # format species name 
  species_name="${species_name// /_}"

  # check if species dir exists then create it 
  if [ ! -d "$destination_dir/$species_name" ]; then
    mkdir -p "$destination_dir/$species_name"
  fi

  echo $destination_dir/$species_name
  # Replace [TAX ID] in the command with the current tax ID
  echo "Downloading database for $tax_id: $species_name"
  if output=$(gsutil -m cp "gs://public-datasets-deepmind-alphafold-v4/proteomes/proteome-tax_id-${tax_id}-*_v4.tar" "$destination_dir/$species_name" 2>&1); then
    echo "Download finished"
  else
    log_missing "$tax_id    $species_name" "$output"
    echo "Download failed"
    rm -r "$destination_dir/$species_name"
    continue
  fi

  # merge proteome shards into a single tar archive 

  # first collect shards 
  shards="$destination_dir/$species_name/*_v4.tar"

  # Initialize an empty tar archive
  output_archive="$destination_dir/$species_name/merged-proteome-shards-${tax_id}-v4.tar"
  tar cvf "$output_archive" --files-from /dev/null

  # Loop through all .tar files in the current directory and append them
  echo "Merging proteome shards..."
  for file in $shards; do
    if [ -e "$file" ]; then
      tar --concatenate --file="$output_archive" "$file"
    fi
  done

  # delete used shards
  echo "Cleaning up shards..."
  rm $destination_dir/$species_name/*_v4.tar


done < "$tax_id_file"