#!/bin/bash

#A typical run takes couple of hours but may be much longer
#SBATCH --job-name=macsyfinder
#SBATCH --time=00:10:00
#SBATCH --partition=short
# Set the job array size
#SBATCH --array=1-200%50

#log files:
#SBATCH -e /private/groups/corbettlab/gabe/evorate_debug_clutter/macsyfinder/test/logs/macsyfinder_jobs_%A_%a_err.txt
#SBATCH -o /private/groups/corbettlab/gabe/evorate_debug_clutter/macsyfinder/test/logs/macsyfinder_jobs_%A_%a_out.txt

#Limit the run to a single node
#SBATCH -N 1

#Adjust this depending on the node
#SBATCH --ntasks=1
#SBATCH --mem=500

source /private/home/gapenunu/.bashrc
mamba activate evorate

models_dir='/private/groups/corbettlab/gabe/evorate_debug_clutter/macsyfinder/models'
input_file_list='/private/groups/corbettlab/gabe/evorate_debug_clutter/macsyfinder/genomes/gpbacteria_ordered_inputs.txt'
output_dir='/private/groups/corbettlab/gabe/evorate_debug_clutter/macsyfinder/test/output/TXSScan_gpbacteria_tree_ordered'

index=$(($SLURM_ARRAY_TASK_ID))

# Read the input file list
IFS=$'\n' read -rd '' -a input_files < "$input_file_list"

# Get the archive proteome file path for the current job
proteome_file="${input_files[$((index - 1))]}"
filename_prefix=$(basename $proteome_file .fasta)

macsyfinder --sequence-db $proteome_file -o $output_dir/$filename_prefix --models TXSScan all --db-type ordered_replicon -w 8 --models-dir $models_dir


