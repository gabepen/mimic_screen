#!/bin/bash

#A typical run takes couple of hours but may be much longer
#SBATCH --job-name=dlgenome
#SBATCH --time=00:10:00
#SBATCH --partition=short
# Set the job array size
#SBATCH --array=1-200%50

#log files:
#SBATCH -e /private/groups/corbettlab/gabe/evorate_debug_clutter/macsyfinder/genomes/logs/dlgenome_jobs_%A_%a_err.txt
#SBATCH -o /private/groups/corbettlab/gabe/evorate_debug_clutter/macsyfinder/genomes/logs/dlgenome_jobs_%A_%a_out.txt

#Limit the run to a single node
#SBATCH -N 1

#Adjust this depending on the node
#SBATCH --ntasks=1
#SBATCH --mem=50


source /private/home/gapenunu/.bashrc
mamba activate evorate

# Path to the text file containing taxids
taxids_file="/private/groups/corbettlab/gabe/mimic_screen/evorate_workflow/hp_outputs/logs/genomes_selected.txt"

index=$(($SLURM_ARRAY_TASK_ID))

# Read the input file list
IFS=$'\n' read -rd '' -a input_files < "$taxids_file"

# Get the archive proteome file path for the current job
taxid="${input_files[$((index - 1))]}"

# Run the datasets command for each taxid
datasets download genome taxon $taxid --filename /private/groups/corbettlab/gabe/evorate_debug_clutter/macsyfinder/genomes/gp_ordered/$taxid.zip --include gff3,genome,protein,cds,seq-report --reference --assembly-source RefSeq --assembly-version latest
