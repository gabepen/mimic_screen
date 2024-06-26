import re
from tqdm import tqdm
import sys
import os

def parse_snakemake_log(log_file):
    failed_samples = {}
    with open(log_file, 'r') as file:
        for line in file:
            match = re.search(r'Error in rule (.+)', line)
            if match:
                rule = match.group(1).strip(':')
                
                for line in file:
                    match = re.search(r'jobid: ', line)
                    if match:
                        jobid = line.split(':')[1].strip()
                    match = re.search(r'wildcards: ', line)
                    if match:
                        sample = line.split('sample=')[1].strip()
                        break
                if sample not in failed_samples:
                    failed_samples[sample] = (rule, jobid)

    return failed_samples

def parse_snakemake_log_slurm(slurm_log_file):
    
    # store samples as dictionary keys where values are error details
    failed_samples = {}
    
    # parse slurm master log file
    
    with open(slurm_log_file, 'r') as file:
        for line in file:
            
            # search for each rule error
            match = re.search(r'Error in rule (.+)', line)
            if match:
                rule = match.group(1).strip(':')
                
                # with error found search for details 
                status = 'NA'
                for line in file:
                    
                    # determine the status of the job
                    match = re.search(r"message:.*SLURM status is:\s+'(\w+)'", line)  # Modify the regex pattern to match the desired string even if it doesn't occur at the start of the line
                    if match:
                        status = match.group(1)
                        
                    # determine the sample name
                    output_match = re.search(r'output: (.+)', line)
                    if output_match:
                        sample = line.split('/')[-2]
                        
                    # determine path to job specific log file to find error details 
                    log_match = re.search(r'log: (.+)', line)
                    if log_match:
                        log_path = log_match.group(1).split()[0]
                        status_detail = 'NA'
                        
                        # open job specific log file and search for error details
                        with open(log_path, 'r') as rule_log_file:
                            for rl in rule_log_file:
                                error_match = re.search(r'ERROR: (.+)', rl)
                                if error_match:
                                    status_detail = error_match.group(1)
                                    break
                        break
                
                # store sample that failed with meta data 
                if sample not in failed_samples:
                    failed_samples[sample] = [rule, status, status_detail]

    return failed_samples

'''
Script for parsing snakemake log files for failed samples and failuer reasons 

Usage: python snakemake_log_parser.py log_file output_file rule_spec
    log_file: path to log file
    output_file: path to output file for sample accession writing 
    rule_spec: only write samples that failed at a certain rule to output_file 
    
'''

log_file = sys.argv[1]
output_file = sys.argv[2]
rule_spec = sys.argv[3]

failed_samples = parse_snakemake_log_slurm(log_file)

if output_file:
    with open(output_file, 'w') as f:
        for sample, meta in failed_samples.items():
            if meta[0] == rule_spec:
                f.write(f"{sample}\n")
        print(f"Failed samples written to {output_file}")
        
print(f"Number of failed samples: {len(failed_samples)}")
for sample, meta in failed_samples.items():
    print(f"Sample: {sample} Rule: {meta[0]} Status: {meta[1]}, {meta[2]}")
print(f"Total number of failed samples: {len(failed_samples)}")