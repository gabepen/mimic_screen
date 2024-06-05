import re

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
    failed_samples = {}
    with open(slurm_log_file, 'r') as file:
        for line in file:
            match = re.search(r'Error in rule (.+)', line)
            if match:
                rule = match.group(1).strip(':')
                status = 'NA'
                for line in file:
                    match = re.search(r"message:.*SLURM status is:\s+'(\w+)'", line)  # Modify the regex pattern to match the desired string even if it doesn't occur at the start of the line
                    if match:
                        status = match.group(1)
                    output_match = re.search(r'output: (.+)', line)
                    if output_match:
                        sample = '_'.join(line.split('/')[-1].strip().split('_')[0:2])
                        break
                if sample not in failed_samples:
                    failed_samples[sample] = (rule, status)

    return failed_samples
# Usage example
log_file = '/private/groups/corbettlab/gabe/mimic_screen/evorate_workflow/.snakemake/log/2024-05-29T162611.929951.snakemake.log'
failed_samples = parse_snakemake_log_slurm(log_file)

print(f"Number of failed samples: {len(failed_samples)}")
for sample, meta in failed_samples.items():
    print(f"Sample: {sample} Rule: {meta[0]} Status: {meta[1]}")
print(f"Total number of failed samples: {len(failed_samples)}")