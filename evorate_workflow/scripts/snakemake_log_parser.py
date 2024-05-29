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

# Usage example
log_file = '/storage1/gabe/mimic_screen/code/mimic_screen/evorate_workflow/.snakemake/log/2024-05-21T165934.414516.snakemake.log'
failed_samples = parse_snakemake_log(log_file)

print(f"Number of failed samples: {len(failed_samples)}")
for sample, meta in failed_samples.items():
    print(f"Sample: {sample}, Failed at Rule: {meta[0]} JobID: {meta[1]}")