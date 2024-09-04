import os

def filter_rbh_results(rbh_results_file: str, output_file: str) -> tuple:
   
    # potential outcome groups for lines in file
    evalue_filtered_rbh_results = []
    coverage_filtered_rbh_results = []
    passed_hits = []
    
    with open(rbh_results_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            fields = line.split('\t')
            # check evalue 
            if float(fields[-1]) > 1e-6:
                evalue_filtered_rbh_results.append((fields[0], fields[1]))
                continue
            # check coverage over shortest protein
            if int(fields[5]) > int(fields[6]):
                coverage = float(fields[8])
            else:
                coverage = float(fields[7])
            # must be greater than 0.6 to be considered ortholog
            if coverage < 0.6:
                coverage_filtered_rbh_results.append((fields[0], fields[1]))
                continue
            passed_hits.append(line)

    # write passed lines to output file 
    with open(output_file, 'w') as f:
        for line in passed_hits:
            f.write(line)
    
    # return results as counts for logging 
    return (len(evalue_filtered_rbh_results), len(coverage_filtered_rbh_results))


def main():
        
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Filter RBH results based on evalue and coverage')
    parser.add_argument('rbh_results_file', type=str, help='Directory containing RBH results')
    args = parser.parse_args()
    
    filter_rbh_results(args.rbh_results_file)

if __name__ == '__main__':
    main()

                