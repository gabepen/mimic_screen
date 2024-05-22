import argparse
import os
import json

def parse_absrel_results(directory, test_type):
    
    # output csv 
    with open('absrel_results.tsv', 'w') as out:
        out.write('Sample\tAvergae Corrected P-value\tsignificant selection branches\n')
        # parse directory of absrel result json files
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                filepath = os.path.join(directory, filename)
                
                with open(filepath) as file:
                    data = json.load(file)
                    
                    corrected_pvalues = []
                    selection_pvalues = 0
                    for branch in data['branch attributes']['0']:
                        corrected_pvalues.append(data['branch attributes']['0'][branch]['Corrected P-value'])
                        if data['branch attributes']['0'][branch]['Corrected P-value'] < 0.05:
                            selection_pvalues += 1
            out.write(f"{filename}\t{round(sum(corrected_pvalues)/len(corrected_pvalues), 2)}\t{selection_pvalues}\n")  
    

def main():
    parser = argparse.ArgumentParser(description='Parse JSON files')
    parser.add_argument('-d','--directory', help='Path to the directory containing JSON files')
    parser.add_argument('-t','--test_type', help='Type of test')
    args = parser.parse_args()
    
    
    if args.test_type == 'absrel':
        parse_absrel_results(args.directory, args.test_type)

if __name__ == '__main__':
    main()