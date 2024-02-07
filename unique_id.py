import argparse
from collections import defaultdict 

def collect_unqiue_ids(results_file, threshold):
    
    '''takes a results file csv generated by alignment_analysis.py 
    
       outputs a list of unique target uniprot ids to stdout within a threshold 
       
       format for results file is expected to be as follows:
       queryID,targetID,tm-score,tcov,fid,fraction-freeliving,avgPLDDT,controlstat
    '''
    
    with open(results_file) as csv_f:
        target_ids = set()
        query_ids = set()
        low_score = 1
        for l in csv_f.readlines():
            l = l.split(',')

            low_score = min(low_score, float(l[2]))
            if float(l[5]) <= float(threshold) and l[0] not in query_ids:
                #print(l[0])
                query_ids.add(l[0])
            
            if float(l[5]) <= float(threshold) and l[1] not in target_ids:
                print(l[1])
                target_ids.add(l[1])
                
def main():
    
    '''collect unique IDs from results file based on fraction freeliving proteomes aligned threshold
    
       inteneded for preparing list of accession for GO enrichment analysis 
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument('-r','--results', type=str, help='path to results file')
    parser.add_argument('-t','--threshold', type=str, help='fraction freeliving proteome aligned threshold')
    args = parser.parse_args()
    
    collect_unqiue_ids(args.results, args.threshold)

if __name__ == '__main__':
    main()