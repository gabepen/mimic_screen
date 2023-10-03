import argparse
from glob import glob 
import os 



            
def alignment_stats(controls, aln_tsv1, thresholds):
    '''
    checks foldseek controls output for high structural similarity
    stores name of query protein in dict object for look up when parsing expirement alignment
    stores all alignments in a data table with selection indicating field (controlled or not)
    ''' 

    data_table = []
    sig_alns = {}
    
    # establish thresholds 
    c_tmscore_filter, c_tcov_filter = thresholds[0], thresholds[1]
    q_tmscore_filter, q_tcov_filter = thresholds[2], thresholds[3]

    if controls:
        for c in controls:
            # open and parse alignment file
            with open(c, 'r') as tsv:
                lines = tsv.readlines()
                # parse line components
                for l in lines:
                    lp = l.strip().split()

                    # foldseek output values
                    query = lp[0]
                    target= lp[1]
                    evalue = float(lp[2])
                    score = lp[3]
                    tcov = float(lp[6]) 
                    qcov = float(lp[7])
                    fident = float(lp[12])

                    # check for high structural homology adding hits to significant alignment list
                    if float(score) > c_tmscore_filter and evalue < 0.01 and tcov > c_tcov_filter:
                        if query not in sig_alns:
                            sig_alns[query] = []
                            sig_alns[query].append((target,float(score),evalue,fident))
                        else:
                            sig_alns[query].append((target,float(score),evalue,fident))
                    

    #parse expiremental alignment 
    with open(aln_tsv1, 'r') as tsv:
        lines = tsv.readlines()
        for l in lines:
            lp = l.strip().split()

            # foldseek alignment values 
            query = lp[0]
            target = lp[1] 
            evalue = float(lp[2])
            fident = float(lp[12])
            score = lp[3]
            tcov =  float(lp[6])

            # apply filters and label alignments in data table 
            if float(score) > q_tmscore_filter and tcov > q_tcov_filter and evalue < 0.01 and query not in sig_alns:
                data_table.append([query, float(score), tcov, fident, 'candidate'])
            elif float(score) < q_tmscore_filter or tcov < q_tcov_filter or evalue > 0.01 :
                data_table.append([query, float(score), tcov, fident, 'filtered'])
            else:
                data_table.append([query, float(score), tcov, fident, 'controlled'])
    
    return data_table

def main():

    '''script for generating datatable of protein structure alignment outcomes based on 
       given tcov and tm-score thresholds and presence in control alignment files

       alignment files should be generated with foldseek using:
        --format-output query,target,evalue,alntmscore,alnlen,qlen,tcov,qcov,tlen,u,t,lddt,fident,pident,prob
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument('thresholds', metavar='TH', type=float, nargs=4, 
                        help='four floats in order as thresholds for' +
                             'control TM-score, Target Coverage, and query TM-score and Target Coverage')
    parser.add_argument('aln_tsv1', metavar='T1', type=str, nargs='?', help='path to an expiremental alignment file')
    parser.add_argument('controls', metavar='C', type=str, nargs='*', help='paths to any number of control alignments')
    args = parser.parse_args()

    data_table = alignment_stats(args.controls, args.aln_tsv1, args.thresholds)


if __name__ == '__main__':
    main()