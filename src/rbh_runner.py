import sys
import subprocess
import argparse
import os
from glob import glob

def main():

    ''' Takes a query protein structure file and runs foldseek easy-rbh
        on a specified set of proteomes. Can be used for multiple samples with multi_rbh.sh
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--query', type=str, help='path to query pdb or cif file')
    parser.add_argument('-t','--targets', type=str, help='comma seperated list of structural proteome database names')
    parser.add_argument('-o','--output', type=str, help='path to parent output folder')
    parser.add_argument('-d','--databases', type=str, help='path to database directory')
    args = parser.parse_args()

    # make output path absolute
    output_prefix = args.output
    if output_prefix[-1] != '/':
        output_prefix += '/'

    # output file naming and directory creation
    file_name = args.query.split('/')[-1].split('.')[0] 
    o_name = '_'.join(list(dict.fromkeys(file_name.split('_'))))
    if not os.path.isdir(output_prefix + o_name):
        os.mkdir(output_prefix + o_name)
    if os.listdir(output_prefix + o_name) != []:
        print(o_name + ' already generated, exiting...')
        sys.exit(1)

    # foldseek calling
    database_prefix = args.targets.split(',')
    for dbid in database_prefix:
        # check if proteome exists in the database directory
        proteome = []
        proteome = glob(args.databases + dbid + '/*0_v4.tar')
        if len(proteome) < 1:
            proteome = glob(args.databases + dbid + '/*.tar')
        if len(proteome) < 1:
            print('No archive found for DBID: {} \n exiting'.format(dbid))
            sys.exit(1)
        # create alignment output path and run foldseek easy-rbh
        output_path = output_prefix + o_name + '/' + o_name + '_' + dbid + '_rbh.tsv' 
        subprocess.run(['foldseek', 'easy-rbh', args.query,
                        proteome[0], output_path, 'rbhtmp', 
                        '--format-output', 'query,target,evalue,alntmscore,alnlen,qlen,tcov,qcov,tlen,u,t,lddt,fident,pident,prob'])

if __name__ == '__main__':
    main()