import sys
import subprocess
import argparse
import os
from glob import glob
from tqdm import tqdm

def main():

    ''' Takes a query protein structure file and runs foldseek easy-rbh
        on a specified set of proteomes. Can be used for multiple samples with multi_rbh.sh
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--query', type=str, help='path to query proteome')
    parser.add_argument('-o','--output', type=str, help='path to parent output folder')
    parser.add_argument('-d','--database_dir', type=str, help='path to database directory')
    parser.add_argument('-t','--threads', type=str, help='threads (INT) to pass to foldseek, default = ALL')
    args = parser.parse_args()

    # make paths absolute
    output_prefix = args.output
    if output_prefix[-1] != '/':
        output_prefix += '/'

    # create output dir 
    if not os.path.isdir(output_prefix):
        os.mkdir(output_prefix)
    
    # make paths absolute
    database_dir = args.database_dir
    if database_dir[-1] != '/':
        database_dir += '/'

    # foldseek calling
    databases = glob(database_dir + '*') 
    for db_path in tqdm(databases):
        
        # check if proteome exists in the database directory
        proteome = []
        proteome = glob(db_path + '/*.tar')
        if len(proteome) < 1:
            print('No archive found for DB: {}'.format(db_path))
            continue
        elif len(proteome) > 1:
            print('Unmerged proteomes shards in DB: {}'.format(db_path))
            continue
        
        # create alignment output path and run foldseek easy-rbh
        db_name = db_path.split('/')[-1]
        output_path = output_prefix + db_name + '_rbh.tsv'

        # check to see if output exists
        if os.path.exists(output_path): 
            continue

        if args.threads:
            subprocess.run(['foldseek', 'easy-rbh', args.query,
                            proteome[0], output_path, 'multirbhtmp', '--threads', args.threads, 
                            '--format-output', 'query,target,evalue,alntmscore,alnlen,qlen,tcov,qcov,tlen,u,t,lddt,fident,pident,prob,qstart,qend,tstart,tend'], stdout=subprocess.DEVNULL)
        else:
            subprocess.run(['foldseek', 'easy-rbh', args.query,
                            proteome[0], output_path, 'multirbhtmp',
                            '--format-output', 'query,target,evalue,alntmscore,alnlen,qlen,tcov,qcov,tlen,u,t,lddt,fident,pident,prob,qstart,qend,tstart,tend'], stdout=subprocess.DEVNULL)

if __name__ == '__main__':
    main()