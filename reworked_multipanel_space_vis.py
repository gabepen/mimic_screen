import os 
from glob import glob 
import argparse
#import seaborn as sns
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import csv
import alignment_analysis as aa
#import pdb

                
            
def draw_box_around_region(xmin, xmax, ymin, ymax, color, plt):
    plt.gca().add_patch(
        plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, 
                      fill=False, edgecolor=color, linestyle='dashed',lw=2))   

def table_to_frame(data_table):

    df = pd.DataFrame(data_table, columns = ['id','tm-score','tcov','fident','algn_fraction','target','classification'])
    df = df.sort_values(by=['classification'], ascending=False)
    a = df.classification=='candidate'
    cdf = df[a] # controlled alignment data frame
    fdf = df[~a] # full alignment data frame 
    return cdf, fdf

def table_to_csv_unique(data_table, out_path):

    # filter dup step 
    filtered_data_table = []
    seen_ids = {}
    for p in data_table:
        if p[0] in seen_ids:
            if seen_ids[p[0]][1] > p[1]:
                seen_ids[p[0]] = p
            else:
                continue 
        else:
            seen_ids[p[0]] = p

    # add to list for writing out and clean up UIDS 
    for pid in seen_ids:
        query_uid = seen_ids[pid][0].split('_')[0]
        target_uid = seen_ids[pid][-1].split('-')[1]
        other_fields = seen_ids[pid][1:-1]
        filtered_data_table.append([query_uid] + other_fields + [target_uid])


        
    with open(out_path, 'w+') as csv_file:
        writer = csv.writer(csv_file)
        header = ['UniProt ID', 'TM-Score', 'tcov', 'fident', 'homprob', 'controlled', 'evalue', 'target ID']
        writer.writerow(header)
        for prot in filtered_data_table:

            writer.writerow(prot)

def table_to_csv(data_table, out_path):

    with open(out_path, 'w+') as csv_file:
        writer = csv.writer(csv_file)
        header = ['UniProt ID', 'TM-Score', 'tcov', 'fident', 'homprob', 'controlled', 'evalue', 'target ID']
        writer.writerow(header)
        for prot in data_table:
            writer.writerow(prot)
            
def plot_space(data_table, plt_name, ax, fig):

    '''Generates a 2D histogram describing alignment space distribution of 
    aln_tsv1 alignments
    '''
    # plotting 
    tcovs = [l[2] for l in data_table]
    scores = [l[1] for l in data_table]
    outcome = [l[4] for l in data_table]
    fidents = [l[3] for l in data_table]

    hh = ax.hist2d(tcovs, scores, bins=250, cmin=1, cmap=mpl.colormaps['plasma']) 
    ax.set_xlabel('Target Coverage')
    ax.set_ylabel('TM-Score')
    ax.set_title(plt_name)
    
    # Create a dashed rectangle patch
    rect = Rectangle(
        (0.5, 0.6), 0.7, 0.7,
        linestyle='--',  # Dashed line style
        edgecolor='red',  # Edge color
        facecolor='none'  # No fill color
    )
    ax.add_patch(rect)

    # add color bar to plot 
    fig.colorbar(hh[3], ax=ax)

    return scores

def plot_seq_conservation(cdf,fdf,scores,x_vals,x_lab,plt_name,ax, fig):

    def color_by_class(cdf,fdf,scores,x_vals,x_lab,plt_name,ax):
        
        
        sizes = [20*2**n for n in scores]
        colors = {'candidate':'red', 'filtered':'navy', 'controlled':'orange'}
        sizes = {'candidate':1.7, 'filtered':0.7, 'controlled':0.7}
        alpha = {'candidate':1, 'filtered':0.3, 'controlled':0.8}
        labels = {'candidate':'Candidate', 'filtered':'Filtered', 'controlled':'Controlled'}
        classificationes = ['Candidate', 'Filtered', 'Controlled']
        
        a = fdf.classification=='filtered'
        xdf = fdf[a]
        b = fdf.classification=='controlled'
        fdf = fdf[b]

        ax.scatter(cdf[x_vals], cdf['fident'], s=cdf['classification'].map(sizes), 
                c=cdf['classification'].map(colors), alpha=cdf['classification'].map(alpha),
                linewidths=1, label='Candidate', zorder=3)
        ax.scatter(xdf[x_vals], xdf['fident'], s=xdf['classification'].map(sizes), 
                c=xdf['classification'].map(colors), alpha=xdf['classification'].map(alpha),
                linewidths=1, label='Filtered', zorder=1)       
        ax.scatter(fdf[x_vals], fdf['fident'], s=fdf['classification'].map(sizes), 
                c=fdf['classification'].map(colors), alpha=fdf['classification'].map(alpha),
                linewidths=1, label='Controlled', zorder=2) 


        ax.legend(loc='upper left')    
        ax.set(xlim=(0,1), xticks=np.arange(0,1,0.1, dtype=float),
            ylim=(0,1), yticks=np.arange(0,1,0.1, dtype=float))
        #ax.legend(classificationes,loc='upper left')

        ax.set_xlabel(x_lab)
        ax.set_ylabel('Sequence Similarity')
        ax.set_title(plt_name)
   
    def color_by_fraction(cdf,fdf,x_vals,x_lab,plt_name,ax, fig):
    
        # plotting seq conservation 
        #sizes = [20*2**n for n in scores]
        colors = {'candidate':'red', 'filtered':'navy', 'controlled':'orange'}
        #sizes = {'candidate':1.7, 'filtered':0.7, 'controlled':0.7}
        alpha = {'candidate':1, 'filtered':0.3, 'controlled':0.8}
        labels = {'candidate':'Candidate', 'filtered':'Filtered', 'controlled':'Controlled'}
        classificationes = ['Candidate', 'Filtered', 'Controlled']

        a = fdf.classification=='filtered'
        xdf = fdf[a]
        b = fdf.classification=='controlled'
        fdf = fdf[b]
        
        hh = ax.scatter(cdf[x_vals], cdf['fident'], s=0.7, 
            c=cdf['algn_fraction'], cmap=mpl.colormaps['plasma'],
            linewidths=1, label='Candidate', zorder=3)

        fig.colorbar(hh, ax=ax)

        ax.set_xlabel(x_lab)
        ax.set_ylabel('Sequence Similarity')
        ax.set_title(plt_name)
    
    color_by_fraction(cdf,fdf,x_vals,x_lab,plt_name,ax,fig)

    

def multi_panel_alignmentspace(data_table1, data_table2, plt_name1, plt_name2, file_name):

    ''' Original space and seq coservation figure
    fig, axs = plt.subplots(ncols=2, nrows=2)
    scores1 = plot_space(data_table1, plt_name1, axs[0][0], fig)
    scores2 = plot_space(data_table2, plt_name2, axs[1][0], fig)
    cdf1, fdf1 = table_to_frame(data_table1)
    cdf2, fdf2 = table_to_frame(data_table2)
    plot_seq_conservation(cdf2, fdf2, scores2, 'tcov', 'Target Coverage', plt_name2, axs[0][1], fig)
    plot_seq_conservation(cdf2, fdf2, scores2, 'tm-score', 'TM-Score', plt_name2, axs[1][1], fig)
    fig.tight_layout(pad=2)
    fig.savefig(file_name)
    '''
    fig, axs = plt.subplots(ncols=2, nrows=2)
    cdf1, fdf1 = table_to_frame(data_table1)
    cdf2, fdf2 = table_to_frame(data_table2)
    plot_seq_conservation(cdf1, fdf1, [], 'tcov', 'Target Coverage', plt_name1, axs[0][0], fig)
    plot_seq_conservation(cdf1, fdf1, [], 'tm-score', 'TM-Score', plt_name1, axs[1][0], fig)
    plot_seq_conservation(cdf2, fdf2, [], 'tcov', 'Target Coverage', plt_name2, axs[0][1], fig)
    plot_seq_conservation(cdf2, fdf2, [], 'tm-score', 'TM-Score', plt_name2, axs[1][1], fig)
    fig.tight_layout(pad=2)
    fig.savefig(file_name)


def main():

    plt.style.use('/home/gabe/matplot/BME163.mplstyle')

    parser = argparse.ArgumentParser()

    parser.add_argument('-a1','--aln_tsv1', type=str, help='path to first alignment file')
    parser.add_argument('-a2','--aln_tsv2', type=str, help='path to second alignment file')
    parser.add_argument('-c1','--controls1', type=str, help='paths to a directory of control alignments')
    parser.add_argument('-c2','--controls2', type=str, help='paths to a directory of control alignments')
    args = parser.parse_args()

    # generate control database statistics for each alignment file 
    control_dict1 = aa.generate_control_dictionary(args.controls1)
    control_dict2 = aa.generate_control_dictionary(args.controls2)
    
    # generate datatable with results 
    data_table1 = aa.alignment_stats(args.aln_tsv1, control_dict1)
    data_table2 = aa.alignment_stats(args.aln_tsv2, control_dict2)

    #table_to_csv(data_table2, '/storage1/gabe/proteome/final_figs/overcontrolhits.csv')

    multi_panel_alignmentspace(data_table1, data_table2, 'Legionella - Human', 'wMel - Drosophila', '/storage1/gabe/proteome/final_figs/alignment_space_control_fraction.png')
    
if __name__ == '__main__':
    main()