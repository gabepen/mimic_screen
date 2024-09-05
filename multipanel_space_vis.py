import os 
from glob import glob 
import argparse
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import csv
import alignment_analysis as aa
import pdb

                
            
def draw_box_around_region(xmin, xmax, ymin, ymax, color, plt):
    plt.gca().add_patch(
        plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, 
                      fill=False, edgecolor=color, linestyle='dashed',lw=2))   

def table_to_frame(data_table):

    df = pd.DataFrame(data_table, columns = ['id','tm-score','tcov','qcov','fident','algn_fraction','target','classification'])
    df = df.sort_values(by=['classification'], ascending=False)
    a = df.classification=='candidate'
    cdf = df[a] # controlled alignment data frame
    #fdf = df[~a] # full alignment data frame 
    return cdf, df

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
        alpha = {'candidate':0.3, 'filtered':0.3, 'controlled':0.8}
        labels = {'candidate':'Candidate', 'filtered':'Filtered', 'controlled':'Controlled'}
        classificationes = ['Candidate', 'Filtered', 'Controlled']

        a = fdf.classification=='filtered'
        xdf = fdf[a]
        #b = fdf.classification=='controlled'
        #fdf = fdf[b]
        
        hh = ax.scatter(cdf['algn_fraction'], cdf['fident'], s=0.7, 
            c=cdf[x_vals], cmap=mpl.colormaps['plasma'],
            linewidths=1, label='Candidate', zorder=3)

        fig.colorbar(hh, ax=ax)

        ax.set_xlabel(x_lab)
        ax.set_ylabel('Sequence Similarity')
        ax.set_title(plt_name)
    
    color_by_fraction(cdf,fdf,x_vals,x_lab,plt_name,ax,fig)

def scatter_hist(cdf,fdf,scores,x_vals,x_lab, ax, ax_histx, ax_histy):

    '''Creates a scatter plot similar to plot_seq_conservation, but also creates a stacked bar plot 
       of percent free living values along each axis 

       y-axis stacked bar plot currently commented out 
    '''

    ax_histx.tick_params(axis="x", labelbottom=False)
    #ax_histy.tick_params(axis="y", labelleft=False)

    # plot scatter plot
    hh = ax.scatter(cdf[x_vals], cdf['fident'], s=0.7, 
            c=cdf['algn_fraction'], cmap=mpl.colormaps['plasma'],
            linewidths=1, label='Candidate')
    
    # set bin width 
    binwidth = 0.025
    xymax = max(np.max(np.abs(cdf[x_vals])), np.max(np.abs(cdf['fident'])))
    lim = (float(xymax/binwidth) + 1) * binwidth

    # crop plot based on selection range
    if x_vals == 'tm-score':
        lower_thresh = 0.4
    else:
        lower_thresh = 0.25

    # arange bins
    bins_x = np.arange(lower_thresh, lim, binwidth)
    bins_y = np.arange(0, 0.8, binwidth)
    
    # bin algn fraction values 
    cdf['algn_fraction_bins'] = pd.cut(cdf['algn_fraction'], bins=np.arange(0, 1.1, 0.025))

    '''
    # stacked histogram
    for fraction_bin, group in cdf.groupby('algn_fraction_bins'):
        color = plt.get_cmap('plasma')(group['algn_fraction'].mean())  # Color-mapping to the mean algn_fraction
        ax_histx.hist(group[x_vals], bins=bins_x, alpha=1, label=f'Fraction: {fraction_bin}',color=color, histtype='bar', stacked=True, density=False)
        ax_histy.hist(group['fident'], bins=bins_y, orientation='horizontal', alpha=1, label=f'Fraction: {fraction_bin}',color=color, histtype='bar', stacked=True, density=False)
    '''

    sorted_groups = cdf.sort_values(by=['algn_fraction']).groupby('algn_fraction_bins')

    
    # Initialize cumulative sum arrays
    cum_counts_x = np.zeros_like(bins_x[:-1])
    cum_counts_y = np.zeros_like(bins_y[:-1])

    # stacked barplot
    for fraction_bin, group in sorted_groups:
        color = plt.get_cmap('plasma')(group['algn_fraction'].mean())  # Color-mapping to the mean algn_fraction

        # Calculate counts for each bin
        counts_x, _ = np.histogram(group[x_vals], bins=bins_x)
        counts_y, _ = np.histogram(group['fident'], bins=bins_y)

        # Manually update cumulative sum for each bin
        cum_counts_x += counts_x
        cum_counts_y += counts_y

        # Filter out bins where counts_x is zero
        non_zero_bins = counts_x > 0

        # Plot the stacked bars for each bin
        ax_histx.bar(bins_x[:-1][non_zero_bins], counts_x[non_zero_bins], width=binwidth, alpha=1, label=f'Fraction: {fraction_bin}', color=color, bottom=cum_counts_x[non_zero_bins] - counts_x[non_zero_bins])
        #ax_histy.barh(bins_y[:-1], counts_y, height=binwidth, alpha=1, label=f'Fraction: {fraction_bin}', color=color, left=cum_counts_y - counts_y)



    # Remove spines
    ax_histx.spines['top'].set_visible(False)
    ax_histx.spines['right'].set_visible(False)
    ax_histx.spines['left'].set_visible(False)
    ax_histx.spines['bottom'].set_visible(False)

    '''
    ax_histy.spines['top'].set_visible(False)
    ax_histy.spines['right'].set_visible(False)
    ax_histy.spines['left'].set_visible(False)
    ax_histy.spines['bottom'].set_visible(False)
    '''

    # format ticks 
    ticks = ax_histx.get_yticks()
    ticks = ticks[ticks != 0]
    ax_histx.set_yticks([ticks[0]])

    '''
    ticks = ax_histy.get_xticks()
    ticks = ticks[ticks != 0]
    ax_histy.set_xticks([ticks[0]])
    '''

    # add labels
    ax.set_xlabel(x_lab)
    ax.set_ylabel('Sequence Similarity')

    # return image object for color bar addition 
    return hh

def add_right_colorbar(fig, cmap, norm, panel_axes, size='5%', pad=0.05):
    """
    Add a color bar to the right side of a multi-panel figure.

    Parameters:
        - fig: Matplotlib figure
        - cmap: Colormap
        - norm: Normalize instance for mapping the data values to the colormap
        - panel_axes: List of Matplotlib axes objects for the panels
        - size: Colorbar size (default is '5%')
        - pad: Colorbar padding (default is 0.05)
    """

     # Create an axis for the color bar on the right side of the entire figure
    divider = make_axes_locatable(fig.add_subplot(1, 1, 1, frame_on=False))
    cax = divider.append_axes("right", size=size, pad=pad)

    # Create the color bar
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax)

    # Hide the axis labels and ticks
    cax.set_xticks([])
    cax.yaxis.tick_left()  # Move colorbar ticks to the left side
    return cbar

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

def multi_panel_hist_scatter(data_table1, data_table2, mp_label_array, plt_name1, plt_name2, file_name):


    fig, axs = plt.subplots(ncols=2, nrows=2)

    cdf1, fdf1 = table_to_frame(data_table1)
    cdf2, fdf2 = table_to_frame(data_table2)

    for i in range(2):
        for j in range(2):
            
            # select subplot
            ax = axs[i, j]

            # add plot name 
            if i == 0:
                if j == 0:
                    ax.set_title(plt_name1)
                else:
                    ax.set_title(plt_name2)
        
            ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
            #ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
            ax_histy = None # configured to not create the y-axis stacked bar plot 

            xtype = mp_label_array[i][j]
            if xtype == 'tcov':
                xlab = 'Target Coverage'
            else:
                xlab = 'TM-Score'
            if j % 2 == 0:   
                im = scatter_hist(cdf1, fdf1, [], xtype, xlab, ax, ax_histx, ax_histy)
            else:
                im = scatter_hist(cdf2, fdf2, [], xtype, xlab, ax, ax_histx, ax_histy)

    fig.tight_layout(pad=1)
    fig.savefig(file_name)
    
def main():

    plt.style.use('/home/gabe/matplot/BME163.mplstyle')

    parser = argparse.ArgumentParser()

    parser.add_argument('-a1','--aln_tsv1', type=str, help='path to first alignment file')
    parser.add_argument('-a2','--aln_tsv2', type=str, help='path to second alignment file')
    parser.add_argument('-c1','--controls1', type=str, help='paths to a directory of control alignments')
    parser.add_argument('-c2','--controls2', type=str, help='paths to a directory of control alignments')
    parser.add_argument('-o','--output_path', type=str, help='output path for figure')
    args = parser.parse_args()

    # generate control database statistics for each alignment file 
    control_dict1 = aa.generate_control_dictionary(args.controls1)
    control_dict2 = aa.generate_control_dictionary(args.controls2)
    
    # generate datatable with results 
    data_table1 = aa.alignment_stats(args.aln_tsv1, control_dict1)
    data_table2 = aa.alignment_stats(args.aln_tsv2, control_dict2)

    #table_to_csv(data_table2, 'dt.csv')

   # multi_panel_alignmentspace(data_table1, data_table2, 'Legionella - Human', 'wMel - Drosophila', )
    mp_label_array = [['tcov', 'tcov'],
                      ['tm-score', 'tm-score']]
    multi_panel_hist_scatter(data_table1, data_table2, mp_label_array, 'Legionella - Acanthamoeba', 'Legionella - Human', args.output_path)

if __name__ == '__main__':
    main()