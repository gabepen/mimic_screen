#  Protein Mimic Screen 

This is the code used in the protein mimic screen paper. 

The taxon selection folder contains the scripts for generating the exapnsive control database. The rest of the code preforms the various analysis and plotting presented in the paper.

##  Dependency Installation 

For taxon selection scripts 
```
$ pip install bardapi==0.1.38 bacdive==0.3.1 biopython==1.79
```
For other scripts 
```
$ pip install numpy pandas tqdm
```
https://github.com/steineggerlab/foldseek


## Usage 

### Alignment analysis and visualtion

mask_cif_bio.py: Mask low confidence ends of protein structures prior to alignment.
```
$ python mask_cif_bio.py structure_directory 
```
- structure directory can be .cif and .pdb structure files
- output is saved a pdb files in a new 'masked' folder within the structure_direcotry path
- enable logging of each structures masked region with '-v', prints to std_out

multi_foldseek.py: Used to automate multiple foldseek alignments between a query proteome and a directory of proteomes.
```
$ python multi_foldseek.py -q query_proteome -d targets_database -o output_directory
```
- query_proteome should be a directory of structure files, or a .tar archive
- targets_database should be a directory of proteome archives or structure files each within a unique subfolder
- specify threads for foldseek to use with -t INT, if not specified will use all available threads

alignment_analysis.py: Generate data tables of results, pull IDs of interest from results, plot fraction of freeliving proteome statistics
```
$ python alignment_analysis.py  -a target_alignment -c control_alignment_directory
```
- Use -h for full list of options 
- Specify a set of structural files used to generate alignments with -p to include average pLDDT values in data table 

multipanel_space_vis.py: Used to produce multipanel figures from paper.
```
$ python multipanel_space_vis.py -h 
```
- Takes two alignments and two control directories as arguments, use -h for details 

### Expansive database generation 

eutils_suite.py: custom EntrezAPI command object for use in other scripts 
- contains examples in main() for testing alternative query functions 

taxon_selector.py: Runs the random selection of taxon from a tree
```
$ pyton taxon_selector.py -t tree -p directory of previously generated csv from tree 
```
- Need to first generate bacterial_tree.json to pass to -t flag:
1. Install and extract NCBI taxondb 
    - ```
        wget -c ftp://ftp.ncbi.nih.gov/pub/taxonomy/taxdump.tar.gz 
        tar -zxvf taxdump.tar.gz
      ```
2. Install taxonkit binaries or through conda 
    - ```
        conda install -c bioconda taxonkit
      ```