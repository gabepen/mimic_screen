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

mask_cif_bio.py: Mask low confidence ends of protein structures prior to alignment.
```
$ python3 mask_cif_bio.py structure_directory 
```
- structure directory can be .cif and .pdb structure files
- output is saved a pdb files in a new 'masked' folder within the structure_direcotry path
- enable logging of each structures masked region with '-v', prints to std_out

multi_foldseek.py: Used to automate multiple foldseek alignments between a query proteome and a directory of proteomes.
```
$ python3 multi_foldseek.py -q query_proteome -d targets_database -o output_directory
```
- query_proteome should be a directory of structure files, or a .tar archive
- targets_database should be a directory of proteome archives or structure files each within a unique subfolder
- specify cores for foldseek to use with 


  


