#  Protein Mimic Screen 

This is the code used in the protein mimic screen paper. 

The taxon selection folder contains the scripts for generating the exapnsive control database. The rest of the code preforms the various analysis and plotting presented in the paper.

##  Dependency Installation 

For taxon selection scripts:
* gsutil from the Google Cloud CLI is required to run the fetch_proteomes.sh script for downloading proteomes from the AlphaFold Structure Database: https://cloud.google.com/storage/docs/gsutil_install


```
$ pip install bardapi==0.1.38 bacdive==0.3.1 biopython==1.79
```
For other scripts 

* https://github.com/steineggerlab/foldseek


```
$ pip install numpy pandas tqdm
```

## Screen Overview  

1. **Control Database Generation**  

    * Generating a new set of bacterial taxon IDs    
        1. Install and extract NCBI taxondb 
            ```
                wget -c ftp://ftp.ncbi.nih.gov/pub/taxonomy/taxdump.tar.gz 
                tar -zxvf taxdump.tar.gz
            ```
        2. Install taxonkit binaries or through conda 
            ```
                conda install -c bioconda taxonkit
            ```
            - https://bioinf.shenwei.me/taxonkit/
        3. Generate subtree with this command
            ```
                taxonkit list --ids 2 -n -r --json --data-dir='/path/to/taxdump' > bacterial_tree.json
            ```
        4. Run taxon_selector.py 
            ```
            $ pyton taxon_selector.py -t bacterial_tree.json -p directory of previously generated csvs
            ```
    * Automated host-association checking of bacterial taxon
        * strain_type_checker_bard.py: Determine if randomly selected taxon are host-associated 
            ```
            $ python strain_type_checker_bard.py -f file_from_taxon_selector.py -o output_file -b bacdive_credentials 
            ```
        - Uses bacdive api which requires login credentials
            - Signup: https://api.bacdive.dsmz.de/
            - Then use with -b email,pw
        - Follow instructions for using cookies with this unofficial bardapi https://github.com/dsdanielpark/Bard-API
            - Different accounts and connections will require different cookies 
            - For this papers use __Secure-1PSID, __Secure-1PSIDTS, and __Secure-1PSIDCC all needed to be specifed and updated 
                - Add these values or more to the cookie_dict in main()
                    ```
                    cookie_dict = {
                        "__Secure-1PSID": "value",
                        "__Secure-1PSIDTS": "value",
                        "__Secure-1PSIDCC": "value"
                        # Any cookie values you want to pass session object.
                    }
                    ```
        - The bardapi is unoffical and will sometimes lead to interuptions as max requests on an account are reached 
            - Use the -r flag with the taxid of the last submitted query to resume script from that point 
    * Downloading AlphaFold Database proteome shards for a list of taxids
        * The ID list used for the control dataset in the paper is provided in this repo as taxon_id_list.txt
        ```
        $ bash fetch_proteomes.sh taxon_id_list.txt /path/to/outputdir /path/to/logfile /path/to/missingtaxIDfile
        ```
        - Taxid file should have one taxid and species name per line seperated by a space 
        - Download directory will contain the merged proteome shards for a taxid in a folder with the species name 
        - For taxids that are missing from the AlphaFold databases the taxid is added to the missing_taxids_file and the output for the gsutil cp command to the log_file 
 
2. **Alignment analysis and visualtion**

    1.  Mask low confidence ends of protein structures prior to alignment
        ```
        $ python mask_cif_bio.py structure_directory 
        ```
        - Structure directory can be .cif and .pdb structure files
        - Output is saved a pdb files in a new 'masked' folder within the structure_direcotry path
        - enable logging of each structures masked region with '-v', prints to std_out

    2. Preform Foldseek alignments between a query proteome and a directory of control proteomes.
        ```
        $ python multi_foldseek.py -q query_proteome -d targets_database -o output_directory
        ```
        - query_proteome should be a directory of structure files, or a .tar archive
        - targets_database should be a directory of proteome archives or structure files each within a unique subfolder
        - specify threads for foldseek to use with -t INT, if not specified will use all available threads

    3.  Generate data tables of results, pull IDs of interest from results, plot fraction of freeliving proteome statistics
        ```
        $ python alignment_analysis.py  -a target_alignment -c control_alignment_directory
        ```
        - Use -h for full list of options 
        - Specify a set of structural files used to generate alignments with -p to include average pLDDT values in data table 

    4. Producing multipanel figures from paper
        ```
        $ python multipanel_space_vis.py -h 
        ```
        - Takes two alignments and two control directories as arguments, use -h for details 