import sys
import argparse
from ete3 import NCBITaxa
from ete3 import PhyloTree
from ete3 import Tree, TreeStyle
ncbi = NCBITaxa()
#ncbi.update_taxonomy_database()
from Bio import Entrez
Entrez.email = "aanakamo@ucsc.edu"

### for each class represented by species in species_list, return a reference panel (list of accessions) for the class based on NCBI taxonomy
###     include all the CCGP species in the panel, then fill in the gaps (aim for 10 species per panel)
### usage: python makeRefPanel.py -s <species_list> -l <classification_level> -m <max_ref_panel_size>
### python makeRefPanel.py -s actinoperti.txt -l class -m 15
### python makeRefPanel.py -s ccgp_species.txt -l class -m 15
### python makeRefPanel.py -s mamalia.txt -l class -m 20
### python makeRefPanel.py -s actinopteri_with_gca.txt -l class -m 15 -r 7955,481459

### Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--species_list', '-s', type=str, action='store', required=True, help='list of species scientific names')
parser.add_argument('--level', '-l', type=str, action='store', required=False, help='Classification level to create the reference panel at (ie. class)')
parser.add_argument('--clade', '-c', type=int, action='store', required=False, default=None, help='Instead of using a classification level, specify the taxid of the exact clade to start at (ie. 7742, which is Vertebrata)')
parser.add_argument('--referenceSpecies', '-r', type=str, action='store', required=False, default=None, help='Specify reference/model organism(s) with annotations to include in the reference panel by a list of taxids (ie. zebrafish and stickleback would be "7955,481459")')
parser.add_argument('--max_refs', '-m', type=int, action='store', required=False, help='Maximum size of the reference panel to create')
args = parser.parse_args()
species_list=args.species_list
level=args.level
clade=args.clade
referenceSpecies=args.referenceSpecies
max_refs=args.max_refs

### read in species list
SPECIES = {}    ## key=txid, value=(species_name, acc)
CLASSES = {}    ## key=class, value=(species_name, taxid)
with open(species_list, 'r') as sp:
    for line in sp:
        species = line.strip().split("\t")[0]
        if species != "Genus Species":
            acc = line.strip().split("\t")[1]
            TR = ncbi.get_name_translator([species])
            try:
                txid = TR[species][0]
                SPECIES[txid] = (species,acc)
                lin_ids = ncbi.get_lineage(txid)
                lin_rank = ncbi.get_rank(lin_ids)
                LIN = { v:ncbi.get_taxid_translator([k])[k] for k,v in lin_rank.items()}
                if clade:
                    group = ncbi.get_taxid_translator([clade])[clade]
                else:
                    group = LIN[level]
                if CLASSES.get(group):
                    CLASSES[group].append((species, txid))
                else:
                    CLASSES[group] = [(species, txid)]
            except:
                print("Not_in_NCBI_Taxonomy: " + species)

## print the number of species in each class, make tree for each class (entire class)
print("# Classes: " + str(len(CLASSES)))
for k,v in sorted(CLASSES.items(), key=lambda x: len(x[1]), reverse=True):
    print(str(len(v)) + "\t" + k)

### use Biopython Entrez to get assembly info
def fetch_genome_accession_and_quality(tax_id):
    ## search for assembly id by taxid
    handle = Entrez.esearch(db="assembly", term=f"txid{tax_id}[Organism:exp]", retmax=100)
    record = Entrez.read(handle)
    handle.close()
    assembly_ids = record['IdList']
    ## get assembly summaries
    try:
        handle = Entrez.esummary(db="assembly", id=",".join(assembly_ids), retmax=100)
        summaries = Entrez.read(handle)
        handle.close()
    except:
        return []
    ## extract summary info
    assemblies_info = []
    for summary in summaries['DocumentSummarySet']['DocumentSummary']:
        assembly_accession = summary['AssemblyAccession']
        assembly_name = summary['AssemblyName']
        species_name = summary['SpeciesName']
        taxid = summary['Taxid']
        release_level = summary['ReleaseLevel']
        assembly_status = summary['AssemblyStatus']
        assembly_status_sort = summary['AssemblyStatusSort']
        sort_order = summary['SortOrder']
        ftp_path_genbank = summary['FtpPath_GenBank']
        ftp_path_refseq = summary['FtpPath_RefSeq']
        assemblies_info.append({
            'AssemblyAccession': assembly_accession,
            'AssemblyName' : assembly_name,
            'SpeciesName' : species_name,
            'Taxid' : taxid,
            'ReleaseLevel': release_level,
            'AssemblyStatus': assembly_status,
            'AssemblyStatusSort' : assembly_status_sort,
            'SortOrder' : sort_order,
            'FtpPath_GenBank' : ftp_path_genbank,
            'FtpPath_RefSeq' : ftp_path_refseq
        })
    return assemblies_info

### initialize the final reference panels for each class with ccgp genomes
PANEL = {}  ## key=class, value=(species_name, taxid, accession, ccgp/additional)
for k,v in sorted(CLASSES.items(), key=lambda x: len(x[1]), reverse=True):
    PANEL[k] = set()
    for sp_name,taxid in v:
        assemblies_info = fetch_genome_accession_and_quality(taxid)
        if len(assemblies_info) > 0:
            acc = SPECIES[taxid][1]
            ftp = None
            for a in assemblies_info:
                curr_a = a['AssemblyAccession']
                if curr_a == acc:
                    if "GCF" in curr_a:
                        ftp_name = a['FtpPath_RefSeq'].split("/")[-1]
                        ftp = a['FtpPath_RefSeq'] + "/" + ftp_name + "_genomic.fna.gz"
                    else:
                        ftp_name = a['FtpPath_GenBank'].split("/")[-1]
                        ftp = a['FtpPath_GenBank'] + "/" + ftp_name + "_genomic.fna.gz"
                    asm_name = a['AssemblyName'].split(".")[0]
            PANEL[k].add((sp_name, taxid, acc, asm_name, "ccgp", ftp))
        else:
            print(sp_name + " assembly not available.")
## add the model species to the reference panel, if there is one
if referenceSpecies:
    refs = [int(r) for r in referenceSpecies.split(",")]
    for taxid in refs:
        sp_name = ncbi.get_taxid_translator([taxid])[taxid]
        assemblies_info = fetch_genome_accession_and_quality(taxid)
        if len(assemblies_info) > 0:
            add = sorted(assemblies_info, key=lambda x: x['SortOrder'])[0]
            sp_name = add['SpeciesName']
            taxid = add['Taxid']
            acc = add['AssemblyAccession']
            asm_name = add['AssemblyName'].split(".")[0]
            if "GCF" in acc:
                ftp_name = add['FtpPath_RefSeq'].split("/")[-1]
                ftp = add['FtpPath_RefSeq'] + "/" + ftp_name + "_genomic.fna.gz"
            else:
                ftp_name = add['FtpPath_GenBank'].split("/")[-1]
                ftp = add['FtpPath_GenBank'] + "/" + ftp_name + "_genomic.fna.gz"
            PANEL[k].add((sp_name, taxid, acc, asm_name, "model", ftp))

### for each class, fill in additional species genomes
for k,v in sorted(CLASSES.items(), key=lambda x: len(x[1]), reverse=True):
    ## construct the entire tree for the class
    if clade:
        group = clade
    else:
        group = k
    descendants = ncbi.get_descendant_taxa(group, collapse_subspecies=True)
    class_tree = ncbi.get_topology(descendants)
    ## traverse the tree in level order, add genomes until reach max_refs limit
    for node in class_tree.traverse("levelorder"):
        if len(PANEL[k]) >= max_refs:
            break
        else:
            ## check if the current node already has any descendants in the PANEL
            ## if so, do nothing. if not, fetch a genome.
            print("Ref panel size: ", len(PANEL[k]), "(out of ", max_refs, ")")
            has_desc = False
            for species_name,taxid,accession,asm_name,ccgp,ftp in PANEL[k]:
                if node.name in ncbi.get_lineage(taxid):
                    has_desc = True
                    break
            if not has_desc:
                assemblies_info = fetch_genome_accession_and_quality(node.name)
                if len(assemblies_info) > 0:
                    add = sorted(assemblies_info, key=lambda x: x['SortOrder'])[0]
                    sp_name = add['SpeciesName']
                    taxid = add['Taxid']
                    acc = add['AssemblyAccession']
                    asm_name = add['AssemblyName'].split(".")[0]
                    if "GCF" in acc:
                        ftp_name = add['FtpPath_RefSeq'].split("/")[-1]
                        ftp = add['FtpPath_RefSeq'] + "/" + ftp_name + "_genomic.fna.gz"
                    else:
                        ftp_name = add['FtpPath_GenBank'].split("/")[-1]
                        ftp = add['FtpPath_GenBank'] + "/" + ftp_name + "_genomic.fna.gz"
                    PANEL[k].add((sp_name, taxid, acc, asm_name, "additional", ftp))

for k,v in PANEL.items():
    print("\nWriting ref panel file for class: " + k + "_" + str(max_refs) + "_RefPanel.txt")
    with open(k + "_" + str(max_refs) + "_RefPanel.txt", 'w') as rf:
        for species_name,taxid,accession,asm_name,ccgp,ftp in PANEL[k]:
            print("\t".join([str(taxid),species_name,accession,asm_name,ccgp,ftp]))
            rf.write("\t".join([str(taxid),species_name,accession,asm_name,ccgp,ftp]) + "\n")

    print("\nWriting ref panel tree file for class: " + k + "_" + str(max_refs) + "_RefPanel.nw")
    NODE_NAMES = { str(m[1]):m[3] for m in PANEL[k] }
    tree = ncbi.get_topology([t for t in NODE_NAMES.keys()])
    for node in tree.traverse():
        if node.is_leaf():
            node.name = NODE_NAMES[node.name]
        else:
            node.name = ""
    tree.resolve_polytomy(recursive=True)
    tree.write(outfile=k + "_" + str(max_refs) + "_RefPanel.nw", format=1)
    print(tree.write())


### Get an annotated tree of ccgp actinoperti (run on command line)
### cut -f1 actinoperti_taxids.txt | ete3 ncbiquery --tree --full_lineage | ete3 view --ncbi --image actinoperti_ccgp.pdf
### Get an annotated tree of ALL actinoperti (run on command line... wait no its too big)
### ete3 ncbiquery --search Actinopteri --tree --rank_limit family | ete3 view --ncbi --image actinoperti_all.pdf

### Get annotated trees for the generated reference panel of Actinopteri
### cat Actinopteri_RefPanel.txt | awk '{ print $1; }' | ete3 ncbiquery --tree --full_lineage | ete3 view --ncbi --image Actinopteri_RefPanel_fullLin.pdf
### cat Actinopteri_RefPanel.txt | awk '{ print $1; }' | ete3 ncbiquery --tree | ete3 view --ncbi --image Actinopteri_RefPanel.pdf
### cat Actinopteri_RefPanel.txt | awk '{ print $1; }' | ete3 ncbiquery --tree | ete3 view --ncbi --image Actinopteri_RefPanel.png --Iu in --Iw 24 --Ir 300

### cat Mamalia_RefPanel.txt | awk '{ print $1; }' | ete3 ncbiquery --tree | ete3 view --ncbi --image Mamalia_RefPanel.png --Iu in --Iw 24 --Ir 30
### cat Mamalia_RefPanel.txt | awk '{ print $1; }' | ete3 ncbiquery --tree --full_lineage | ete3 view --ncbi --image Mamalia_RefPanel_fullLin.pdf