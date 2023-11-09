import argparse
import os
import gzip
import pdb
from glob import glob
from Bio import PDB


class ResidueRangeSelector:
    def __init__(self, start_residue, end_residue):
        self.start_residue = start_residue
        self.end_residue = end_residue

    def accept_residue(self, residue):
        res_id = residue.get_id()[1]
        if self.start_residue <= res_id <= self.end_residue:
            return True
        return False


def keep_residues_in_range(structure_file, start_residue, end_residue, outfile):

    # determine file type for parser 
    # exits if not pdb or cif
    if '.cif' in structure_file:
        parser = PDB.MMCIFParser()
    elif '.pdb' in structure_file:
        parser = PDB.PDBParser()
    else:
        return

    # init structure object
    structure = parser.get_structure('structure', structure_file)

    # init range into range selection object
    selection = ResidueRangeSelector(start_residue, end_residue)

    class SavingSelect:
        def accept_model(self, model):
            return True

        def accept_chain(self, chain):
            return True

        def accept_residue(self, residue):
            return selection.accept_residue(residue)

        def accept_atom(self, atom):
            return True

    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(outfile, select=SavingSelect())


def get_mask_bounds(pdb):

    with open(pdb) as pf:
        atoms = pf.readlines()
        print(len(atoms))
        # remove low confidence atoms C term
        c_mask = 0
        for a in atoms:
            vals = a.split()
            try:
                if len(vals) < 9 or vals[0] != 'ATOM':
                    continue
                elif float(vals[14]) > 65:
                    c_mask = int(vals[8])
                    break
            except ValueError:
                break

        # N term
        n_mask = len(atoms)
        for a in reversed(atoms):
            vals = a.split()
            try:
                if len(vals) < 9 or vals[0] != 'ATOM':
                    continue
                elif float(vals[14]) > 65:
                    n_mask = int(vals[8])
                    break
            except ValueError:
                break

        return c_mask, n_mask




def main():

    ''' Removes low plDDT atoms without breaking chains save output as pdb files'''

    parser = argparse.ArgumentParser()
    parser.add_argument('structure_dir', metavar='dir', type=str, nargs='?', help='path to directory of .cif or .pdb files')
    parser.add_argument('-v', '--verbose', action='store_true',help='print structure file and mask range to STDOUT')
    args = parser.parse_args()

    # make path consistent
    if args.structure_dir[-1] != '/':
        structure_dir = args.structure_dir + '/'
    else:
        structure_dir = args.structure_dir

    # collect structure files
    structure_files = glob(structure_dir + '*.cif') + glob(structure_dir + '*.pdb')

    # make output directory as subdir of structure dir 
    if not os.path.isdir(structure_dir + 'masked'):
        os.mkdir(structure_dir + 'masked')

    # mask each file in directory
    for file in structure_files:


        struct_name = file.split('/')[-1]
        c_mask, n_mask = get_mask_bounds(file)

        # print ranges in verbose mode
        if args.verbose:
            print(struct_name, c_mask, n_mask)
        
        keep_residues_in_range(file, c_mask, n_mask, structure_dir + 'masked/' + struct_name)

if __name__ == '__main__':
    main()

