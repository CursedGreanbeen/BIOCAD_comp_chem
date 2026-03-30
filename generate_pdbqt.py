import os
from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import pybel
from openbabel import openbabel
from pathlib import Path


def no_hydrogens(mol):
    if any(atom.GetSymbol() == 'H' for atom in mol.GetAtoms()):
        return None
    return True


def check_2d(mol):
    try:
        conf = mol.GetConformer()
        pos = conf.GetPositions()
        if all(pos[:, 2] == 0):
            return True
    except ValueError as e:
        print(f'Error: {e}')


def generate_3d(mol):
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    success = AllChem.EmbedMolecule(mol, params)
    if success == -1:
        print(f'Error: RDKit couldn\'t generate 3D')


def read_sdf(input_sdf):
    with Chem.SDWriter(f'{input_sdf.stem}_temp.sdf') as output_sdf:
        with Chem.SDMolSupplier(input_sdf) as suppl:
            for mol in suppl:
                if mol is None:
                    print('Invalid SDF')
                    continue
                name = mol.GetProp('_Name')
                if no_hydrogens(mol):
                    mol = Chem.AddHs(mol)
                if check_2d(mol):
                    generate_3d(mol)
                mol.SetProp('_Name', name)
                output_sdf.write(mol)
    return f'{input_sdf.stem}_temp.sdf'


def read_smiles(input_smi):
    with Chem.SDWriter(f'{input_smi.stem}_temp.sdf') as output_sdf:
        with open(input_smi, 'r') as smiles:
            next(smiles)
            for mol in smiles:
                if mol is None:
                    print('Invalid SMILES')
                    continue
                mol, name = mol.strip().split()
                mol = Chem.MolFromSmiles(mol)
                mol_h = Chem.AddHs(mol)
                generate_3d(mol_h)
                mol_h.SetProp('_Name', name)
                output_sdf.write(mol_h)
    return f'{input_smi.stem}_temp.sdf'


def convert_to_pdbqt(input_file, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for mol_sdf in pybel.readfile("sdf", input_file):
        try:
            mol_name = mol_sdf.title
            mol_sdf.calccharges("gasteiger")
            output_pdbqt = os.path.join(out_dir, f"{mol_name}.pdbqt")
            mol_sdf.write("pdbqt", output_pdbqt, overwrite=True)
            print(f"Converted {mol_name}.sdf to {mol_name}.pdbqt")
        except Exception as e:
            print(f"Error: {e}")
    os.remove(input_file)


input_dir = input()
path = os.listdir(input_dir)
output_dir = r'converted_pdbqts'
for file in path:
    if ('.sdf' in file) and ('_temp' not in file):
        result = read_sdf(Path(file))
        convert_to_pdbqt(result, output_dir)
    elif '.smi' in file:
        result = read_smiles(Path(file))
        convert_to_pdbqt(result, output_dir)
