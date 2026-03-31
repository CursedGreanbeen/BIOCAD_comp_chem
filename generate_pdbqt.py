import os
from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import pybel
from openbabel import openbabel
from collections import defaultdict
from pathlib import Path


def deduplicate(name, counts):
    idx = counts[name]
    counts[name] += 1
    return f'{name}_{idx}' if idx else f'{name}'


def generate_3d(mol):
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    success = AllChem.EmbedMolecule(mol, params)
    if success == -1:
        print(f'Error: RDKit couldn\'t generate 3D')
    return mol


def rdkit_to_pybel(rdkit_mol):
    sdf_block = Chem.MolToMolBlock(rdkit_mol)
    pb_mol = pybel.readstring("sdf", sdf_block)
    return pb_mol


def read_sdf(input_sdf):
    with Chem.SDMolSupplier(input_sdf) as suppl:
        for mol in suppl:
            if mol is None:
                continue

            name = mol.GetProp('_Name') if mol.HasProp('_Name') else "unknown"
            mol_h = Chem.AddHs(mol)
            mol_3d = generate_3d(mol_h)
            # mol_3d.SetProp('_Name', name)
            yield mol_3d, name


def read_smiles(input_smi):
    with open(input_smi, 'r') as smiles:
        next(smiles)
        for mol in smiles:
            if mol is None:
                continue

            mol, name = mol.strip().split()
            mol = Chem.MolFromSmiles(mol)
            mol_h = Chem.AddHs(mol)
            mol_3d = generate_3d(mol_h)
            # mol_3d.SetProp('_Name', name) if name else mol_3d.SetProp('_Name', "unknown")
            yield mol_3d, name


def convert_to_pdbqt(rdkit_mol, mol_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    try:
        pb_mol = rdkit_to_pybel(rdkit_mol)
        pb_mol.calccharges("gasteiger")
        output_pdbqt = os.path.join(out_dir, f"{mol_name}.pdbqt")
        pb_mol.write("pdbqt", output_pdbqt, overwrite=True)
        print(f"Converted {mol_name} to {mol_name}.pdbqt")

    except Exception as e:
        print(f"Error: {e}\nCouldn't covert {mol_name}")


input_dir = input()
path = os.listdir(input_dir)
output_dir = r'converted_pdbqts'
for file in path:
    if ('.sdf' in file) and ('_temp' not in file):
        name_counts = defaultdict(int)
        for mol, name in read_sdf(Path(file)):
            convert_to_pdbqt(mol, name, output_dir)
    elif '.smi' in file:
        name_counts = defaultdict(int)
        for mol, name in read_smiles(Path(file)):
            convert_to_pdbqt(mol, name, output_dir)
