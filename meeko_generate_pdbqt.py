import os
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation
from meeko import PDBQTWriterLegacy
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


def read_sdf(input_sdf):
    name_counts = defaultdict(int)
    with Chem.SDMolSupplier(input_sdf) as suppl:
        for mol in suppl:
            if mol is None:
                continue

            name = mol.GetProp('_Name') if mol.HasProp('_Name') else "unknown"
            mol_name = deduplicate(name, name_counts)
            mol_h = Chem.AddHs(mol)
            mol_3d = generate_3d(mol_h)
            
            yield mol_3d, mol_name


def read_smiles(input_smi):
    name_counts = defaultdict(int)
    with open(input_smi, 'r') as smiles:
        next(smiles)
        for mol in smiles:
            if mol is None:
                continue

            mol, name = mol.strip().split()
            mol_name = deduplicate(name, name_counts)
            mol = Chem.MolFromSmiles(mol)
            mol_h = Chem.AddHs(mol)
            mol_3d = generate_3d(mol_h)
            
            yield mol_3d, mol_name


def convert_to_pdbqt(rdkit_mol, mol_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    prepper = MoleculePreparation(
            break_macrocycles=True,    
            min_macrocycle_size=6,     
            max_macrocycle_size=15    
        )

    try:
        prepper.prepare(rdkit_mol)
        pdbqt_string = prepper.write_pdbqt_string() 
        with open(f"{out_dir}/{name}.pdbqt", "w") as f:
            f.write(pdbqt_string)       
        print(f"Converted {mol_name} to {mol_name}.pdbqt")

    except Exception as e:
        print(f"Error: {e}\nCouldn't convert {mol_name}")


input_dir = input()
path = os.listdir(input_dir)
output_dir = r'converted_pdbqts'
for file in path:
    if ('.sdf' in file) and ('_temp' not in file):
        for mol, name in read_sdf(Path(file)):
            convert_to_pdbqt(mol, name, output_dir)
    elif '.smi' in file:
        for mol, name in read_smiles(Path(file)):
            convert_to_pdbqt(mol, name, output_dir)
