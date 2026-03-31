from rdkit import Chem
from rdkit.Chem import AllChem
from openff.toolkit import Molecule, Topology, ForceField
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
from openff.units import unit


def deduplicate(name, counts):
    idx = counts[name]
    counts[name] += 1
    return f'{name}_{idx}' if idx else f'{name}'


def generate_3d(mol, name):
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    success = AllChem.EmbedMolecule(mol, params)
    if success == -1:
        print(f'Error: RDKit couldn\'t generate 3D for {name}')
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def read_sdf(input_sdf):
    name_counts = defaultdict(int)
    with Chem.SDMolSupplier(input_sdf) as suppl:
        for mol in suppl:
            if mol is None:
                continue

            name = mol.GetProp('_Name')
            mol_name = deduplicate(name, name_counts)
            yield mol, mol_name


def prepare_molecule(mol_from_sdf, name):
    mol_h = Chem.AddHs(mol_from_sdf)
    if mol_h.GetNumConformers() == 0:
        generate_3d(mol_h, name)

    ligand = Molecule.from_rdkit(mol_h)
    ligand.name = name
    print(f"Preparing {name}")

    force_field = ForceField("openff-2.1.0.offxml")
    try:
        interchange = force_field.create_interchange(ligand.to_topology())
        print(f'Interchange for {name} is created')
        interchange.box = [4, 4, 4] * unit.nanometer
        print(f'Box for {name} is created')

        os.makedirs(name, exist_ok=True)
        output_path = os.path.join(name, name)
        interchange.to_gromacs(prefix=str(output_path))

        old_top = os.path.join(name, f"{name}.top")
        new_top = os.path.join(name, "topol.top")

        if os.path.exists(old_top):
            os.rename(old_top, new_top)

        print(f'Ligand {name} is prepared for MD simulation!')

    except Exception as e:
        print(f'Error: {e} \nCouldn\'t prepare {name}')


import subprocess


def validate_with_grompp(name):
    mdp_path = os.path.join(f"{name}_pointenergy.mdp")
    output_dir = name

    result = subprocess.run(
        ["gmx", "grompp",
         "-f", mdp_path,
         "-c", f"{name}.gro",
         "-p", "topol.top",
         "-o", "validate.tpr",
         "-maxwarn", "1"],
        cwd=output_dir,
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print(f"Validation passed for {name}")
    else:
        print(f"Validation failed for {name}")
        print(result.stderr)


if __name__ == '__main__':
    print('Enter path: ')
    input_dir = input()
    path = os.listdir(input_dir)
    for file in path:
        if '.sdf' in file:
            for mol, name in read_sdf(Path(file)):
                prepare_molecule(mol, name)
                # validate_with_grompp(name)
