from rdkit import Chem
from rdkit.Chem import AllChem
from openff.toolkit import Molecule, Topology, ForceField
from openff.interchange import Interchange
import os
from pathlib import Path
from collections import defaultdict
from pprint import pprint
from shutil import which
from openff.interchange.drivers import get_gromacs_energies, get_lammps_energies
from openff.interchange.drivers.all import get_summary_data


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


def read_sdf(input_sdf):
    with Chem.SDMolSupplier(input_sdf) as suppl:
        for mol in suppl:
            if mol is None:
                continue
            yield mol, mol.GetProp('_Name')


def prepare_molecule(mol_from_sdf, name):
    mol_h = Chem.AddHs(mol_from_sdf)
    if mol_h.GetNumConformers() == 0:
        generate_3d(mol_h, name)

    ligand = Molecule.from_rdkit(mol_h)
    ligand.name = name

    force_field = ForceField("openff-2.3.0.offxml")
    try:
        interchange = force_field.create_interchange(ligand.to_topology())
        os.makedirs(name, exist_ok=True)
        output_path = os.path.join(name, name)
        interchange.to_gromacs(prefix=str(output_path))
        print(f'Ligand {name} is prepared for MD simulation!')

        if which("lmp_serial"):
            pprint(get_lammps_energies(interchange).energies)

        if which("gmx"):
            pprint(get_gromacs_energies(interchange).energies)

    except Exception as e:
        print(f'Error: {e} \nCouldn\'t prepare {name}')


if __name__ == '__main__':
    input_dir = input()
    path = os.listdir(input_dir)
    for file in path:
        if '.sdf' in file:
            name_counts = defaultdict(int)
            for mol, name in read_sdf(Path(file)):
                mol_name = deduplicate(name, name_counts)
                os.makedirs(mol_name, exist_ok=True)
                prepare_molecule(mol, mol_name)
