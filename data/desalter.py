from rdkit.Chem import MolToSmiles, MolFromSmiles, RemoveStereochemistry, SaltRemover, MolStandardize, GetMolFrags
from rdkit.Chem.rdchem import Mol as rdMol
from pathlib import Path

class MyStandardizerDesalter:
    def __init__(self) -> None:
        path = Path(__file__).absolute().parents[0].joinpath('salts.txt')
        with open(path, 'r') as file:
            salts = file.read()
        self.remover = SaltRemover.SaltRemover(defnData=salts)
        self.uncharger = MolStandardize.rdMolStandardize.Uncharger()

    def choose_fragment(self, mol:rdMol, desalted:rdMol, verbose=0):
        frags = GetMolFrags(desalted, asMols=True)    
        if len(frags) < 2:
            return desalted
        
        if 2 < len(frags) :
            assert False, f'too many fragments: orig({MolToSmiles(mol)}), desalted({MolToSmiles(desalted)})'
        
        if verbose > 0:
            print(f'2 fragment \tmol: {MolToSmiles(mol)} \tdesalted: {MolToSmiles(desalted)}')
        
        canonical_frag_0 = MolToSmiles(frags[0], canonical=True)
        canonical_frag_1 = MolToSmiles(frags[1], canonical=True) 
        if canonical_frag_0 == canonical_frag_1:
            if verbose > 0:
                print("\t\tRepeated active. Just dropped one!")
            return frags[0]
        else:
            assert False, f'assymetric fragments, manual inspection needed: orig({MolToSmiles(canonical_frag_0)}), desalted({MolToSmiles(canonical_frag_1)})'

        
    def desalt_mol(self, mol:rdMol, verbose=0):
        mol = MolStandardize.rdMolStandardize.Cleanup(mol)
        
        desalted = self.remover.StripMol(mol) 
        desalted = self.choose_fragment(mol, desalted, verbose=verbose)

        # Standardize the molecule
        RemoveStereochemistry(desalted)
        desalted = self.uncharger.uncharge(desalted)
        desalted = MolStandardize.rdMolStandardize.Cleanup(desalted)
        
        return desalted

    def desalt_smiles(self, smiles:str, verbose=0):  
        mol = MolFromSmiles(smiles)
        desalted = self.desalt_mol(mol)
        return MolToSmiles(desalted)
