"""
Generate PDB structures for team to work on
"""
import sys
sys.path.insert(0, '/root/protein_pipeline')

from common.esmfold_api import ESMFoldAPI

# Example protein sequences
proteins = {
    "insulin": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN",
    "small_peptide": "MKTIIALSYIFCLVFA",
    "lysozyme": "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL",
}

print("=" * 60)
print("GENERATING STRUCTURES FOR TEAM")
print("=" * 60)

for name, sequence in proteins.items():
    print(f"\n{name.upper()} ({len(sequence)} residues)")
    print("-" * 60)
    
    structure = ESMFoldAPI.predict(sequence)
    print(f"✓ Saved to: {structure.pdb_path}")
    print(f"  pLDDT: {structure.plddt_scores.mean():.1f}")

print("\n" + "=" * 60)
print("✓ ALL STRUCTURES GENERATED!")
print("=" * 60)
print("\nPDB files are in: outputs/structures/")
print("\nYour team can now work on their algorithms!")
