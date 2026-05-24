"""
Generate v2 structures - same proteins, saved with v2 naming
"""
import sys
sys.path.insert(0, '/root/protein_pipeline')

from common.esmfold_api import ESMFoldAPI

# Same proteins, will get same structures
proteins = {
    "insulin_v2": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN",
    "small_peptide_v2": "MKTIIALSYIFCLVFA",
    "lysozyme_v2": "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL",
}

print("=" * 60)
print("GENERATING V2 STRUCTURES")
print("=" * 60)

for name, sequence in proteins.items():
    print(f"\n{name.upper()} ({len(sequence)} residues)")
    print("-" * 60)
    
    structure = ESMFoldAPI.predict(sequence, output_dir="outputs/structures/v2")
    print(f"✓ Saved to: {structure.pdb_path}")
    print(f"  pLDDT: {structure.plddt_scores.mean():.1f}")

print("\n" + "=" * 60)
print("✓ V2 STRUCTURES GENERATED!")
print("=" * 60)
print("\nPDB files are in: outputs/structures/v2/")
