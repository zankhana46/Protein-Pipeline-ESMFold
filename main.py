"""
Protein Structure Analysis Pipeline - top-level entry point.

Currently wired to Member 1 (Structure & Stability Analysis).
Other members (geometry, ML pockets, fusion) will be integrated here.

Quick start
-----------
    # Analyse all ESMFold structures in the default output directory:
    python main.py

    # Analyse a single structure:
    python main.py --input outputs/structures/structure_110res.pdb

    # Analyse a directory and write results to a custom folder:
    python main.py --input outputs/structures/ --output my_results/
"""
from __future__ import annotations

import argparse
from pathlib import Path

from member1_stability import StabilityAnalyzer


def run_pipeline(input_path: Path, output_dir: Path) -> None:
    print("=" * 65)
    print("  ESMFold Protein Structure Analysis Pipeline  -  Member 1")
    print("=" * 65)

    analyzer = StabilityAnalyzer(output_dir=output_dir)

    if input_path.is_file():
        report = analyzer.analyze_file(input_path)
        label  = input_path.stem
        analyzer.save_report(report, label=label)
        reports = [(label, report)]

    elif input_path.is_dir():
        reports_list = analyzer.analyze_directory(input_path)
        reports = []
        for i, rep in enumerate(reports_list, start=1):
            # Derive a label from the first PDB in the group
            lbl = Path(rep.pdb_paths[0]).stem if rep.pdb_paths else f"group{i}"
            if rep.n_structures > 1:
                lbl = f"ensemble_{rep.n_structures}structs"
            analyzer.save_report(rep, label=lbl)
            reports.append((lbl, rep))

    else:
        raise FileNotFoundError(f"Input not found: {input_path}")

    # Summary table
    print(f"\n{'-'*65}")
    print(f"  {'Label':<30} {'N':>5}  {'Stable%':>8}  {'MeanRMSF':>9}")
    print(f"  {'-'*30} {'-'*5}  {'-'*8}  {'-'*9}")
    for label, rep in reports:
        print(
            f"  {label:<30} {len(rep.residues):>5}  "
            f"{rep.fraction_stable()*100:>7.1f}%  "
            f"{rep.mean_rmsf():>8.3f}A"
        )
    print(f"\n  Results written to: {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ESMFold Protein Structure Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i",
        default=Path("outputs/structures"),
        type=Path,
        metavar="PATH",
        help="PDB file or directory of PDB files (default: outputs/structures/)",
    )
    parser.add_argument(
        "--output", "-o",
        default=Path("outputs/stability"),
        type=Path,
        metavar="DIR",
        help="Output directory for results (default: outputs/stability/)",
    )
    args = parser.parse_args()
    run_pipeline(input_path=args.input, output_dir=args.output)


if __name__ == "__main__":
    main()
