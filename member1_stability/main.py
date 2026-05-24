"""
Member 1 - Structure & Stability Analysis  |  Entry point

Analyses ESMFold PDB structures for RMSD, RMSF, and per-residue stability.

Usage
-----
Single structure (pLDDT-based RMSF approximation):
    python -m member1_stability.main \\
        --input  outputs/structures/structure_110res.pdb \\
        --output outputs/stability/

Directory (ensemble RMSF when same-length structures are detected):
    python -m member1_stability.main \\
        --input  outputs/structures/ \\
        --output outputs/stability/

Options
-------
--input   PATH   PDB file or directory of PDB files
--output  DIR    Directory for JSON and CSV outputs   (default: outputs/stability/)
--chain   ID     Chain ID to extract               (default: first chain)
--group-tolerance INT  Max residue-count difference for ensemble grouping (default: 5)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a script or as a module
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "member1_stability"

from .stability import (
    analyze_single_structure,
    analyze_ensemble,
)
from .utils import (
    load_pdb,
    load_structures_from_dir,
    extract_ca_data,
    group_by_length,
    report_to_dataframe,
    report_to_dict,
    save_json,
    save_csv,
)


# ---------------------------------------------------------------------------
# Core pipeline logic
# ---------------------------------------------------------------------------

def run_analysis(
    input_path: Path,
    output_dir: Path,
    chain_id: str | None = None,
    group_tolerance: int = 5,
) -> None:
    """
    Run stability analysis on *input_path* (file or directory) and write
    results to *output_dir*.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load structures ─────────────────────────────────────────────────────
    if input_path.is_file():
        print(f"\n[mode] Single-structure analysis -> {input_path.name}")
        structure = load_pdb(input_path)
        ca_coords, plddt, res_names = extract_ca_data(structure, chain_id=chain_id)
        entries = [(str(input_path), ca_coords, plddt, res_names)]
    elif input_path.is_dir():
        print(f"\n[mode] Directory scan -> {input_path}")
        entries = load_structures_from_dir(input_path, pattern="*.pdb")
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")

    if not entries:
        print("[error] No structures loaded. Exiting.")
        sys.exit(1)

    # ── Group by length to detect potential ensembles ───────────────────────
    groups = group_by_length(entries, tolerance=group_tolerance)
    print(
        f"\n[group] {len(entries)} structure(s) -> "
        f"{len(groups)} analysis group(s) "
        f"(tolerance = +/-{group_tolerance} residues)"
    )

    all_reports = []

    for g_idx, group in enumerate(groups, start=1):
        n_struct = len(group)
        first_path = Path(group[0][0])
        label = first_path.stem if n_struct == 1 else f"group{g_idx}_{n_struct}structs"

        print(f"\n{'-'*60}")
        print(f"  Group {g_idx}: {n_struct} structure(s)  |  label = {label}")

        pdb_paths  = [e[0] for e in group]
        coord_list = [e[1] for e in group]
        plddt_list = [e[2] for e in group]
        names_list = [e[3] for e in group]

        # ── Choose analysis mode ─────────────────────────────────────────────
        if n_struct == 1:
            print("  [mode] Single-structure  -> pLDDT-approximated RMSF")
            report = analyze_single_structure(
                ca_coords=coord_list[0],
                plddt=plddt_list[0],
                residue_names=names_list[0],
                pdb_path=pdb_paths[0],
            )
        else:
            print(f"  [mode] Ensemble  -> true RMSF from {n_struct} Kabsch-aligned structures")
            report = analyze_ensemble(
                ca_coord_list=coord_list,
                plddt_list=plddt_list,
                residue_names_list=names_list,
                pdb_paths=pdb_paths,
            )
            print(
                f"  [rmsd] Mean pairwise RMSD = {report.mean_rmsd:.3f} A\n"
                f"  [rmsd] Matrix:\n{_format_rmsd_matrix(report.pairwise_rmsd_matrix, pdb_paths)}"
            )

        # ── Print per-group summary ──────────────────────────────────────────
        print(
            f"  [rmsf] Mean RMSF = {report.mean_rmsf():.3f} A  |  "
            f"Mean pLDDT = {report.mean_plddt():.1f}"
        )
        print(
            f"  [class] Stable={report.n_stable}  "
            f"Moderate={report.n_moderate}  "
            f"Flexible={report.n_flexible}  "
            f"({report.fraction_stable() * 100:.1f}% stable)"
        )

        # ── Save outputs ─────────────────────────────────────────────────────
        json_path = output_dir / f"{label}_stability.json"
        csv_path  = output_dir / f"{label}_stability.csv"

        save_json(report_to_dict(report), json_path)
        save_csv(report_to_dataframe(report), csv_path)

        all_reports.append((label, report))

    # ── Global summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Analysis complete  |  {len(all_reports)} report(s) written to {output_dir}")
    for label, rep in all_reports:
        print(
            f"  * {label}: {len(rep.residues)} residues  "
            f"| stable={rep.n_stable}  moderate={rep.n_moderate}  flexible={rep.n_flexible}"
        )


def _format_rmsd_matrix(matrix: "np.ndarray", paths: list[str]) -> str:
    """Pretty-print a pairwise RMSD matrix with short filename labels."""
    import numpy as np
    names = [Path(p).stem[:12] for p in paths]
    header = "         " + "  ".join(f"{n:>12}" for n in names)
    lines  = [header]
    for i, row in enumerate(matrix):
        row_str = "  ".join(f"{v:>12.3f}" for v in row)
        lines.append(f"  {names[i]:>12}  {row_str}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Member 1 - Protein Structure & Stability Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        type=Path,
        metavar="PATH",
        help="PDB file or directory of PDB files",
    )
    parser.add_argument(
        "--output", "-o",
        default=Path("outputs/stability"),
        type=Path,
        metavar="DIR",
        help="Output directory for JSON and CSV results (default: outputs/stability/)",
    )
    parser.add_argument(
        "--chain",
        default=None,
        metavar="ID",
        help="Chain ID to extract (default: first chain in model)",
    )
    parser.add_argument(
        "--group-tolerance",
        default=5,
        type=int,
        metavar="INT",
        help="Max residue-count difference for ensemble grouping (default: 5)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_analysis(
        input_path=args.input,
        output_dir=args.output,
        chain_id=args.chain,
        group_tolerance=args.group_tolerance,
    )


if __name__ == "__main__":
    main()
