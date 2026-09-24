# Validation record

This folder documents the validated DNA run used in the accompanying manuscript.

## Input
- Target PDB: `examples/LEAK.pdb`
- Target sequence encoded by this PDB: `MDAKDKLKEKAKE` (13 residues; residue 7 is Leu in this deposited input)
- Aptamer type: DNA
- Initial sampling: 5,000 configurations per nucleotide
- Subsequent sampling: 5,000 configurations per nucleotide/extension direction
- `-t 25`, which produces a 26-nt final sequence because MAWS first selects one nucleotide and then performs 25 extension steps
- beta: 0.01
- OpenMM platform in the validated code path: CUDA
- DNA force field loaded by LEaP: `leaprc.DNA.OL15`

## Command

```bash
python MAWS.py -p LEAK.pdb -n LEAK_test_DNA -t 25 -a DNA
```

## Validated output
- Run start: 2025-10-27 04:17:56
- Run end: 2025-10-27 07:25:12
- Wall time: approximately 3 h 7 min
- Final 26-nt DNA sequence: `CTTCTGGTTCTTGCTTGTCTGCCTCC`
- Residue form in the MAWS output log: `DCN DTN DTN DCN DTN DGN DGN DTN DTN DCN DTN DTN DGN DCN DTN DTN DGN DTN DCN DTN DGN DCN DCN DTN DCN DCN`

The final `ENERGY` value in the entropy log is the potential energy of the entire modeled complex under the MAWS/OpenMM evaluation context. It is **not** an experimental binding energy, dissociation constant, or rigorous binding free energy.

## Scope

The deposited validation proves end-to-end generation of a full-length DNA candidate on the stated protein target using the patched code path. It does not by itself establish affinity or specificity. Independent docking, replicated molecular dynamics, and experimental binding measurements remain separate validation layers.
