# MAWS protein→DNA restoration patch (reproducibility mirror)

This directory contains a **tested restoration patch** for the protein-target DNA path of MAWS (Making Aptamers Without SELEX), based on the public `iGEM-NU-Kazakhstan/MAWS-Heidelberg-x-NU_Kazakhstan` code lineage. The patch preserves the original entropy/Kullback–Leibler selection logic and focuses on LEaP force-field loading, PDB-library handling, DNA initialization, error handling, and OpenMM topology/context synchronization.

> **Provenance and credit.** MAWS originated from the Heidelberg iGEM work and was subsequently adapted by NU Kazakhstan. This repository is a reproducibility mirror of a downstream repair and does not claim authorship of the original MAWS algorithm. The upstream MIT license is retained.

## 1. Create the environment

Linux or WSL2 is recommended.

```bash
conda env create -f environment.yml
conda activate maws-restored
```

The environment includes AmberTools/LEaP, OpenMM, NumPy, SciPy, Matplotlib, mpmath, numba, scikit-learn, and lxml.

## 2. Obtain the upstream MAWS source

```bash
git clone https://github.com/iGEM-NU-Kazakhstan/MAWS-Heidelberg-x-NU_Kazakhstan.git
cd MAWS-Heidelberg-x-NU_Kazakhstan
```

## 3. Apply the restoration patch

Download `maws-restoration.patch` from this reproducibility mirror and place it in the cloned MAWS directory, then run:

```bash
git apply --check maws-restoration.patch
git apply maws-restoration.patch
```

## 4. Prepare a standard protein PDB

Place the cleaned target PDB in the MAWS directory. Standard protein residues should be compatible with `leaprc.protein.ff14SB`.

The historical code path expects a target-specific `.frcmod` file even when no overrides are required. For the default PDB library name used by MAWS, create a minimal placeholder before the run:

```bash
printf 'remark  minimal frcmod placeholder (no overrides)\n' > PDB.frcmod
```

MAWS will generate `PDB.lib` from the input PDB through LEaP.

## 5. Reproduce the deposited LEA-K DNA run

Copy `examples/LEAK.pdb` into the MAWS directory and execute:

```bash
python MAWS.py -p LEAK.pdb -n LEAK_test_DNA -t 25 -a DNA
```

Important: in this MAWS version, `-t` is the number of **further extension steps** after the first nucleotide is selected. Therefore `-t 25` yields a **26-nt** final candidate.

Expected final DNA sequence:

```text
5'-CTTCTGGTTCTTGCTTGTCTGCCTCC-3'
```

The validated run used 5,000 configurations in the initial step, 5,000 per subsequent step, beta = 0.01, and the CUDA OpenMM platform. Its wall time was approximately 3 h 7 min on the machine used for validation.

## CPU-only systems

The exact deposited validation used CUDA. On a CPU-only system, edit the patched `Complex.py` line

```python
mm.Platform.getPlatformByName('CUDA')
```

to

```python
mm.Platform.getPlatformByName('CPU')
```

The algorithm is unchanged; runtime will generally be much longer.

## Output files

A normal run writes:
- `<job>_output.log` — run metadata and final sequence
- `<job>_entropy.log` — candidate entropy scores and whole-complex potential-energy samples
- `<job>_RESULT.pdb` — final generated complex
- Amber topology/coordinate files generated during the workflow

Do **not** interpret the logged whole-complex potential energy as a binding free energy or Kd.

## What the patch fixes

See `docs/CHANGES.md` for the code-level changes and `docs/VALIDATION.md` for the exact validation record.

## Reproducibility note

Public MAWS descendants have historically been reported as difficult to execute reproducibly on modern environments. This patch is intentionally scoped to a tested protein→DNA workflow. It should be validated independently on additional targets, random seeds, and compute backends before broad claims of generality.

## License

MIT license inherited from the upstream NU Kazakhstan repository. Original MAWS authorship and upstream contributors remain credited in the source headers and project history.
