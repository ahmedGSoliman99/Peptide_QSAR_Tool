# What was changed

The patch is intentionally narrow and preserves the MAWS algorithm. It repairs execution and topology construction rather than changing the entropy-selection principle.

1. **DNA/RNA force-field awareness**
   - DNA chains trigger `source leaprc.DNA.OL15`.
   - RNA chains trigger `source leaprc.RNA.OL3`.

2. **Correct LEaP object construction**
   - Empty aptamer chains are skipped before initialization.
   - A one-unit PDB-derived protein library object is loaded directly as `CHAINn = PDB` rather than incorrectly wrapped in `sequence { ... }`.
   - Nucleic-acid chains continue to be constructed through `sequence { ... }`.

3. **DNA initialization repair**
   - DNA starts as an empty chain and is initialized by the first MAWS nucleotide-selection step, preventing a preloaded `G C A T` sequence from contaminating sequence growth.

4. **Error propagation**
   - Non-zero `tleap` exit status now raises a Python exception instead of allowing the workflow to continue with invalid/missing topology files.

5. **OpenMM topology/context synchronization**
   - The simulation context is recreated after topology-changing rebuild operations.
   - `get_energy()` checks coordinate/topology atom-count consistency before evaluating energy.

6. **Validated compute backend**
   - The deposited validation run used the CUDA OpenMM platform. For a CPU-only machine, replace `CUDA` with `CPU` in the patched `Complex.py`. This portability edit does not change the MAWS selection algorithm but was not the backend used for the deposited validation run.
