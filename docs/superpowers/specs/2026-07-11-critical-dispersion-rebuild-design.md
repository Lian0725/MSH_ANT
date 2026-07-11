# Critical Dispersion Rebuild Reliability Design

## Scope

Only the archived 04-dispersion rebuild and validation path is changed. Azimuthal correction, inversion, map masking, scientific thresholds, and existing result data remain untouched.

## Rebuild contract

`rebuild_04dispersion_from_03cc_stackdata.sh` will use the actual `2014/1D_1D` layout, read station coordinates from `01RawData/2014/MetaData/1D`, and pass `1D.*.dat` explicitly to the GPU runner. All paths remain overridable through environment variables.

Parallel shard process IDs will be collected and waited individually. Any failed shard makes the rebuild fail. After successful extraction, the rebuild invokes the verifier with the exact DAT, curve, NPZ, log, and shard-count paths.

## Conversion contract

When StationXML is requested, every converted source and receiver must have a StationXML coordinate. Missing coordinates, unreadable stacks, invalid stack shapes, or an empty conversion selection are errors. The converter returns nonzero when any conversion error occurs; distance-filtered pairs remain legitimate skips.

## Extraction status contract

Pair processing has three states:

- `success`: at least one usable group or phase pick was produced.
- `no_pick`: processing completed normally but QC retained no usable pick; its curves and NPZ remain valid outputs.
- `error`: image construction, interpolation, model execution, or output writing raised an exception.

On `error`, partial outputs for that pair are removed rather than replaced by all-zero placeholder files. A shard exits nonzero if it encounters any `error`. A DAT glob matching zero inputs is an error, while a resume run that finds all matched inputs already complete remains successful.

Resume validation rejects zero-byte, unreadable, or legacy NPZ files carrying a nonempty `failure_reason`, so historical placeholder failures are retried.

## Verification contract

The verifier accepts explicit directories and a configurable expected shard count. It rejects:

- zero DAT inputs;
- missing or extra curve/NPZ pairs;
- zero-byte outputs;
- nonempty NPZ `failure_reason` markers;
- malformed sampled curves or NPZ arrays;
- processing failures or tracebacks in logs;
- missing shard logs or completion lines.

It prints a JSON report and returns 0 only when the complete contract passes.

## Testing

One consolidated test module will cover the shell argument wiring and child-status propagation, converter coordinate/error behavior, runner empty-input/resume/error semantics, and verifier success/failure contracts. Tests use temporary directories and small synthetic files; they do not run the CNN or modify archived data.
