# grand-pattern-bench

**Cross-language conservation law benchmarks — does the math actually hold?**

This benchmark tests whether the double-entry bookkeeping conservation law (`|Z_in| ≈ |Z_out|`) holds under stress across different configurations.

## What It Tests

The core claim: perception entries and prediction entries should stay balanced. This benchmark:

1. Runs 10,000 ticks on a 10-room graph
2. Varies dimension (8, 16, 32), window size (3, 5, 10), GC threshold (0.01, 0.05)
3. After each tick, checks conservation error `|Z_in| - |Z_out|`
4. Reports: max error, average error, ticks where conservation broke (>tolerance)

## Key Questions

1. **Does conservation hold?** (Should be 0 violations if theory is correct)
2. **Does dimension affect conservation?** (Higher dim = more room for drift?)
3. **Does window size affect it?** (Larger window = more stable predictions?)
4. **Does GC threshold affect it?** (Aggressive GC might break balance)
5. **Does vibe converge across rooms?** (Should drift toward fleet average)

## Running

```bash
cargo run --release
```

Outputs a markdown table and generates `CONCLUSIONS.md`.

## Testing

```bash
cargo test
```

12 tests covering conservation invariants, edge cases (empty graph, single room), metric sanity checks, and property tests.

## Architecture

- Pure Rust, zero external dependencies
- Deterministic pseudo-random perception generation (sin-based)
- Simple moving-average predictor with configurable window
- Magnitude-based garbage collection with configurable threshold
- Cosine similarity for vibe convergence measurement

## License

MIT
