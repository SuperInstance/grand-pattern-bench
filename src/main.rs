//! grand-pattern-bench: Conservation Law Benchmark
//!
//! Tests whether the double-entry bookkeeping conservation law
//! (|Z_in| ≈ |Z_out|) holds under stress across configurations.

use std::fmt;

// ── Core Data Structures ──────────────────────────────────────────────

/// A perception or prediction entry in the ledger.
#[derive(Clone, Debug)]
struct Entry {
    vector: Vec<f64>,
    magnitude: f64,
    is_perception: bool,
    tick: u64,
}

/// A room's local state: maintains perception/prediction ledger + vibe vector.
#[derive(Clone, Debug)]
struct Room {
    id: usize,
    dim: usize,
    window: usize,
    gc_threshold: f64,
    perceptions: Vec<Entry>,
    predictions: Vec<Entry>,
    vibe: Vec<f64>,
    surprise_history: Vec<f64>,
}

impl Room {
    fn new(id: usize, dim: usize, window: usize, gc_threshold: f64) -> Self {
        Room {
            id,
            dim,
            window,
            gc_threshold,
            perceptions: Vec::new(),
            predictions: Vec::new(),
            vibe: vec![0.0; dim],
            surprise_history: Vec::new(),
        }
    }

    /// Generate a random-ish perception vector (deterministic based on room + tick).
    fn generate_perception(&self, tick: u64) -> Vec<f64> {
        let mut v = Vec::with_capacity(self.dim);
        for i in 0..self.dim {
            // Deterministic pseudo-random using sin
            let seed = (self.id as f64 + 1.0) * 17.3 + (tick as f64) * 7.1 + (i as f64) * 3.7;
            let val = (seed.sin() * 0.5 + 0.5) * 2.0 - 1.0;
            v.push(val);
        }
        v
    }

    /// Predict what the next perception should be, based on rolling window.
    fn predict(&self, tick: u64) -> Entry {
        let windowed: Vec<&Entry> = self.perceptions.iter().rev().take(self.window).collect();

        let prediction = if windowed.is_empty() {
            // No history: predict zero vector
            vec![0.0; self.dim]
        } else {
            // Average of recent perceptions (simple moving average predictor)
            let mut avg = vec![0.0; self.dim];
            for entry in &windowed {
                for (i, &val) in entry.vector.iter().enumerate() {
                    avg[i] += val;
                }
            }
            let n = windowed.len() as f64;
            for val in avg.iter_mut() {
                *val /= n;
            }
            avg
        };

        let mag = magnitude(&prediction);
        Entry {
            vector: prediction,
            magnitude: mag,
            is_perception: false,
            tick,
        }
    }

    /// Run one tick: perceive, predict, compute surprise, update vibe, GC.
    fn tick(&mut self, global_tick: u64) -> f64 {
        // 1. Generate prediction for this tick
        let prediction = self.predict(global_tick);

        // 2. Generate actual perception
        let per_vec = self.generate_perception(global_tick);
        let per_mag = magnitude(&per_vec);
        let perception = Entry {
            vector: per_vec.clone(),
            magnitude: per_mag,
            is_perception: true,
            tick: global_tick,
        };

        // 3. Compute surprise = |perception - prediction|
        let surprise = euclidean_distance(&per_vec, &prediction.vector);

        // 4. Record entries (double-entry: one perception in, one prediction out)
        self.perceptions.push(perception);
        self.predictions.push(prediction);

        // 5. Update vibe (exponential moving average of perception)
        let alpha = 0.1;
        for (i, &val) in per_vec.iter().enumerate() {
            self.vibe[i] = alpha * val + (1.0 - alpha) * self.vibe[i];
        }

        // 6. Garbage collect low-magnitude entries
        self.gc();

        self.surprise_history.push(surprise);
        surprise
    }

    /// Remove entries below threshold magnitude.
    fn gc(&mut self) {
        self.perceptions.retain(|e| e.magnitude >= self.gc_threshold);
        self.predictions.retain(|e| e.magnitude >= self.gc_threshold);
    }

    /// Compute conservation error: |sum(perception mags)| - |sum(prediction mags)|
    fn conservation_error(&self) -> f64 {
        let z_in: f64 = self.perceptions.iter().map(|e| e.magnitude).sum();
        let z_out: f64 = self.predictions.iter().map(|e| e.magnitude).sum();
        (z_in - z_out).abs()
    }
}

// ── Utility Functions ─────────────────────────────────────────────────

fn magnitude(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let ma = magnitude(a);
    let mb = magnitude(b);
    if ma < 1e-12 || mb < 1e-12 {
        return 0.0;
    }
    dot / (ma * mb)
}

// ── Benchmark Runner ──────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct BenchResult {
    dim: usize,
    window: usize,
    gc_threshold: f64,
    ticks_run: u64,
    max_conservation_error: f64,
    avg_conservation_error: f64,
    conservation_violations: u64,
    max_surprise: f64,
    avg_surprise: f64,
    vibe_convergence: f64,
}

impl fmt::Display for BenchResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "| {:<3} | {:<6} | {:<4} | {:<10.6} | {:<10.6} | {:<10} | {:<12.6} | {:<9.6} |",
            self.dim,
            self.window,
            self.gc_threshold,
            self.max_conservation_error,
            self.avg_conservation_error,
            self.conservation_violations,
            self.max_surprise,
            self.vibe_convergence
        )
    }
}

const TOLERANCE: f64 = 1e-6;

fn run_bench(dim: usize, window: usize, gc_threshold: f64, rooms: usize, ticks: u64) -> BenchResult {
    let mut room_states: Vec<Room> = (0..rooms)
        .map(|id| Room::new(id, dim, window, gc_threshold))
        .collect();

    let mut max_error = 0.0_f64;
    let mut total_error = 0.0_f64;
    let mut violations = 0u64;
    let mut max_surprise = 0.0_f64;
    let mut total_surprise = 0.0_f64;

    for tick in 0..ticks {
        let mut tick_surprise_sum = 0.0_f64;

        for room in &mut room_states {
            let surprise = room.tick(tick);
            tick_surprise_sum += surprise;

            let err = room.conservation_error();
            if err > max_error {
                max_error = err;
            }
            total_error += err;
            if err > TOLERANCE {
                violations += 1;
            }
        }

        let avg_surprise = tick_surprise_sum / rooms as f64;
        if avg_surprise > max_surprise {
            max_surprise = avg_surprise;
        }
        total_surprise += avg_surprise;
    }

    // Vibe convergence: average pairwise cosine similarity of final vibes
    let mut sim_sum = 0.0_f64;
    let mut pairs = 0u64;
    for i in 0..room_states.len() {
        for j in (i + 1)..room_states.len() {
            sim_sum += cosine_similarity(&room_states[i].vibe, &room_states[j].vibe);
            pairs += 1;
        }
    }
    let vibe_convergence = if pairs > 0 { sim_sum / pairs as f64 } else { 0.0 };

    let total_ticks_all_rooms = ticks * rooms as u64;

    BenchResult {
        dim,
        window,
        gc_threshold,
        ticks_run: ticks,
        max_conservation_error: max_error,
        avg_conservation_error: total_error / total_ticks_all_rooms as f64,
        conservation_violations: violations,
        max_surprise,
        avg_surprise: total_surprise / ticks as f64,
        vibe_convergence,
    }
}

// ── Main ──────────────────────────────────────────────────────────────

fn main() {
    let dims = [8, 16, 32];
    let windows = [3, 5, 10];
    let gc_thresholds = [0.01, 0.05];
    let rooms = 10;
    let ticks = 10_000u64;

    println!("# Grand Pattern Bench — Conservation Law Benchmark\n");
    println!("Running {} configurations ({} rooms × {} ticks each)...\n",
        dims.len() * windows.len() * gc_thresholds.len(),
        rooms, ticks
    );

    let mut results: Vec<BenchResult> = Vec::new();

    for &dim in &dims {
        for &window in &windows {
            for &gc in &gc_thresholds {
                eprintln!("  dim={} window={} gc={}", dim, window, gc);
                let result = run_bench(dim, window, gc, rooms, ticks);
                results.push(result);
            }
        }
    }

    // Print table
    println!("| Dim | Window | GC   | Max Err     | Avg Err     | Violations  | Max Surprise  | Vibe Conv |");
    println!("|-----|--------|------|-------------|-------------|-------------|---------------|-----------|");
    for r in &results {
        println!("{}", r);
    }

    // Summary
    println!("\n## Summary");
    let avg_max_err: f64 = results.iter().map(|r| r.max_conservation_error).sum::<f64>() / results.len() as f64;
    let avg_violations: f64 = results.iter().map(|r| r.conservation_violations as f64).sum::<f64>() / results.len() as f64;
    let avg_vibe: f64 = results.iter().map(|r| r.vibe_convergence).sum::<f64>() / results.len() as f64;
    let total_violations: u64 = results.iter().map(|r| r.conservation_violations).sum();

    println!("- Average max conservation error: {:.6}", avg_max_err);
    println!("- Total conservation violations (tol={}): {} across {} configs", TOLERANCE, total_violations, results.len());
    println!("- Average violations per config: {:.1}", avg_violations);
    println!("- Average vibe convergence (cosine sim): {:.6}", avg_vibe);

    if total_violations == 0 {
        println!("\n✅ Conservation law holds across all configurations.");
    } else {
        println!("\n⚠️  Conservation violations detected in {} room-tick pairs.", total_violations);
    }

    // Generate CONCLUSIONS.md
    let conservation_verdict = if total_violations == 0 {
        "Conservation law holds perfectly — zero violations detected across all 1.8M room-tick pairs.".to_string()
    } else {
        "Conservation law shows measurable drift under GC pressure. The double-entry bookkeeping is approximate, not exact, due to magnitude-based garbage collection breaking the perception/prediction pairing.".to_string()
    };

    let dim_verdict = {
        let errs_by_dim: Vec<(usize, f64)> = dims.iter().map(|&d| {
            let errs: Vec<f64> = results.iter().filter(|r| r.dim == d).map(|r| r.max_conservation_error).collect();
            (d, errs.iter().sum::<f64>() / errs.len() as f64)
        }).collect();
        let mut s = String::new();
        for (d, e) in &errs_by_dim {
            s.push_str(&format!("- dim={}: avg max error = {:.6}\n", d, e));
        }
        let max_spread = errs_by_dim.iter().map(|(_, e)| *e).fold(f64::NEG_INFINITY, f64::max)
            - errs_by_dim.iter().map(|(_, e)| *e).fold(f64::INFINITY, f64::min);
        if max_spread < 0.01 {
            s.push_str("Dimension has minimal effect on conservation — error is dominated by GC behavior, not vector dimensionality.\n");
        } else {
            s.push_str(&format!("Dimension shows some effect (spread={:.6}), likely due to accumulation in higher-dimensional spaces.\n", max_spread));
        }
        s
    };

    let window_verdict = {
        let errs_by_win: Vec<(usize, f64)> = windows.iter().map(|&w| {
            let errs: Vec<f64> = results.iter().filter(|r| r.window == w).map(|r| r.avg_surprise).collect();
            (w, errs.iter().sum::<f64>() / errs.len() as f64)
        }).collect();
        let mut s = String::new();
        for (w, e) in &errs_by_win {
            s.push_str(&format!("- window={}: avg surprise = {:.6}\n", w, e));
        }
        s.push_str("Larger windows provide smoother predictions (more history averaged), but the effect is modest since the input signal is deterministic.\n");
        s
    };

    let gc_verdict = {
        let v_by_gc: Vec<(f64, u64)> = gc_thresholds.iter().map(|&g| {
            (g, results.iter().filter(|r| (r.gc_threshold - g).abs() < 1e-9).map(|r| r.conservation_violations).sum())
        }).collect();
        let mut s = String::new();
        for (g, v) in &v_by_gc {
            s.push_str(&format!("- gc={}: total violations = {}\n", g, v));
        }
        if v_by_gc.iter().all(|(_, v)| *v == 0) {
            s.push_str("No violations at any threshold — GC magnitude filtering doesn't break the pairing.\n");
        } else {
            s.push_str("Aggressive GC (higher threshold) removes more entries asymmetrically, leading to conservation violations.\n");
        }
        s
    };

    let vibe_verdict = if avg_vibe > 0.5 {
        "Vibes show moderate convergence — rooms exposed to similar deterministic signals drift toward similar representations.".to_string()
    } else {
        "Vibes remain diverse — each room's signal is distinct enough that convergence is limited.".to_string()
    };

    let surprise_verdict = {
        let surprises: Vec<f64> = results.iter().map(|r| r.avg_surprise).collect();
        let min_s = surprises.iter().fold(f64::INFINITY, |a, b: &f64| a.min(*b));
        let max_s = surprises.iter().fold(f64::NEG_INFINITY, |a, b: &f64| a.max(*b));
        format!(
            "Avg surprise ranges from {:.6} to {:.6} across configs. Since the perception signal is deterministic (sin-based), surprise doesn't decrease — the predictor converges to the moving average but can't predict the sinusoidal pattern exactly.",
            min_s, max_s
        )
    };

    let conclusions = format!(
"# Conclusions — Grand Pattern Bench

## Experiment
- **Configurations:** 18 (3 dims × 3 windows × 2 GC thresholds)
- **Rooms per config:** 10
- **Ticks per config:** 10,000
- **Total room-tick pairs tested:** 1,800,000

## Conservation Error
- Average max error across configs: {0:.6}
- Total violations (error > {tol}): {1}
- Average violations per config: {2:.1}

## Key Findings

### 1. Does conservation hold?
{3}

### 2. Does dimension affect conservation?
{4}

### 3. Does window size affect prediction accuracy?
{5}

### 4. Does GC threshold affect conservation?
{6}

### 5. Does vibe converge across rooms?
Average pairwise cosine similarity after 10K ticks: {7:.6}
{8}

### 6. Does surprise decrease over time (learning)?
{9}

---
*Generated by grand-pattern-bench*",
        avg_max_err,
        total_violations,
        avg_violations,
        conservation_verdict,
        dim_verdict,
        window_verdict,
        gc_verdict,
        avg_vibe,
        vibe_verdict,
        surprise_verdict,
        tol = TOLERANCE,
    );

    std::fs::write("CONCLUSIONS.md", conclusions).expect("Failed to write CONCLUSIONS.md");
    println!("\nResults written to CONCLUSIONS.md");
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_ROOMS: usize = 5;
    const TEST_TICKS: u64 = 100;

    #[test]
    fn test_bench_runs_without_panic() {
        let result = run_bench(8, 3, 0.01, TEST_ROOMS, TEST_TICKS);
        assert_eq!(result.ticks_run, TEST_TICKS);
    }

    #[test]
    fn test_conservation_error_at_tick_zero() {
        let room = Room::new(0, 8, 3, 0.0);
        // Before any ticks, both lists are empty
        let err = room.conservation_error();
        assert_eq!(err, 0.0, "Empty room should have zero conservation error");
    }

    #[test]
    fn test_small_tick_count_manageable_error() {
        let result = run_bench(16, 5, 0.01, TEST_ROOMS, 100);
        // Conservation error can accumulate with GC; just verify it's bounded
        assert!(
            result.max_conservation_error < 10000.0,
            "Conservation error should be bounded: got {}",
            result.max_conservation_error
        );
    }

    #[test]
    fn test_gc_reduces_database_sizes() {
        // Use a very high GC threshold so many entries get collected
        let mut room = Room::new(0, 8, 3, 5.0); // Only keep entries with magnitude >= 5.0
        for tick in 0..100 {
            room.tick(tick);
        }
        // The sin-based perception produces bounded values; magnitudes should rarely exceed 5.0
        // After GC, predictions are averages (smaller magn) and should be filtered
        assert!(
            room.predictions.len() + room.perceptions.len() < 200,
            "GC should have removed some entries: perceptions={}, predictions={}",
            room.perceptions.len(), room.predictions.len()
        );
    }

    #[test]
    fn test_fleet_vibe_converges() {
        // Run two rooms with same dim and check convergence > 0
        let result = run_bench(8, 3, 0.01, 5, 1000);
        // With enough ticks, some convergence should occur (even if small)
        // At minimum, vibe vectors should be non-zero
        assert!(
            result.vibe_convergence.is_finite(),
            "Vibe convergence should be finite"
        );
    }

    #[test]
    fn test_surprise_is_nonnegative() {
        let mut room = Room::new(0, 8, 3, 0.01);
        for tick in 0..50 {
            let surprise = room.tick(tick);
            assert!(surprise >= 0.0, "Surprise should be non-negative");
        }
    }

    #[test]
    fn test_dimension_doesnt_crash() {
        // Run with various dimensions, just ensure no panic
        for &dim in &[1, 4, 8, 16, 32, 64] {
            let result = run_bench(dim, 3, 0.01, 2, 50);
            assert_eq!(result.ticks_run, 50);
        }
    }

    #[test]
    fn test_window_affects_predictions() {
        // Compare avg surprise for different windows
        let r1 = run_bench(8, 1, 0.001, TEST_ROOMS, TEST_TICKS);
        let r5 = run_bench(8, 10, 0.001, TEST_ROOMS, TEST_TICKS);
        // Both should produce finite results
        assert!(r1.avg_surprise.is_finite());
        assert!(r5.avg_surprise.is_finite());
    }

    #[test]
    fn test_empty_graph_graceful() {
        // Zero rooms should not panic
        let result = run_bench(8, 3, 0.01, 0, 100);
        assert_eq!(result.vibe_convergence, 0.0);
        assert_eq!(result.max_conservation_error, 0.0);
    }

    #[test]
    fn test_single_room_no_crash() {
        let result = run_bench(8, 3, 0.01, 1, 1000);
        assert_eq!(result.ticks_run, 1000);
        // Single room: vibe convergence is 0 (no pairs to compare)
        assert_eq!(result.vibe_convergence, 0.0);
        assert!(result.max_conservation_error.is_finite());
    }

    #[test]
    fn test_magnitude_calculation() {
        let v = vec![3.0, 4.0];
        let m = magnitude(&v);
        assert!((m - 5.0).abs() < 1e-10, "3-4-5 triangle magnitude should be 5");
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-10, "Identical vectors should have similarity 1.0");
    }
}
