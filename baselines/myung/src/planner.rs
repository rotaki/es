//! Eq. 2 adaptive resource allocation from Myung §3.2.
//!
//! Given (data size D, memory budget M, user-requested threads T, device
//! interleaving floor C), enforce M >= T * sqrt(C * D). If infeasible, the
//! paper's preference is to raise M; our benchmarks cap M via cgroup, so we
//! reduce T uniformly (single pool for both run formation and merge).

/// Plan output used by the run-formation and merge stages.
#[derive(Clone, Debug)]
pub struct MyungPlan {
    /// Effective thread count (after Eq. 2 adjustment).
    pub threads: usize,
    /// Thread count the user requested (logged for transparency).
    pub user_threads: usize,
    /// Total memory budget (bytes).
    pub memory_bytes: u64,
    /// Bytes reserved for the upfront sample buffer. Subtracted from M
    /// before computing per-thread budgets — mirrors CrocSort's explicit
    /// reservation for sparse indexes.
    pub sample_reserve_bytes: u64,
    /// NVMe interleaving floor C (bytes, from fio calibration).
    pub interleave_floor_bytes: u64,
    /// Input data size (bytes).
    pub data_size_bytes: u64,
    /// Whether the user-requested (M, T) was feasible without adjustment.
    pub initially_feasible: bool,
    /// Per-thread memory floor sqrt(C * D).
    pub per_thread_floor_bytes: u64,
}

impl MyungPlan {
    /// Compute a plan. C and D are in bytes. `sample_reserve_bytes` is the
    /// memory carved out for the sampler before splitting M across threads.
    pub fn new(
        data_size_bytes: u64,
        memory_bytes: u64,
        user_threads: usize,
        interleave_floor_bytes: u64,
        sample_reserve_bytes: u64,
    ) -> Self {
        assert!(user_threads > 0, "user_threads must be > 0");
        assert!(memory_bytes > 0, "memory_bytes must be > 0");
        assert!(data_size_bytes > 0, "data_size_bytes must be > 0");
        assert!(
            interleave_floor_bytes > 0,
            "interleave_floor_bytes must be > 0"
        );
        assert!(
            sample_reserve_bytes < memory_bytes,
            "sample_reserve_bytes ({}) must be < memory_bytes ({})",
            sample_reserve_bytes,
            memory_bytes
        );

        let usable_memory_bytes = memory_bytes - sample_reserve_bytes;

        // Per-thread memory floor: (M - sample) / T >= sqrt(C * D)
        // Use floating-point sqrt — precision-safe at these magnitudes.
        let cd_product = (interleave_floor_bytes as f64) * (data_size_bytes as f64);
        let per_thread_floor = cd_product.sqrt().ceil() as u64;

        // Max T such that usable_M / T >= per_thread_floor:
        //   T <= usable_M / per_thread_floor
        let max_feasible_threads = (usable_memory_bytes / per_thread_floor).max(1) as usize;

        let (threads, initially_feasible) = if user_threads <= max_feasible_threads {
            (user_threads, true)
        } else {
            (max_feasible_threads, false)
        };

        MyungPlan {
            threads,
            user_threads,
            memory_bytes,
            sample_reserve_bytes,
            interleave_floor_bytes,
            data_size_bytes,
            initially_feasible,
            per_thread_floor_bytes: per_thread_floor,
        }
    }

    /// Memory available to run formation + merge after carving out the
    /// sample reserve. This is the budget the per-thread split is based on.
    pub fn usable_memory_bytes(&self) -> u64 {
        self.memory_bytes - self.sample_reserve_bytes
    }

    /// Per-thread memory ((M - sample) / T).
    pub fn per_thread_memory_bytes(&self) -> u64 {
        self.usable_memory_bytes() / self.threads as u64
    }

    /// Expected total number of runs R = D / (M/T) under load-sort-store (E=1).
    pub fn expected_num_runs(&self) -> u64 {
        (self.data_size_bytes + self.per_thread_memory_bytes() - 1) / self.per_thread_memory_bytes()
    }

    /// Expected merge read-block size b = (M/T) / R.
    pub fn expected_merge_block_bytes(&self) -> u64 {
        let r = self.expected_num_runs();
        if r == 0 {
            self.per_thread_memory_bytes()
        } else {
            self.per_thread_memory_bytes() / r
        }
    }

    pub fn log_summary(&self) {
        eprintln!("[myung/planner] Eq. 2 adaptive allocation");
        eprintln!("  D           = {} bytes", self.data_size_bytes);
        eprintln!("  M           = {} bytes", self.memory_bytes);
        eprintln!(
            "  sample rsv  = {} bytes (usable M = {})",
            self.sample_reserve_bytes,
            self.usable_memory_bytes()
        );
        eprintln!("  C (floor)   = {} bytes", self.interleave_floor_bytes);
        eprintln!(
            "  sqrt(C*D)   = {} bytes (per-thread memory floor)",
            self.per_thread_floor_bytes
        );
        eprintln!(
            "  user T      = {}, effective T = {} (feasible={})",
            self.user_threads, self.threads, self.initially_feasible
        );
        eprintln!(
            "  expected R  = {}, block b = {} bytes (floor C = {})",
            self.expected_num_runs(),
            self.expected_merge_block_bytes(),
            self.interleave_floor_bytes
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;
    const KIB: u64 = 1 << 10;

    #[test]
    fn config_a_feasible_8gib_16threads() {
        // D=200 GiB, M=8 GiB, T=16, C=64 KiB  (plan §Part 1b example A)
        let plan = MyungPlan::new(200 * GIB, 8 * GIB, 16, 64 * KIB, 0);
        assert!(plan.initially_feasible);
        assert_eq!(plan.threads, 16);
        // block >> 64 KiB
        assert!(plan.expected_merge_block_bytes() > 64 * KIB);
    }

    #[test]
    fn config_b_barely_feasible_2gib_16threads() {
        let plan = MyungPlan::new(200 * GIB, 2 * GIB, 16, 64 * KIB, 0);
        assert!(plan.initially_feasible);
        assert_eq!(plan.threads, 16);
        // block around 80 KiB, just above the 64 KiB floor
        let b = plan.expected_merge_block_bytes();
        assert!(b >= 64 * KIB, "block {} KiB < 64 KiB", b / KIB);
    }

    #[test]
    fn config_c_infeasible_2gib_32threads_reduces_t() {
        // sqrt(C*D) = sqrt(64 KiB * 200 GiB) ≈ 118.63 MB
        // max T = floor(2 GiB / 118.63 MB) = 18 (plan prose said ~16, off by ~10%)
        let plan = MyungPlan::new(200 * GIB, 2 * GIB, 32, 64 * KIB, 0);
        assert!(!plan.initially_feasible);
        assert_eq!(plan.threads, 18, "expected T reduced to 18");
    }

    #[test]
    fn config_d_infeasible_1gib_16threads_reduces_t() {
        // max T = floor(1 GiB / 118.63 MB) = 9 (plan prose said ~8)
        let plan = MyungPlan::new(200 * GIB, 1 * GIB, 16, 64 * KIB, 0);
        assert!(!plan.initially_feasible);
        assert_eq!(plan.threads, 9, "expected T reduced to 9");
    }

    #[test]
    fn sample_reserve_shrinks_per_thread_budget() {
        // 5% reserve out of 8 GiB → usable = 7.6 GiB; per-thread = 7.6 GiB / 16.
        let plan = MyungPlan::new(200 * GIB, 8 * GIB, 16, 64 * KIB, 8 * GIB / 20);
        assert!(plan.initially_feasible);
        assert_eq!(plan.threads, 16);
        let usable = 8 * GIB - 8 * GIB / 20;
        assert_eq!(plan.usable_memory_bytes(), usable);
        assert_eq!(plan.per_thread_memory_bytes(), usable / 16);
    }

    #[test]
    fn sample_reserve_can_force_thread_reduction() {
        // sqrt(C*D) ≈ 118.63 MB; 2 GiB usable → max T = 18.
        // Reserve 1 GiB so usable = 1 GiB → max T = 9 (matches config_d).
        let plan = MyungPlan::new(200 * GIB, 2 * GIB, 32, 64 * KIB, GIB);
        assert!(!plan.initially_feasible);
        assert_eq!(plan.threads, 9);
    }
}
