use es::{ExternalSorter, Input, RunsOutput, SortOutput, SortStats, TempDirInfo};
use std::sync::Arc;

#[test]
fn test_separate_run_generation_and_merge() {
    // Create test data
    let mut data = vec![
        (vec![5, 2, 3], vec![1, 2, 3, 4, 5]),
        (vec![1, 2, 3], vec![6, 7, 8, 9, 10]),
        (vec![9, 8, 7], vec![11, 12, 13, 14, 15]),
        (vec![3, 3, 3], vec![16, 17, 18, 19, 20]),
    ];

    // Add more data to force multiple runs with low memory
    for i in 0..50 {
        data.push((vec![i as u8, (i + 1) as u8], vec![i as u8; 10]));
    }

    let input = Box::new(Input { data: data.clone() });

    // Create temporary directory and file manager
    let temp_dir_info = Arc::new(TempDirInfo::new("./temp_test_separate_phases", true));

    // Test run generation phase
    let (runs, run_gen_stats) = ExternalSorter::run_generation(
        input,
        2,   // num_threads
        512, // per_thread_mem (very small to force multiple runs)
        temp_dir_info.as_ref(),
    )
    .expect("Run generation failed");

    // Verify run generation stats
    assert_eq!(run_gen_stats.num_runs, runs.len());
    assert!(
        run_gen_stats.num_runs > 1,
        "Should have multiple runs with small memory"
    );
    assert_eq!(run_gen_stats.runs_info.len(), run_gen_stats.num_runs);

    // Test merge phase
    let (merged_runs, merge_stats) = ExternalSorter::merge(
        runs,
        4, // Different thread count for merge
        temp_dir_info.as_ref(),
    )
    .expect("Merge failed");

    // Verify merge stats
    assert!(merge_stats.output_runs > 0);
    assert_eq!(merge_stats.merge_entry_num.len(), merge_stats.output_runs);

    // Combine stats and create output
    let sort_stats = SortStats {
        num_runs: run_gen_stats.num_runs,
        runs_info: run_gen_stats.runs_info,
        run_generation_time_ms: Some(run_gen_stats.time_ms),
        merge_entry_num: merge_stats.merge_entry_num,
        merge_time_ms: Some(merge_stats.time_ms),
        run_generation_io_stats: run_gen_stats.io_stats,
        merge_io_stats: merge_stats.io_stats,
    };

    let output = Box::new(RunsOutput {
        runs: merged_runs,
        stats: sort_stats,
    });

    // Verify output is sorted
    let mut prev_key: Option<Vec<u8>> = None;
    let mut count = 0;

    for (key, _value) in output.iter() {
        if let Some(ref prev) = prev_key {
            assert!(prev <= &key, "Output not sorted!");
        }
        prev_key = Some(key);
        count += 1;
    }

    assert_eq!(count, data.len(), "Output should have all input records");
}

#[test]
fn test_different_thread_counts() {
    // Test that we can use different thread counts for run generation and merge
    let data = vec![
        (vec![5, 2, 3], vec![1, 2, 3]),
        (vec![1, 2, 3], vec![4, 5, 6]),
        (vec![9, 8, 7], vec![7, 8, 9]),
    ];

    let input = Box::new(Input { data });

    let temp_dir_info = Arc::new(TempDirInfo::new("./temp_test_thread_counts", true));

    // Use 1 thread for run generation, 8 threads for merge
    let (runs, _run_gen_stats) = ExternalSorter::run_generation(
        input,
        1, // Single thread for run generation
        1024 * 1024,
        temp_dir_info.as_ref(),
    )
    .expect("Run generation failed");

    let (_merged_runs, merge_stats) = ExternalSorter::merge(
        runs,
        8, // Many threads for merge
        temp_dir_info.as_ref(),
    )
    .expect("Merge failed");

    // Just verify it worked - merge completed
}
