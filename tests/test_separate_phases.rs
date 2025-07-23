use es::{ExternalSorter, Input, GlobalFileManager, TempDirInfo};
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
    let temp_dir_info = Arc::new(TempDirInfo {
        path: std::path::PathBuf::from("./temp_test_separate_phases"),
        should_delete: true,
    });
    std::fs::create_dir_all(&temp_dir_info.path).unwrap();
    
    let file_manager = Arc::new(GlobalFileManager::new());
    
    // Test run generation phase
    let (runs, run_gen_stats) = ExternalSorter::run_generation(
        input,
        2,  // num_threads
        512,  // per_thread_mem (very small to force multiple runs)
        Arc::clone(&file_manager),
        Arc::clone(&temp_dir_info),
    ).expect("Run generation failed");
    
    // Verify run generation stats
    assert_eq!(run_gen_stats.num_runs, runs.len());
    assert!(run_gen_stats.num_runs > 1, "Should have multiple runs with small memory");
    assert_eq!(run_gen_stats.runs_info.len(), run_gen_stats.num_runs);
    assert!(run_gen_stats.time_ms >= 0);
    
    // Test merge phase
    let (output, merge_stats) = ExternalSorter::merge(
        runs,
        run_gen_stats,
        4,  // Different thread count for merge
        Arc::clone(&file_manager),
        Arc::clone(&temp_dir_info),
    ).expect("Merge failed");
    
    // Verify merge stats
    assert!(merge_stats.output_runs > 0);
    assert_eq!(merge_stats.merge_entry_num.len(), merge_stats.output_runs);
    assert!(merge_stats.time_ms >= 0);
    
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
    
    let temp_dir_info = Arc::new(TempDirInfo {
        path: std::path::PathBuf::from("./temp_test_thread_counts"),
        should_delete: true,
    });
    std::fs::create_dir_all(&temp_dir_info.path).unwrap();
    
    let file_manager = Arc::new(GlobalFileManager::new());
    
    // Use 1 thread for run generation, 8 threads for merge
    let (runs, run_gen_stats) = ExternalSorter::run_generation(
        input,
        1,  // Single thread for run generation
        1024 * 1024,
        Arc::clone(&file_manager),
        Arc::clone(&temp_dir_info),
    ).expect("Run generation failed");
    
    let (_output, merge_stats) = ExternalSorter::merge(
        runs,
        run_gen_stats,
        8,  // Many threads for merge
        Arc::clone(&file_manager),
        Arc::clone(&temp_dir_info),
    ).expect("Merge failed");
    
    // Just verify it worked
    assert!(merge_stats.time_ms >= 0);
}