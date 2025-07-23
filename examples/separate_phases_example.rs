use es::{ExternalSorter, Input, GlobalFileManager, TempDirInfo, RunGenerationStats, MergeStats};
use std::sync::Arc;

fn main() -> Result<(), String> {
    // Create test data
    let mut data = vec![
        (vec![5, 2, 3], vec![1, 2, 3, 4, 5]),
        (vec![1, 2, 3], vec![6, 7, 8, 9, 10]),
        (vec![9, 8, 7], vec![11, 12, 13, 14, 15]),
        (vec![3, 3, 3], vec![16, 17, 18, 19, 20]),
    ];
    
    // Add more data to force multiple runs
    for i in 0..100 {
        data.push((vec![i as u8, (i + 1) as u8], vec![i as u8; 10]));
    }
    
    let input = Box::new(Input { data });
    
    // Create temporary directory and file manager
    let temp_dir_info = Arc::new(TempDirInfo {
        path: std::path::PathBuf::from("./temp_sort_example"),
        should_delete: true,
    });
    std::fs::create_dir_all(&temp_dir_info.path).unwrap();
    
    let file_manager = Arc::new(GlobalFileManager::new());
    
    // Run Generation Phase
    println!("=== Run Generation Phase ===");
    let (runs, run_gen_stats) = ExternalSorter::run_generation(
        input,
        2,  // num_threads
        1024 * 1024,  // per_thread_mem (1MB)
        Arc::clone(&file_manager),
        Arc::clone(&temp_dir_info),
    )?;
    
    print_run_gen_stats(&run_gen_stats);
    
    // Merge Phase
    println!("\n=== Merge Phase ===");
    let (output, merge_stats) = ExternalSorter::merge(
        runs,
        run_gen_stats,
        4,  // num_threads for merge (can be different!)
        Arc::clone(&file_manager),
        Arc::clone(&temp_dir_info),
    )?;
    
    print_merge_stats(&merge_stats);
    
    // Verify sorted output
    println!("\n=== Verifying Output ===");
    let mut prev_key: Option<Vec<u8>> = None;
    let mut count = 0;
    
    for (key, _value) in output.iter() {
        if let Some(ref prev) = prev_key {
            assert!(prev <= &key, "Output not sorted!");
        }
        prev_key = Some(key);
        count += 1;
    }
    
    println!("Verified {} sorted entries", count);
    
    // Get overall stats from output
    let overall_stats = output.stats();
    println!("\n=== Overall Stats ===");
    println!("Total runs generated: {}", overall_stats.num_runs);
    println!("Run generation time: {:?} ms", overall_stats.run_generation_time_ms);
    println!("Merge time: {:?} ms", overall_stats.merge_time_ms);
    
    Ok(())
}

fn print_run_gen_stats(stats: &RunGenerationStats) {
    println!("Number of runs: {}", stats.num_runs);
    println!("Time: {} ms", stats.time_ms);
    
    for (i, run_info) in stats.runs_info.iter().enumerate() {
        println!("  Run {}: {} entries, {} bytes", i, run_info.entries, run_info.file_size);
    }
    
    if let Some(ref io_stats) = stats.io_stats {
        println!("I/O Stats:");
        println!("  Read: {} ops, {} bytes", io_stats.read_ops, io_stats.read_bytes);
        println!("  Write: {} ops, {} bytes", io_stats.write_ops, io_stats.write_bytes);
    }
}

fn print_merge_stats(stats: &MergeStats) {
    println!("Output runs: {}", stats.output_runs);
    println!("Time: {} ms", stats.time_ms);
    
    for (i, entries) in stats.merge_entry_num.iter().enumerate() {
        println!("  Output run {}: {} entries", i, entries);
    }
    
    if let Some(ref io_stats) = stats.io_stats {
        println!("I/O Stats:");
        println!("  Read: {} ops, {} bytes", io_stats.read_ops, io_stats.read_bytes);
        println!("  Write: {} ops, {} bytes", io_stats.write_ops, io_stats.write_bytes);
    }
}