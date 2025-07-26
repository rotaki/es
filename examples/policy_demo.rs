use es::sort_policy::*;

fn main() {
    let config = SortConfig {
        memory_mb: 4096.0,      // 4 GB
        dataset_mb: 32768.0,    // 32 GB  
        page_size_kb: 64.0,     // 64 KB
        max_threads: 32.0,
    };
    
    println!("External Sorting Policy Analysis");
    println!("================================\n");
    
    // Show all policies
    for policy in Policy::all() {
        let params = calculate_policy_parameters(policy, config);
        
        println!("{}", params.name);
        println!("  - Run Size: {:.1} MB ({} runs total)", 
                 params.run_size_mb, 
                 calculate_run_count(config.dataset_mb, params.run_size_mb) as i32);
        println!("  - Run Generation: {:.0} threads", params.run_gen_threads);
        println!("  - Merge: {:.0} threads", params.merge_threads);
        println!("  - Bottleneck: {}\n", get_bottleneck(&params));
    }
    
    // Show comparison table
    print_comparison_table(&config);
}