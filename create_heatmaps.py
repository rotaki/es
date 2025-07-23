#!/usr/bin/env python3
"""
Heatmap Visualization Script for External Sorter Benchmarks

Generates heatmaps showing performance characteristics across different
thread counts (x-axis) and memory sizes (y-axis).

Usage: python3 create_heatmaps.py <benchmark_results_directory>

The script will parse all log files in the directory and calculate
averages across multiple runs for each configuration.
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from pathlib import Path

def parse_benchmark_directory(results_dir):
    """Parse all benchmark log files in the directory and calculate averages."""
    print(f"Parsing benchmark logs from directory {results_dir}...")
    
    all_results = []
    
    # Find all log files in the directory
    log_files = sorted(results_dir.glob("benchmark_*.log"))
    
    if not log_files:
        print(f"No benchmark log files found in {results_dir}")
        return pd.DataFrame()
    
    print(f"Found {len(log_files)} log files")
    
    # Parse each log file
    for log_file in log_files:
        # Extract configuration from filename
        filename = log_file.name
        
        # Try new format first (separate run_gen and merge threads)
        match = re.match(r'benchmark_r(\d+)_g(\d+)_m(\d+)_run(\d+)\.log', filename)
        if match:
            run_gen_threads = int(match.group(1))
            merge_threads = int(match.group(2))
            memory_mb = int(match.group(3))
            run_num = int(match.group(4))
            result = parse_single_benchmark_log(log_file, run_gen_threads, merge_threads, memory_mb, run_num)
            if result:
                all_results.append(result)
            continue
        
        # Try old format (single thread count)
        match = re.match(r'benchmark_t(\d+)_m(\d+)_run(\d+)\.log', filename)
        if match:
            threads = int(match.group(1))
            memory_mb = int(match.group(2))
            run_num = int(match.group(3))
            result = parse_single_benchmark_log(log_file, threads, threads, memory_mb, run_num)
            if result:
                all_results.append(result)
            continue
            
        print(f"  Skipping file with unexpected name format: {filename}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    if df.empty:
        return df
    
    # Calculate averages across runs for each configuration
    avg_df = calculate_averages(df)
    
    print(f"\nSuccessfully parsed {len(df)} individual runs")
    print(f"Averaged into {len(avg_df)} configurations")
    if not avg_df.empty:
        # Create a combined thread label for display
        avg_df['threads'] = avg_df['run_gen_threads'].astype(str) + '/' + avg_df['merge_threads'].astype(str)
        print(f"Thread configurations (run_gen/merge): {sorted(avg_df['threads'].unique())}")
        print(f"Memory sizes (MB): {sorted(avg_df['memory_mb'].unique())}")
    
    return avg_df

def calculate_averages(df):
    """Calculate averages across multiple runs for each configuration."""
    # Group by run_gen_threads, merge_threads and memory_mb
    grouped = df.groupby(['run_gen_threads', 'merge_threads', 'memory_mb'])
    
    # Calculate mean for numerical columns, count for validation
    avg_data = []
    
    for (run_gen_threads, merge_threads, memory_mb), group in grouped:
        avg_result = {
            'run_gen_threads': run_gen_threads,
            'merge_threads': merge_threads,
            'memory_mb': memory_mb,
            'num_runs_averaged': len(group),  # How many runs were averaged
        }
        
        # Calculate averages for each metric
        numeric_columns = ['run_gen_time_ms', 'merge_time_ms', 'total_time_ms', 
                          'throughput_meps', 'num_runs', 'total_entries', 'imbalance_ratio']
        
        for col in numeric_columns:
            if col in group.columns:
                valid_values = group[col].dropna()
                if len(valid_values) > 0:
                    avg_result[col] = valid_values.mean()
                    # Also calculate std dev for key metrics
                    if col in ['throughput_meps', 'total_time_ms']:
                        avg_result[f'{col}_std'] = valid_values.std()
                else:
                    avg_result[col] = None
            else:
                avg_result[col] = None
        
        avg_data.append(avg_result)
    
    return pd.DataFrame(avg_data)

def parse_single_benchmark_log(log_file, run_gen_threads, merge_threads, memory_mb, run_num):
    """Parse a single benchmark log file and return the results."""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"  Error reading {log_file}: {e}")
        return None
    
    # Check if benchmark failed
    if "ERROR: Benchmark failed" in content:
        print(f"  Skipping failed benchmark: run_gen={run_gen_threads}, merge={merge_threads} threads, {memory_mb}MB, run {run_num}")
        return None
    
    # Initialize result dict
    result = {
        'run_gen_threads': run_gen_threads,
        'merge_threads': merge_threads,
        'memory_mb': memory_mb,
        'run_num': run_num,
        'run_gen_time_ms': None,
        'merge_time_ms': None,
        'total_time_ms': None,
        'throughput_meps': None,
        'num_runs': None,
        'total_entries': None,
        'imbalance_ratio': None
    }
    
    # Parse metrics using same regex patterns as before
    # Parse run generation time
    run_gen_match = re.search(r'Generated \d+ runs in (\d+) ms', content)
    if run_gen_match:
        result['run_gen_time_ms'] = float(run_gen_match.group(1))
    
    # Parse merge time
    merge_match = re.search(r'Merge phase took (\d+) ms', content)
    if merge_match:
        result['merge_time_ms'] = float(merge_match.group(1))
    
    # Parse total time
    total_match = re.search(r'Sort completed in ([\d.]+) seconds', content)
    if total_match:
        result['total_time_ms'] = float(total_match.group(1)) * 1000  # Convert to ms
    
    # Parse number of runs
    runs_match = re.search(r'Sort statistics: (\d+) runs generated', content)
    if runs_match:
        result['num_runs'] = int(runs_match.group(1))
    
    # Parse total entries
    entries_match = re.search(r'Verified (\d+) entries', content)
    if entries_match:
        result['total_entries'] = int(entries_match.group(1))
    
    # Parse throughput
    throughput_match = re.search(r'(\d+)\s+[\d.]+\s*\w*B?\s+\d+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+\d+\s+([\d.]+)', content)
    if throughput_match:
        result['throughput_meps'] = float(throughput_match.group(2))
    elif result['total_entries'] and result['total_time_ms']:
        # Calculate throughput manually if not in table
        result['throughput_meps'] = result['total_entries'] / (result['total_time_ms'] / 1000) / 1_000_000
    
    # Parse imbalance ratio
    imbalance_match = re.search(r'Imbalance ratio \(max/min\): ([\d.]+)', content)
    if imbalance_match:
        result['imbalance_ratio'] = float(imbalance_match.group(1))
    
    return result

def parse_benchmark_log(log_file):
    """Parse the benchmark log file and extract performance data."""
    print(f"Parsing benchmark log from {log_file}...")
    
    results = []
    current_config = {}
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Split into individual benchmark runs - handle both old and new formats
    # Try new format first
    benchmark_blocks = re.split(r'=== BENCHMARK START: run_gen_threads=(\d+), merge_threads=(\d+), memory_mb=(\d+), run=\d+ ===', content)
    if len(benchmark_blocks) <= 1:
        # Fall back to old format
        benchmark_blocks = re.split(r'=== BENCHMARK START: threads=(\d+), memory_mb=(\d+) ===', content)
    
    # Process each benchmark block (skip first empty element)
    for i in range(1, len(benchmark_blocks), 3):
        if i + 2 >= len(benchmark_blocks):
            break
            
        threads = int(benchmark_blocks[i])
        memory_mb = int(benchmark_blocks[i + 1])
        benchmark_output = benchmark_blocks[i + 2]
        
        # Skip if benchmark failed
        if "ERROR: Benchmark failed" in benchmark_output:
            print(f"  Skipping failed benchmark: {threads} threads, {memory_mb}MB")
            continue
        
        # Initialize result dict
        result = {
            'threads': threads,
            'memory_mb': memory_mb,
            'run_gen_time_ms': None,
            'merge_time_ms': None,
            'total_time_ms': None,
            'throughput_meps': None,
            'num_runs': None,
            'total_entries': None,
            'imbalance_ratio': None
        }
        
        # Parse run generation time from direct output
        run_gen_match = re.search(r'Generated \d+ runs in (\d+) ms', benchmark_output)
        if run_gen_match:
            result['run_gen_time_ms'] = float(run_gen_match.group(1))
        
        # Parse merge time from direct output
        merge_match = re.search(r'Merge phase took (\d+) ms', benchmark_output)
        if merge_match:
            result['merge_time_ms'] = float(merge_match.group(1))
        
        # Parse total time from direct output
        total_match = re.search(r'Sort completed in ([\d.]+) seconds', benchmark_output)
        if total_match:
            result['total_time_ms'] = float(total_match.group(1)) * 1000  # Convert to ms
        
        # Parse number of runs
        runs_match = re.search(r'Sort statistics: (\d+) runs generated', benchmark_output)
        if runs_match:
            result['num_runs'] = int(runs_match.group(1))
        
        # Parse total entries
        entries_match = re.search(r'Verified (\d+) entries', benchmark_output)
        if entries_match:
            result['total_entries'] = int(entries_match.group(1))
        
        # Parse throughput from table (if available) - look for the table format
        throughput_match = re.search(r'(\d+)\s+[\d.]+\s*\w*B?\s+\d+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+\d+\s+([\d.]+)', benchmark_output)
        if throughput_match:
            result['throughput_meps'] = float(throughput_match.group(2))
        elif result['total_entries'] and result['total_time_ms']:
            # Calculate throughput manually if not in table
            result['throughput_meps'] = result['total_entries'] / (result['total_time_ms'] / 1000) / 1_000_000
        
        # Parse imbalance ratio
        imbalance_match = re.search(r'Imbalance ratio \(max/min\): ([\d.]+)', benchmark_output)
        if imbalance_match:
            result['imbalance_ratio'] = float(imbalance_match.group(1))
        
        results.append(result)
        print(f"  Parsed: {threads} threads, {memory_mb}MB - throughput: {result['throughput_meps']:.2f} M entries/s")
    
    df = pd.DataFrame(results)
    print(f"\nSuccessfully parsed {len(df)} benchmark results")
    if not df.empty:
        print(f"Thread counts: {sorted(df['threads'].unique())}")
        print(f"Memory sizes (MB): {sorted(df['memory_mb'].unique())}")
    
    return df

def create_heatmap(df, metric_col, title, output_file, fmt='.0f', cmap='viridis', show_runs=False):
    """Create a heatmap for the given metric."""
    # Pivot data for heatmap
    pivot_df = df.pivot(index='memory_mb', columns='threads', values=metric_col)
    
    # Sort indices for better visualization
    pivot_df = pivot_df.sort_index(ascending=False)  # Memory descending (high at top)
    pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)  # Threads ascending
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    if show_runs and 'num_runs' in df.columns:
        # Create custom annotations showing both metric and number of runs
        pivot_runs = df.pivot(index='memory_mb', columns='threads', values='num_runs')
        pivot_runs = pivot_runs.sort_index(ascending=False)
        pivot_runs = pivot_runs.reindex(sorted(pivot_runs.columns), axis=1)
        
        # Create annotations combining metric value and number of runs
        annot_labels = []
        for i in range(len(pivot_df.index)):
            row_labels = []
            for j in range(len(pivot_df.columns)):
                metric_val = pivot_df.iloc[i, j]
                runs_val = pivot_runs.iloc[i, j]
                if pd.notna(metric_val) and pd.notna(runs_val):
                    if fmt == '.0f':
                        row_labels.append(f'{metric_val:.0f}ms\n({runs_val:.0f} runs)')
                    else:
                        row_labels.append(f'{metric_val:.2f}\n({runs_val:.0f} runs)')
                else:
                    row_labels.append('')
            annot_labels.append(row_labels)
        
        # Create heatmap with custom annotations
        ax = sns.heatmap(pivot_df, 
                         annot=annot_labels, 
                         fmt='', 
                         cmap=cmap,
                         cbar_kws={'label': metric_col.replace('_', ' ').title()})
    else:
        # Create standard heatmap
        ax = sns.heatmap(pivot_df, 
                         annot=True, 
                         fmt=fmt, 
                         cmap=cmap,
                         cbar_kws={'label': metric_col.replace('_', ' ').title()})
    
    # Customize plot
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Number of Threads', fontsize=14)
    plt.ylabel('Memory Size (MB)', fontsize=14)
    
    # Format y-axis labels to show memory in GB for large values
    y_labels = []
    for label in ax.get_yticklabels():
        mb = int(label.get_text())
        if mb >= 1024:
            y_labels.append(f'{mb//1024}GB')
        else:
            y_labels.append(f'{mb}MB')
    ax.set_yticklabels(y_labels)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap: {output_file}")
    plt.close()

def create_performance_comparison_plots(df, output_dir):
    """Create additional performance analysis plots."""
    
    # 1. Scaling efficiency heatmap
    plt.figure(figsize=(12, 8))
    
    # Calculate scaling efficiency (throughput per total threads)
    # For configurations with separate thread counts, use the max of run_gen and merge threads
    if 'run_gen_threads' in df.columns and 'merge_threads' in df.columns:
        df['total_threads'] = df[['run_gen_threads', 'merge_threads']].max(axis=1)
        df['efficiency'] = df['throughput_meps'] / df['total_threads']
    else:
        # Fallback for old format
        df['efficiency'] = df['throughput_meps'] / df['threads'].str.split('/').str[0].astype(float)
    pivot_efficiency = df.pivot(index='memory_mb', columns='threads', values='efficiency')
    pivot_efficiency = pivot_efficiency.sort_index(ascending=False)
    pivot_efficiency = pivot_efficiency.reindex(sorted(pivot_efficiency.columns), axis=1)
    
    ax = sns.heatmap(pivot_efficiency, 
                     annot=True, 
                     fmt='.2f', 
                     cmap='RdYlBu_r',
                     cbar_kws={'label': 'Throughput per Thread (M entries/s/thread)'})
    
    plt.title('Thread Scaling Efficiency', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Threads', fontsize=14)
    plt.ylabel('Memory Size (MB)', fontsize=14)
    
    # Format y-axis labels
    y_labels = []
    for label in ax.get_yticklabels():
        mb = int(label.get_text())
        if mb >= 1024:
            y_labels.append(f'{mb//1024}GB')
        else:
            y_labels.append(f'{mb}MB')
    ax.set_yticklabels(y_labels)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scaling_efficiency_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Memory efficiency plot
    plt.figure(figsize=(10, 6))
    
    for threads in sorted(df['threads'].unique()):
        thread_data = df[df['threads'] == threads]
        plt.plot(thread_data['memory_mb'], thread_data['throughput_meps'], 
                marker='o', linewidth=2, label=f'{threads} (r/m)')
    
    plt.xlabel('Memory Size (MB)', fontsize=12)
    plt.ylabel('Throughput (M entries/s)', fontsize=12)
    plt.title('Memory vs Performance by Thread Count', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    
    # Format x-axis
    ax = plt.gca()
    x_ticks = sorted(df['memory_mb'].unique())
    x_labels = [f'{mb//1024}GB' if mb >= 1024 else f'{mb}MB' for mb in x_ticks]
    plt.xticks(x_ticks, x_labels)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'memory_performance_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_performance_summary(df, output_file):
    """Generate a performance summary report."""
    with open(output_file, 'w') as f:
        f.write("External Sorter Benchmark Summary (Averaged Results)\n")
        f.write("=" * 50 + "\n\n")
        
        # Basic statistics
        f.write("Dataset Statistics:\n")
        f.write(f"  Total benchmark configurations: {len(df)}\n")
        f.write(f"  Thread counts tested: {sorted(df['threads'].unique())}\n")
        f.write(f"  Memory sizes tested (MB): {sorted(df['memory_mb'].unique())}\n")
        if 'num_runs_averaged' in df.columns:
            f.write(f"  Runs averaged per configuration: {df['num_runs_averaged'].iloc[0]}\n")
        f.write("\n")
        
        # Performance highlights
        clean_df = df.dropna(subset=['throughput_meps'])
        if not clean_df.empty:
            best_perf = clean_df.loc[clean_df['throughput_meps'].idxmax()]
            f.write("Performance Highlights:\n")
            f.write(f"  Best throughput: {best_perf['throughput_meps']:.2f} M entries/s")
            if 'throughput_meps_std' in best_perf and pd.notna(best_perf['throughput_meps_std']):
                f.write(f" (±{best_perf['throughput_meps_std']:.2f})")
            f.write("\n")
            f.write(f"    Configuration: {best_perf['threads']} threads, {best_perf['memory_mb']}MB\n")
            f.write(f"    Run generation time: {best_perf['run_gen_time_ms']:.0f}ms\n")
            f.write(f"    Merge time: {best_perf['merge_time_ms']:.0f}ms\n")
            if 'total_time_ms_std' in best_perf and pd.notna(best_perf['total_time_ms_std']):
                f.write(f"    Total time: {best_perf['total_time_ms']:.0f}ms (±{best_perf['total_time_ms_std']:.0f}ms)\n")
            f.write("\n")
        
        # Thread scaling analysis
        f.write("Thread Scaling Analysis:\n")
        for memory in sorted(df['memory_mb'].unique()):
            mem_data = df[df['memory_mb'] == memory].dropna(subset=['throughput_meps'])
            if len(mem_data) > 1:
                single_thread = mem_data[mem_data['threads'] == 1]
                max_threads = mem_data[mem_data['threads'] == mem_data['threads'].max()]
                
                if not single_thread.empty and not max_threads.empty:
                    speedup = max_threads['throughput_meps'].iloc[0] / single_thread['throughput_meps'].iloc[0]
                    max_thread_count = max_threads['threads'].iloc[0]
                    efficiency = speedup / max_thread_count * 100
                    
                    f.write(f"  {memory}MB: {speedup:.1f}x speedup with {max_thread_count} threads ")
                    f.write(f"({efficiency:.1f}% efficiency)\n")
        f.write("\n")
        
        # Memory scaling analysis
        f.write("Memory Scaling Analysis:\n")
        for threads in sorted(df['threads'].unique()):
            thread_data = df[df['threads'] == threads].dropna(subset=['throughput_meps'])
            if len(thread_data) > 1:
                min_mem = thread_data[thread_data['memory_mb'] == thread_data['memory_mb'].min()]
                max_mem = thread_data[thread_data['memory_mb'] == thread_data['memory_mb'].max()]
                
                if not min_mem.empty and not max_mem.empty:
                    improvement = max_mem['throughput_meps'].iloc[0] / min_mem['throughput_meps'].iloc[0]
                    f.write(f"  {threads} threads: {improvement:.1f}x improvement from ")
                    f.write(f"{min_mem['memory_mb'].iloc[0]}MB to {max_mem['memory_mb'].iloc[0]}MB\n")

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 create_heatmaps.py <benchmark_results_directory>")
        sys.exit(1)
    
    results_dir = Path(sys.argv[1])
    if not results_dir.exists() or not results_dir.is_dir():
        print(f"Error: Directory {results_dir} not found or is not a directory")
        sys.exit(1)
    
    # Parse all log files in the directory
    df = parse_benchmark_directory(results_dir)
    
    if df.empty:
        print("No valid benchmark results found in log file")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(f"heatmaps_{results_dir.name}")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Creating heatmaps in {output_dir}/")
    
    # Create main heatmaps
    create_heatmap(df, 'run_gen_time_ms', 
                   'Run Generation Time Heatmap\n(Time & Number of Runs Generated)', 
                   output_dir / 'run_generation_time_heatmap.png',
                   fmt='.0f', cmap='viridis_r', show_runs=True)
    
    create_heatmap(df, 'merge_time_ms',
                   'Merge Time Heatmap\n(Lower is Better)',
                   output_dir / 'merge_time_heatmap.png',
                   fmt='.0f', cmap='viridis_r')
    
    create_heatmap(df, 'total_time_ms',
                   'Total Sort Time Heatmap\n(Lower is Better)',
                   output_dir / 'total_time_heatmap.png',
                   fmt='.0f', cmap='viridis_r')
    
    create_heatmap(df, 'throughput_meps',
                   'Throughput Heatmap\n(Higher is Better)',
                   output_dir / 'throughput_heatmap.png',
                   fmt='.2f', cmap='viridis')
    
    create_heatmap(df, 'imbalance_ratio',
                   'Partition Imbalance Ratio Heatmap\n(Lower is Better, 1.0 = Perfect Balance)',
                   output_dir / 'imbalance_ratio_heatmap.png',
                   fmt='.2f', cmap='viridis_r')
    
    create_heatmap(df, 'num_runs',
                   'Number of Runs Generated Heatmap\n(Shows Run Generation Strategy)',
                   output_dir / 'num_runs_heatmap.png',
                   fmt='.0f', cmap='viridis')
    
    # Create additional analysis plots
    create_performance_comparison_plots(df, output_dir)
    
    # Generate summary report
    generate_performance_summary(df, output_dir / 'performance_summary.txt')
    
    print("\nHeatmap generation completed!")
    print(f"All files saved in: {output_dir}/")
    print("\nGenerated files:")
    print("  - run_generation_time_heatmap.png (main focus - shows time + # runs)")
    print("  - num_runs_heatmap.png (number of runs generated)")
    print("  - merge_time_heatmap.png")
    print("  - total_time_heatmap.png") 
    print("  - throughput_heatmap.png")
    print("  - imbalance_ratio_heatmap.png")
    print("  - scaling_efficiency_heatmap.png")
    print("  - memory_performance_curves.png")
    print("  - performance_summary.txt")

if __name__ == "__main__":
    main()