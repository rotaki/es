#!/usr/bin/env python3
"""
Heatmap Visualization Script for External Sorter Benchmarks

Generates heatmaps showing performance characteristics across different
thread counts (x-axis) and memory sizes (y-axis).

Usage: python3 create_heatmaps.py <benchmark_log_file>
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from pathlib import Path

def parse_benchmark_log(log_file):
    """Parse the benchmark log file and extract performance data."""
    print(f"Parsing benchmark log from {log_file}...")
    
    results = []
    current_config = {}
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Split into individual benchmark runs
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
    
    # Calculate scaling efficiency (throughput per thread)
    df['efficiency'] = df['throughput_meps'] / df['threads']
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
                marker='o', linewidth=2, label=f'{threads} threads')
    
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
        f.write("External Sorter Benchmark Summary\n")
        f.write("=" * 40 + "\n\n")
        
        # Basic statistics
        f.write("Dataset Statistics:\n")
        f.write(f"  Total benchmark configurations: {len(df)}\n")
        f.write(f"  Thread counts tested: {sorted(df['threads'].unique())}\n")
        f.write(f"  Memory sizes tested (MB): {sorted(df['memory_mb'].unique())}\n\n")
        
        # Performance highlights
        clean_df = df.dropna(subset=['throughput_meps'])
        if not clean_df.empty:
            best_perf = clean_df.loc[clean_df['throughput_meps'].idxmax()]
            f.write("Performance Highlights:\n")
            f.write(f"  Best throughput: {best_perf['throughput_meps']:.2f} M entries/s\n")
            f.write(f"    Configuration: {best_perf['threads']} threads, {best_perf['memory_mb']}MB\n")
            f.write(f"    Run generation time: {best_perf['run_gen_time_ms']:.0f}ms\n")
            f.write(f"    Merge time: {best_perf['merge_time_ms']:.0f}ms\n\n")
        
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
        print("Usage: python3 create_heatmaps.py <benchmark_log_file>")
        sys.exit(1)
    
    log_file = sys.argv[1]
    if not Path(log_file).exists():
        print(f"Error: File {log_file} not found")
        sys.exit(1)
    
    # Parse log data
    df = parse_benchmark_log(log_file)
    
    if df.empty:
        print("No valid benchmark results found in log file")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(f"heatmaps_{Path(log_file).stem}")
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