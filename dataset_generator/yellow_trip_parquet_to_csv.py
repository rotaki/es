import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.compute as pc
import glob
import sys
import time
import os
import gc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from threading import Lock

# Global lock for synchronized printing
print_lock = Lock()

def format_bytes(bytes_size):
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"

def format_throughput(bytes_per_second):
    """Convert bytes/second to human readable format"""
    return f"{format_bytes(bytes_per_second)}/s"

def progress_bar(current, total, width=50):
    """Create a text progress bar"""
    percent = current / total
    filled = int(width * percent)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{bar}] {percent*100:.1f}%"

def process_single_file(parquet_file, standard_columns, file_index, total_files, batch_start_idx):
    """Process a single parquet file and return standardized table"""
    try:
        start_time = time.time()
        
        # Read the parquet file
        table = pq.read_table(parquet_file)
        file_rows = table.num_rows
        file_size = os.path.getsize(parquet_file)
        
        # Get current column names (all lowercase)
        current_columns = {col.lower(): col for col in table.column_names}
        
        # Build list of columns for final table
        new_columns = []
        new_fields = []
        
        for target_col, target_type in standard_columns.items():
            if target_col in current_columns:
                # Column exists - get it and cast to target type
                original_col = current_columns[target_col]
                column = table[original_col]
                
                # Cast to target type if needed
                if column.type != target_type:
                    try:
                        # Special handling for timestamp columns
                        if pa.types.is_timestamp(target_type):
                            column = pc.cast(column, target_type)
                        # For numeric types, use safe casting
                        elif pa.types.is_floating(target_type) or pa.types.is_integer(target_type):
                            column = pc.cast(column, target_type, safe=False)
                        else:
                            column = pc.cast(column, target_type)
                    except:
                        # Create null column as fallback
                        column = pa.nulls(file_rows, type=target_type)
            else:
                # Column missing - create null column
                column = pa.nulls(file_rows, type=target_type)
            
            new_columns.append(column)
            new_fields.append(pa.field(target_col, target_type))
        
        # Create new table with standard schema
        standardized_table = pa.Table.from_arrays(new_columns, schema=pa.schema(new_fields))
        
        # Clear original table to free memory
        del table
        del new_columns
        
        process_time = time.time() - start_time
        
        # Print status in a clean format
        with print_lock:
            print(f"  ✓ [{batch_start_idx + file_index + 1:3d}/{total_files}] {os.path.basename(parquet_file):30s} "
                  f"| {file_rows:>10,} rows | {format_bytes(file_size):>9s} | {process_time:>4.1f}s")
        
        return {
            'table': standardized_table,
            'rows': file_rows,
            'file': parquet_file,
            'index': file_index,
            'success': True
        }
        
    except Exception as e:
        with print_lock:
            print(f"  ✗ [{batch_start_idx + file_index + 1:3d}/{total_files}] {os.path.basename(parquet_file):30s} "
                  f"| ERROR: {str(e)[:50]}")
        return {
            'table': None,
            'rows': 0,
            'file': parquet_file,
            'index': file_index,
            'success': False,
            'error': str(e)
        }

def process_batch(batch_files, standard_columns, csv_writer, target_schema, batch_num, total_batches, num_threads, total_files_processed):
    """Process a batch of files and write to CSV"""
    print(f"\n{'='*80}")
    print(f"BATCH {batch_num}/{total_batches} - Processing {len(batch_files)} files with {num_threads} threads")
    print(f"{'='*80}")
    print(f"  #  | {'File':^30s} | {'Rows':^10s} | {'Size':^9s} | Time")
    print(f"{'-'*80}")
    
    batch_start_time = time.time()
    batch_rows = 0
    batch_processed = 0
    
    # Process files in this batch in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all files in batch for processing
        futures = {
            executor.submit(process_single_file, file, standard_columns, i, 
                          sum(len(b) for b in [batch_files]), total_files_processed): i 
            for i, file in enumerate(batch_files)
        }
        
        # Collect results as they complete
        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
    
    # Sort results by index to maintain order
    results.sort(key=lambda x: x['index'])
    
    print(f"{'-'*80}")
    print("Writing to CSV...")
    
    # Write results to CSV
    write_start = time.time()
    for result in results:
        if result['success'] and result['table'] is not None:
            # Write the table
            csv_writer.write_table(result['table'])
            
            batch_rows += result['rows']
            batch_processed += 1
            
            # Free memory immediately after writing
            del result['table']
    
    write_time = time.time() - write_start
    
    # Force garbage collection after batch
    gc.collect()
    
    batch_time = time.time() - batch_start_time
    
    print(f"\n✓ Batch {batch_num} complete:")
    print(f"  • Files processed: {batch_processed}/{len(batch_files)}")
    print(f"  • Rows written: {batch_rows:,}")
    print(f"  • Batch time: {batch_time:.1f}s (read: {batch_time-write_time:.1f}s, write: {write_time:.1f}s)")
    print(f"  • Speed: {batch_rows/batch_time:,.0f} rows/sec")
    
    return batch_rows, batch_processed

def combine_yellow_taxi_files_memory_efficient(input_pattern="yellow_tripdata_*.parquet", 
                                             output_file="yellow_tripdata_combined.csv",
                                             start_year=2011,
                                             csv_batch_size=1000000,
                                             files_per_batch=10,
                                             num_threads=None):
    """
    Combine yellow taxi parquet files with memory-efficient batch processing.
    
    Args:
        input_pattern: Glob pattern for input files
        output_file: Output CSV filename
        start_year: Start from this year (default 2011)
        csv_batch_size: Rows to write at once to CSV
        files_per_batch: Number of files to process before releasing memory (default: 16)
        num_threads: Number of threads per batch (default: CPU count)
    """
    
    # Standard 19 columns - ALL LOWERCASE
    standard_columns = {
        'vendorid': pa.int64(),
        'tpep_pickup_datetime': pa.timestamp('us'),
        'tpep_dropoff_datetime': pa.timestamp('us'),
        'passenger_count': pa.float64(),
        'trip_distance': pa.float64(),
        'ratecodeid': pa.float64(),
        'store_and_fwd_flag': pa.string(),
        'pulocationid': pa.int64(),
        'dolocationid': pa.int64(),
        'payment_type': pa.int64(),
        'fare_amount': pa.float64(),
        'extra': pa.float64(),
        'mta_tax': pa.float64(),
        'tip_amount': pa.float64(),
        'tolls_amount': pa.float64(),
        'improvement_surcharge': pa.float64(),
        'total_amount': pa.float64(),
        'congestion_surcharge': pa.float64(),
        'airport_fee': pa.float64()
    }
    
    # Set number of threads
    if num_threads is None:
        num_threads = min(multiprocessing.cpu_count(), 40)  # Default to 40 threads for high performance
    
    # Find all matching files
    all_files = sorted(glob.glob(input_pattern))
    
    # Filter files from start_year onwards
    parquet_files = []
    for f in all_files:
        try:
            basename = os.path.basename(f)
            year = int(basename.split('-')[0].split('_')[-1])
            if year >= start_year:
                parquet_files.append(f)
        except:
            print(f"Warning: Could not parse year from {f}")
    
    if not parquet_files:
        print(f"No files found matching pattern: {input_pattern} from year {start_year} onwards")
        return
    
    print(f"Found {len(parquet_files)} yellow taxi files from {start_year} onwards")
    print(f"Files range from {os.path.basename(parquet_files[0])} to {os.path.basename(parquet_files[-1])}")
    
    # Get total input size
    total_input_size = sum(os.path.getsize(f) for f in parquet_files)
    print(f"Total input size: {format_bytes(total_input_size)}")
    print(f"Output will have {len(standard_columns)} columns (all lowercase)")
    print(f"Processing in batches of {files_per_batch} files with {num_threads} threads")
    
    # Create target schema for CSV writer
    target_schema = pa.schema([(name, dtype) for name, dtype in standard_columns.items()])
    
    # Start timing
    start_time = time.time()
    
    # Initialize CSV writer
    csv_writer = pcsv.CSVWriter(
        output_file,
        target_schema,
        write_options=pcsv.WriteOptions(
            include_header=True,
            batch_size=csv_batch_size,
            delimiter=','
        )
    )
    print("Created output file with header")
    
    # Process files in batches
    total_rows = 0
    total_processed = 0
    total_batches = (len(parquet_files) + files_per_batch - 1) // files_per_batch
    
    print(f"\n📊 PROCESSING PLAN:")
    print(f"  • Total files: {len(parquet_files)}")
    print(f"  • Batches: {total_batches} × {files_per_batch} files")
    print(f"  • Threads per batch: {num_threads}")
    
    for batch_num, i in enumerate(range(0, len(parquet_files), files_per_batch), 1):
        batch_files = parquet_files[i:i + files_per_batch]
        
        batch_rows, batch_processed = process_batch(
            batch_files, 
            standard_columns, 
            csv_writer, 
            target_schema, 
            batch_num, 
            total_batches,
            num_threads,
            total_processed
        )
        
        total_rows += batch_rows
        total_processed += batch_processed
        
        # Show overall progress with progress bar
        elapsed = time.time() - start_time
        rate = total_rows / elapsed if elapsed > 0 else 0
        eta = (len(parquet_files) - total_processed) / (total_processed / elapsed) if total_processed > 0 else 0
        
        print(f"\n📈 OVERALL PROGRESS:")
        print(f"  {progress_bar(total_processed, len(parquet_files))}")
        print(f"  • Files: {total_processed}/{len(parquet_files)}")
        print(f"  • Rows: {total_rows:,}")
        print(f"  • Speed: {rate:,.0f} rows/sec")
        print(f"  • ETA: {eta/60:.1f} minutes" if eta > 0 else "  • ETA: Calculating...")
        
        if batch_num < total_batches:
            print(f"\n💾 Memory released, preparing next batch...")
    
    # Close the CSV writer
    csv_writer.close()
    
    # End timing and calculate metrics
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Get output file size
    output_size = os.path.getsize(output_file) if os.path.exists(output_file) else 0
    
    # Print final report
    print("\n" + "="*80)
    print("✅ COMBINATION COMPLETE!")
    print("="*80)
    print(f"\n📊 FINAL STATISTICS:")
    print(f"  • Files processed: {total_processed}/{len(parquet_files)}")
    print(f"  • Total rows: {total_rows:,}")
    print(f"  • Total columns: {len(standard_columns)} (all lowercase)")
    print(f"  • Output file: {output_file}")
    print(f"  • Output size: {format_bytes(output_size)}")
    print(f"\n⏱️  PERFORMANCE:")
    print(f"  • Total time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} minutes)")
    print(f"  • Throughput: {format_throughput(output_size / elapsed_time)}")
    print(f"  • Speed: {output_size / (1024 * 1024) / elapsed_time:.2f} MB/s")
    print(f"  • Row rate: {total_rows / elapsed_time:,.0f} rows/second")
    print(f"  • IOPS: {output_size / elapsed_time:,.0f} bytes/second")
    print(f"\n🔧 CONFIGURATION:")
    print(f"  • Threads: {num_threads}")
    print(f"  • Files per batch: {files_per_batch}")
    print("="*80)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Combine yellow taxi parquet files into a single CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python combine_yellow_taxi.py
  python combine_yellow_taxi.py -o my_output.csv
  python combine_yellow_taxi.py -o my_output.csv -y 2015
  python combine_yellow_taxi.py -o my_output.csv -b 20 -t 2
        '''
    )
    
    parser.add_argument('-o', '--output', 
                       default='yellow_tripdata_combined.csv',
                       help='Output CSV file (default: yellow_tripdata_combined.csv)')
    
    parser.add_argument('-y', '--year', 
                       type=int, 
                       default=2011,
                       help='Start year (default: 2011)')
    
    parser.add_argument('-b', '--batch-size', 
                       type=int, 
                       default=40,
                       help='Files to process per batch (default: 40)')
    
    parser.add_argument('-t', '--threads', 
                       type=int, 
                       default=None,
                       help='Number of threads (default: CPU count, max 12)')
    
    args = parser.parse_args()
    
    # Run the combination
    combine_yellow_taxi_files_memory_efficient(
        output_file=args.output,
        start_year=args.year,
        files_per_batch=args.batch_size,
        num_threads=args.threads
    )

if __name__ == "__main__":
    main()