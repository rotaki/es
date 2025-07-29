//! Parquet file input with Direct I/O support

use crate::file::SharedFd;
use crate::{
    IoStatsTracker, SortInput, aligned_reader::AlignedChunkReader, order_preserving_encoding::*,
};
use arrow::array::Array;
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::path::Path;
use std::sync::Arc;

/// Configuration for Parquet Direct I/O reading
#[derive(Clone)]
pub struct ParquetDirectConfig {
    /// Column indices to use as key (will be concatenated)
    pub key_columns: Vec<usize>,
    /// Column indices to use as value (will be concatenated)  
    pub value_columns: Vec<usize>,
}

impl Default for ParquetDirectConfig {
    fn default() -> Self {
        Self {
            key_columns: vec![0],
            value_columns: vec![1],
        }
    }
}

/// ParquetInputDirect reads key-value pairs from a Parquet file using Direct I/O
pub struct ParquetInputDirect {
    fd: Arc<SharedFd>,
    config: ParquetDirectConfig,
    num_rows: usize,
    num_row_groups: usize,
}

impl ParquetInputDirect {
    /// Create a new ParquetInputDirect from a Parquet file with GlobalFileManager
    pub fn new(path: impl AsRef<Path>, config: ParquetDirectConfig) -> Result<Self, String> {
        let path = path.as_ref().to_path_buf();

        if !path.exists() {
            return Err(format!("Parquet file does not exist: {:?}", path));
        }

        if config.key_columns.is_empty() {
            return Err("At least one key column must be specified".to_string());
        }

        if config.value_columns.is_empty() {
            return Err("At least one value column must be specified".to_string());
        }

        let fd = Arc::new(SharedFd::new_from_path(&path).map_err(|e| {
            format!(
                "Failed to open file with Direct I/O: {}: {}",
                path.display(),
                e
            )
        })?);

        // Use ManagedAlignedChunkReader to read metadata
        let chunk_reader = AlignedChunkReader::new(fd.clone())?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(chunk_reader)
            .map_err(|e| format!("Failed to create reader builder: {}", e))?;

        let metadata = builder.metadata();
        let num_rows = metadata.file_metadata().num_rows() as usize;
        let num_row_groups = metadata.num_row_groups();

        // Validate schema
        let schema = builder.schema();
        let num_cols = schema.fields().len();

        // Check that all column indices are valid
        for &col in &config.key_columns {
            if col >= num_cols {
                return Err(format!(
                    "Key column index {} is out of bounds (file has {} columns)",
                    col, num_cols
                ));
            }
        }

        for &col in &config.value_columns {
            if col >= num_cols {
                return Err(format!(
                    "Value column index {} is out of bounds (file has {} columns)",
                    col, num_cols
                ));
            }
        }

        Ok(Self {
            fd,
            config,
            num_rows,
            num_row_groups,
        })
    }

    /// Create with default configuration
    pub fn new_default(path: impl AsRef<Path>) -> Result<Self, String> {
        Self::new(path, ParquetDirectConfig::default())
    }

    /// Get the number of rows
    pub fn len(&self) -> usize {
        self.num_rows
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.num_rows == 0
    }
}

impl SortInput for ParquetInputDirect {
    fn create_parallel_scanners(
        &self,
        num_scanners: usize,
        io_tracker: Option<IoStatsTracker>,
    ) -> Vec<Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + Send>> {
        if num_scanners == 0 || self.is_empty() {
            return vec![];
        }

        // Simple partitioning: divide row groups among scanners
        let mut scanners = Vec::new();
        let groups_per_scanner = self.num_row_groups.div_ceil(num_scanners);

        for i in 0..num_scanners {
            let start_group = i * groups_per_scanner;
            let end_group = ((i + 1) * groups_per_scanner).min(self.num_row_groups);

            if start_group >= self.num_row_groups {
                break;
            }

            let scanner = ParquetPartitionDirect {
                fd: self.fd.clone(),
                config: self.config.clone(),
                start_row_group: start_group,
                end_row_group: end_group,
                current_batch: None,
                current_row: 0,
                reader: None,
                io_stats: io_tracker.clone(),
            };

            scanners.push(Box::new(scanner) as Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + Send>);
        }

        scanners
    }
}

/// Iterator over a partition of row groups using Direct I/O
struct ParquetPartitionDirect {
    fd: Arc<SharedFd>,
    config: ParquetDirectConfig,
    start_row_group: usize,
    end_row_group: usize,
    current_batch: Option<RecordBatch>,
    current_row: usize,
    reader: Option<Box<dyn arrow::record_batch::RecordBatchReader + Send>>,
    io_stats: Option<IoStatsTracker>,
}

impl ParquetPartitionDirect {
    fn init_reader(&mut self) -> Option<()> {
        // Need to get file_manager - add it to ParquetPartitionDirect struct
        let chunk_reader =
            AlignedChunkReader::new_with_tracker(self.fd.clone(), self.io_stats.clone()).ok()?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(chunk_reader).ok()?;

        // Select only our row groups
        let row_groups: Vec<usize> = (self.start_row_group..self.end_row_group).collect();
        let reader = builder.with_row_groups(row_groups).build().ok()?;

        self.reader = Some(Box::new(reader));
        Some(())
    }
}

impl Iterator for ParquetPartitionDirect {
    type Item = (Vec<u8>, Vec<u8>);

    fn next(&mut self) -> Option<Self::Item> {
        // Initialize reader on first access
        if self.reader.is_none() {
            self.init_reader()?;
        }

        loop {
            // Try to get row from current batch
            if let Some(ref batch) = self.current_batch {
                if self.current_row < batch.num_rows() {
                    // Extract and concatenate key columns
                    let mut key = Vec::new();
                    for (i, &col_idx) in self.config.key_columns.iter().enumerate() {
                        if i > 0 {
                            key.push(0); // Separator byte
                        }
                        let col_bytes = extract_bytes_from_batch(batch, col_idx, self.current_row)?;
                        key.extend_from_slice(&col_bytes);
                    }

                    // Extract and concatenate value columns
                    let mut value = Vec::new();
                    for (i, &col_idx) in self.config.value_columns.iter().enumerate() {
                        if i > 0 {
                            value.push(0); // Separator byte
                        }
                        let col_bytes = extract_bytes_from_batch(batch, col_idx, self.current_row)?;
                        value.extend_from_slice(&col_bytes);
                    }

                    self.current_row += 1;
                    return Some((key, value));
                }
            }

            // Need next batch
            match self.reader.as_mut()?.next() {
                Some(Ok(batch)) => {
                    self.current_batch = Some(batch);
                    self.current_row = 0;
                }
                _ => return None,
            }
        }
    }
}

// Order-preserving encoding functions are now imported from order_preserving_encoding module

/// Extract bytes from a column in a record batch
fn extract_bytes_from_batch(batch: &RecordBatch, column: usize, row: usize) -> Option<Vec<u8>> {
    use arrow::array::AsArray;

    let array = batch.column(column);

    match array.data_type() {
        DataType::Binary => {
            let binary_array = array.as_binary::<i32>();
            Some(binary_array.value(row).to_vec())
        }
        DataType::LargeBinary => {
            let binary_array = array.as_binary::<i64>();
            Some(binary_array.value(row).to_vec())
        }
        DataType::Utf8 => {
            let string_array = array.as_string::<i32>();
            Some(string_array.value(row).as_bytes().to_vec())
        }
        DataType::LargeUtf8 => {
            let string_array = array.as_string::<i64>();
            Some(string_array.value(row).as_bytes().to_vec())
        }
        DataType::Int32 => {
            let int_array = array.as_primitive::<arrow::datatypes::Int32Type>();
            let value = int_array.value(row);
            Some(i32_to_order_preserving_bytes(value).to_vec())
        }
        DataType::Int64 => {
            let int_array = array.as_primitive::<arrow::datatypes::Int64Type>();
            let value = int_array.value(row);
            Some(i64_to_order_preserving_bytes(value).to_vec())
        }
        DataType::Float32 => {
            let float_array = array.as_primitive::<arrow::datatypes::Float32Type>();
            let value = float_array.value(row);
            Some(f32_to_order_preserving_bytes(value).to_vec())
        }
        DataType::Float64 => {
            let float_array = array.as_primitive::<arrow::datatypes::Float64Type>();
            let value = float_array.value(row);
            Some(f64_to_order_preserving_bytes(value).to_vec())
        }
        DataType::Date32 => {
            // Date32 is days since epoch
            let date_array = array.as_primitive::<arrow::datatypes::Date32Type>();
            let value = date_array.value(row);
            // Convert to since CE (Jan 1, Year 1)
            let days_since_ce = unix_epoch_to_ce_days(value);
            Some(i32_to_order_preserving_bytes(days_since_ce).to_vec())
        }
        DataType::Decimal128(_, _) => {
            let decimal_array = array.as_primitive::<arrow::datatypes::Decimal128Type>();
            let value = decimal_array.value(row);
            Some(i128_to_order_preserving_bytes(value).to_vec())
        }
        _ => None,
    }
}

pub fn unix_epoch_to_ce_days(unix_epoch_days: i32) -> i32 {
    // Convert Unix epoch days to CE days
    unix_epoch_days + 719_163 // 719_163 is days from 1970-01-01 to 0001-01-01
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExternalSorter, Sorter};
    use arrow::array::{
        BinaryArray, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray,
    };
    use arrow::datatypes::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use parquet::arrow::arrow_writer::ArrowWriterOptions;
    use parquet::basic::{Compression, Encoding, GzipLevel};
    use parquet::file::properties::WriterProperties;
    use std::collections::HashMap;
    use std::fs::File;
    use std::path::PathBuf;
    use std::sync::Arc;

    fn test_dir() -> PathBuf {
        let dir = std::env::temp_dir().join("parquet_input_direct_tests");
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Create a large Parquet file with multiple row groups
    fn create_large_parquet(
        path: &PathBuf,
        num_records: usize,
        records_per_row_group: usize,
    ) -> Result<(), String> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("data", DataType::Utf8, false),
        ]));

        // Configure writer to create multiple row groups
        let props = WriterProperties::builder()
            .set_max_row_group_size(records_per_row_group)
            .set_compression(Compression::SNAPPY)
            .build();

        let file = File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;

        let mut writer = ArrowWriter::try_new_with_options(
            file,
            schema.clone(),
            ArrowWriterOptions::new().with_properties(props),
        )
        .map_err(|e| format!("Failed to create writer: {}", e))?;

        // Write data in batches
        let batch_size = 1000;
        for batch_start in (0..num_records).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(num_records);
            let _batch_records = batch_end - batch_start;

            let ids: Vec<String> = (batch_start..batch_end)
                .map(|i| format!("id_{:08}", i))
                .collect();
            let data: Vec<String> = (batch_start..batch_end)
                .map(|i| format!("data_for_record_{}", i))
                .collect();

            let id_array = StringArray::from(ids);
            let data_array = StringArray::from(data);

            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(id_array), Arc::new(data_array)],
            )
            .map_err(|e| format!("Failed to create batch: {}", e))?;

            writer
                .write(&batch)
                .map_err(|e| format!("Failed to write batch: {}", e))?;
        }

        writer
            .close()
            .map_err(|e| format!("Failed to close writer: {}", e))?;

        Ok(())
    }

    /// Create a Parquet file with different data types
    fn create_mixed_types_parquet(path: &PathBuf) -> Result<(), String> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("string_col", DataType::Utf8, false),
            Field::new("int_col", DataType::Int32, false),
            Field::new("float_col", DataType::Float64, false),
            Field::new("binary_col", DataType::Binary, false),
        ]));

        let strings = vec!["zebra", "apple", "mango", "banana", "orange"];
        let ints = vec![5, 1, 3, 2, 4];
        let floats = vec![5.5, 1.1, 3.3, 2.2, 4.4];
        let binaries: Vec<&[u8]> = vec![b"zzz", b"aaa", b"mmm", b"bbb", b"ooo"];

        let string_array = StringArray::from(strings);
        let int_array = Int32Array::from(ints);
        let float_array = Float64Array::from(floats);
        let binary_array = BinaryArray::from(binaries);

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(string_array),
                Arc::new(int_array),
                Arc::new(float_array),
                Arc::new(binary_array),
            ],
        )
        .map_err(|e| format!("Failed to create batch: {}", e))?;

        let file = File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;

        let mut writer = ArrowWriter::try_new(file, schema, None)
            .map_err(|e| format!("Failed to create writer: {}", e))?;

        writer
            .write(&batch)
            .map_err(|e| format!("Failed to write batch: {}", e))?;

        writer
            .close()
            .map_err(|e| format!("Failed to close writer: {}", e))?;

        Ok(())
    }

    #[test]
    fn test_parquet_input_large_file() {
        let path = test_dir().join("large_parquet.parquet");

        // Create a large file with multiple row groups
        let num_records = 10_000;
        let records_per_row_group = 1_000;
        create_large_parquet(&path, num_records, records_per_row_group).unwrap();

        // Read with ParquetInputDirect
        let input = ParquetInputDirect::new(
            &path,
            ParquetDirectConfig {
                key_columns: vec![0],
                value_columns: vec![1],
            },
        )
        .unwrap();
        assert_eq!(input.len(), num_records);

        // Test partitioning with different partition counts
        for num_partitions in [1, 2, 4, 8, 16] {
            let partitions = input.create_parallel_scanners(num_partitions, None);
            assert!(partitions.len() <= num_partitions);
            assert!(!partitions.is_empty());

            // Collect all entries
            let mut all_entries = Vec::new();
            for partition in partitions {
                let entries: Vec<_> = partition.collect();
                all_entries.extend(entries);
            }

            // Verify we got all records
            assert_eq!(all_entries.len(), num_records);

            // Verify all records are unique
            let mut seen = HashMap::new();
            for (key, _value) in all_entries {
                let key_str = String::from_utf8_lossy(&key);
                *seen.entry(key_str.to_string()).or_insert(0) += 1;
            }

            for (key, count) in seen {
                assert_eq!(count, 1, "Key {} appeared {} times", key, count);
            }
        }

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_parquet_input_with_external_sorter() {
        let path = test_dir().join("sort_test.parquet");

        // Create file with 5000 random records
        let num_records = 5000;
        create_large_parquet(&path, num_records, 500).unwrap();

        // Sort using external sorter
        let input = ParquetInputDirect::new(
            &path,
            ParquetDirectConfig {
                key_columns: vec![0],
                value_columns: vec![1],
            },
        )
        .unwrap();
        let mut sorter = ExternalSorter::new(4, 1024 * 1024); // 4 threads, 1MB buffer

        let output = sorter.sort(Box::new(input)).unwrap();

        // Verify sorted order
        let entries: Vec<_> = output.iter().collect();
        assert_eq!(entries.len(), num_records);

        // Check that entries are sorted
        for i in 1..entries.len() {
            assert!(
                entries[i - 1].0 <= entries[i].0,
                "Not sorted at position {}: {:?} > {:?}",
                i,
                String::from_utf8_lossy(&entries[i - 1].0),
                String::from_utf8_lossy(&entries[i].0)
            );
        }

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_parquet_input_mixed_types() {
        let path = test_dir().join("mixed_types.parquet");

        // Create file with mixed types
        create_mixed_types_parquet(&path).unwrap();

        // Read with ParquetInputDirect - it should use first two columns
        let input = ParquetInputDirect::new(
            &path,
            ParquetDirectConfig {
                key_columns: vec![0],
                value_columns: vec![1],
            },
        )
        .unwrap();
        assert_eq!(input.len(), 5);

        // Collect all entries
        let partitions = input.create_parallel_scanners(1, None);
        let entries: Vec<_> = partitions.into_iter().flatten().collect();
        assert_eq!(entries.len(), 5);

        // Verify string column is used as key
        let keys: Vec<String> = entries
            .iter()
            .map(|(k, _)| String::from_utf8_lossy(k).to_string())
            .collect();

        assert!(keys.contains(&"zebra".to_string()));
        assert!(keys.contains(&"apple".to_string()));
        assert!(keys.contains(&"mango".to_string()));

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_parquet_input_empty_file() {
        let path = test_dir().join("empty.parquet");

        // Create empty file
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("value", DataType::Utf8, false),
        ]));

        let file = File::create(&path).unwrap();
        let writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.close().unwrap();

        // Read with ParquetInputDirect
        let input = ParquetInputDirect::new(
            &path,
            ParquetDirectConfig {
                key_columns: vec![0],
                value_columns: vec![1],
            },
        )
        .unwrap();
        assert_eq!(input.len(), 0);
        assert!(input.is_empty());

        // Test partitioning
        let partitions = input.create_parallel_scanners(4, None);
        assert_eq!(partitions.len(), 0);

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_parquet_input_single_column_error() {
        let path = test_dir().join("single_column.parquet");

        // Create file with only one column
        let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Utf8, false)]));

        let keys = vec!["a", "b", "c"];
        let key_array = StringArray::from(keys);

        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(key_array)]).unwrap();

        let file = File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        // Should fail to create ParquetInputDirect
        let result = ParquetInputDirect::new(
            &path,
            ParquetDirectConfig {
                key_columns: vec![0],
                value_columns: vec![1],
            },
        );
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.contains("out of bounds"));
        }

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_parquet_input_compression() {
        let path = test_dir().join("compressed.parquet");

        // Create compressed file
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("value", DataType::Utf8, false),
        ]));

        let props = WriterProperties::builder()
            .set_compression(Compression::GZIP(GzipLevel::default()))
            .set_encoding(Encoding::DELTA_BYTE_ARRAY)
            .build();

        let file = File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new_with_options(
            file,
            schema.clone(),
            ArrowWriterOptions::new().with_properties(props),
        )
        .unwrap();

        // Write compressed data
        for i in 0..100 {
            let key = format!("key_{:03}", i);
            let value = format!("value_{}", i);

            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(StringArray::from(vec![key])),
                    Arc::new(StringArray::from(vec![value])),
                ],
            )
            .unwrap();

            writer.write(&batch).unwrap();
        }

        writer.close().unwrap();

        // Read compressed file
        let input = ParquetInputDirect::new(
            &path,
            ParquetDirectConfig {
                key_columns: vec![0],
                value_columns: vec![1],
            },
        )
        .unwrap();
        assert_eq!(input.len(), 100);

        let partitions = input.create_parallel_scanners(1, None);
        let entries: Vec<_> = partitions.into_iter().flatten().collect();
        assert_eq!(entries.len(), 100);

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_parquet_input_parallel_partitions() {
        use std::thread;

        let path = test_dir().join("parallel_test.parquet");

        // Create file with multiple row groups
        create_large_parquet(&path, 1000, 100).unwrap();

        let input = ParquetInputDirect::new(
            &path,
            ParquetDirectConfig {
                key_columns: vec![0],
                value_columns: vec![1],
            },
        )
        .unwrap();

        // Get multiple partitions
        let partitions = input.create_parallel_scanners(4, None);
        assert_eq!(partitions.len(), 4);

        // Process partitions in parallel
        let handles: Vec<_> = partitions
            .into_iter()
            .map(|mut partition| {
                thread::spawn(move || {
                    let mut count = 0;
                    let mut keys = Vec::new();

                    for (key, _value) in partition.by_ref() {
                        keys.push(String::from_utf8_lossy(&key).to_string());
                        count += 1;
                    }

                    (count, keys)
                })
            })
            .collect();

        // Collect results
        let mut total_count = 0;
        let mut all_keys = Vec::new();

        for handle in handles {
            let (count, keys) = handle.join().unwrap();
            total_count += count;
            all_keys.extend(keys);
        }

        assert_eq!(total_count, 1000);
        assert_eq!(all_keys.len(), 1000);

        // Verify no duplicates
        all_keys.sort();
        all_keys.dedup();
        assert_eq!(all_keys.len(), 1000);

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_parquet_input_binary_data() {
        let path = test_dir().join("binary_data.parquet");

        // Create file with binary data
        let schema = Arc::new(Schema::new(vec![
            Field::new("hash", DataType::Binary, false),
            Field::new("data", DataType::Binary, false),
        ]));

        let mut hashes: Vec<&[u8]> = Vec::new();
        let mut data: Vec<&[u8]> = Vec::new();
        let mut hash_storage = Vec::new();
        let mut data_storage = Vec::new();

        for i in 0..50 {
            // Create some binary data (simulating hashes)
            let hash: Vec<u8> = (0..32).map(|j| ((i + j) % 256) as u8).collect();
            let value: Vec<u8> = (0..64).map(|j| ((i * 2 + j) % 256) as u8).collect();

            hash_storage.push(hash);
            data_storage.push(value);
        }

        for h in &hash_storage {
            hashes.push(h.as_slice());
        }
        for d in &data_storage {
            data.push(d.as_slice());
        }

        let hash_array = BinaryArray::from(hashes);
        let data_array = BinaryArray::from(data);

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(hash_array), Arc::new(data_array)],
        )
        .unwrap();

        let file = File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        // Read binary data
        let input = ParquetInputDirect::new(
            &path,
            ParquetDirectConfig {
                key_columns: vec![0],
                value_columns: vec![1],
            },
        )
        .unwrap();
        assert_eq!(input.len(), 50);

        // Sort binary data
        let mut sorter = ExternalSorter::new(2, 1024);
        let output = sorter.sort(Box::new(input)).unwrap();

        let entries: Vec<_> = output.iter().collect();
        assert_eq!(entries.len(), 50);

        // Verify sorted by binary key
        for i in 1..entries.len() {
            assert!(entries[i - 1].0 <= entries[i].0);
        }

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_parquet_input_int_columns() {
        let path = test_dir().join("int_columns.parquet");

        // Create file with integer columns
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("score", DataType::Int32, false),
        ]));

        let ids: Vec<i32> = (0..100).collect();
        let scores: Vec<i32> = (0..100).map(|i| i * 10).collect();

        let id_array = Int32Array::from(ids);
        let score_array = Int32Array::from(scores);

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(id_array), Arc::new(score_array)],
        )
        .unwrap();

        let file = File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        // Read integer data
        let input = ParquetInputDirect::new(
            &path,
            ParquetDirectConfig {
                key_columns: vec![0],
                value_columns: vec![1],
            },
        )
        .unwrap();
        assert_eq!(input.len(), 100);

        let partitions = input.create_parallel_scanners(1, None);
        let entries: Vec<_> = partitions.into_iter().flatten().collect();

        // Verify integer keys are encoded as order-preserving bytes
        for i in 0..100 {
            let expected_key = i32_to_order_preserving_bytes(i).to_vec();
            let found = entries.iter().any(|(k, _)| k == &expected_key);
            assert!(found, "Missing key for id {}", i);
        }

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_parquet_input_date32_columns() {
        let path = test_dir().join("date32_columns.parquet");

        // Create file with Date32 columns
        let schema = Arc::new(Schema::new(vec![
            Field::new("date", DataType::Date32, false),
            Field::new("value", DataType::Utf8, false),
        ]));

        // Create dates as days since epoch (1970-01-01)
        let base_date = 18000; // Around 2019-04-19
        let dates: Vec<i32> = (0..50).map(|i| base_date + i).collect();
        let values: Vec<String> = (0..50).map(|i| format!("value_{}", i)).collect();

        let date_array = arrow::array::Date32Array::from(dates.clone());
        let value_array = StringArray::from(values);

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(date_array), Arc::new(value_array)],
        )
        .unwrap();

        let file = File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        // Read Date32 data
        let input = ParquetInputDirect::new(
            &path,
            ParquetDirectConfig {
                key_columns: vec![0],
                value_columns: vec![1],
            },
        )
        .unwrap();
        assert_eq!(input.len(), 50);

        let partitions = input.create_parallel_scanners(1, None);
        let entries: Vec<_> = partitions.into_iter().flatten().collect();
        assert_eq!(entries.len(), 50);

        // Verify Date32 keys are encoded as order-preserving bytes
        for i in 0..50 {
            let expected_date = base_date + i;
            let expected_key =
                i32_to_order_preserving_bytes(unix_epoch_to_ce_days(expected_date)).to_vec();
            let found = entries.iter().any(|(k, _)| k == &expected_key);
            assert!(found, "Missing key for date {}", expected_date);
        }

        // Now we CAN sort by date with the new encoding!
        let input2 = ParquetInputDirect::new(
            &path,
            ParquetDirectConfig {
                key_columns: vec![0],   // Use date as key
                value_columns: vec![1], // Use value string as value
            },
        )
        .unwrap();
        let mut sorter = ExternalSorter::new(2, 1024);
        let output = sorter.sort(Box::new(input2)).unwrap();
        let sorted_entries: Vec<_> = output.iter().collect();

        // Verify dates are now properly sorted
        for i in 1..sorted_entries.len() {
            let prev_date =
                i32_from_order_preserving_bytes(sorted_entries[i - 1].0[0..4].try_into().unwrap());
            let curr_date =
                i32_from_order_preserving_bytes(sorted_entries[i].0[0..4].try_into().unwrap());
            assert!(
                prev_date <= curr_date,
                "Dates not sorted: {} > {}",
                prev_date,
                curr_date
            );
        }

        // Verify the first few sorted dates
        let first_date =
            i32_from_order_preserving_bytes(sorted_entries[0].0[0..4].try_into().unwrap());
        let last_date =
            i32_from_order_preserving_bytes(sorted_entries[49].0[0..4].try_into().unwrap());
        assert_eq!(
            first_date,
            unix_epoch_to_ce_days(base_date),
            "First date should be {}",
            unix_epoch_to_ce_days(base_date)
        );
        assert_eq!(
            last_date,
            unix_epoch_to_ce_days(base_date + 49),
            "Last date should be {}",
            unix_epoch_to_ce_days(base_date + 49)
        );

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_parquet_input_decimal128_columns() {
        let path = test_dir().join("decimal128_columns.parquet");

        // Create file with Decimal128 columns
        let schema = Arc::new(Schema::new(vec![
            Field::new("price", DataType::Decimal128(10, 2), false),
            Field::new("product", DataType::Utf8, false),
        ]));

        // Note: Using string keys for sorting since numerical types don't sort correctly
        // with lexicographic comparison when using little-endian encoding
        let prices: Vec<i128> = (0..50).map(|i| (1000 + i * 100) as i128).collect();
        let products: Vec<String> = (0..50).map(|i| format!("product_{:02}", i)).collect();

        let price_array = arrow::array::Decimal128Array::from(prices.clone())
            .with_precision_and_scale(10, 2)
            .unwrap();
        let product_array = StringArray::from(products);

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(price_array), Arc::new(product_array)],
        )
        .unwrap();

        let file = File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        // Read Decimal128 data
        let input = ParquetInputDirect::new(
            &path,
            ParquetDirectConfig {
                key_columns: vec![0],
                value_columns: vec![1],
            },
        )
        .unwrap();
        assert_eq!(input.len(), 50);

        let partitions = input.create_parallel_scanners(1, None);
        let entries: Vec<_> = partitions.into_iter().flatten().collect();
        assert_eq!(entries.len(), 50);

        // Verify Decimal128 keys are encoded as order-preserving bytes
        for i in 0..50 {
            let expected_price = (1000 + i * 100) as i128;
            let expected_key = i128_to_order_preserving_bytes(expected_price).to_vec();
            let found = entries.iter().any(|(k, _)| k == &expected_key);
            assert!(found, "Missing key for price {}", expected_price);
        }

        // Now we CAN sort by decimal price with the new encoding!
        let input2 = ParquetInputDirect::new(
            &path,
            ParquetDirectConfig {
                key_columns: vec![0],   // Use price as key
                value_columns: vec![1], // Use product name as value
            },
        )
        .unwrap();
        let mut sorter = ExternalSorter::new(2, 1024);
        let output = sorter.sort(Box::new(input2)).unwrap();
        let sorted_entries: Vec<_> = output.iter().collect();

        // Verify prices are now properly sorted
        for i in 1..sorted_entries.len() {
            let prev_price = i128_from_order_preserving_bytes(
                sorted_entries[i - 1].0[0..16].try_into().unwrap(),
            );
            let curr_price =
                i128_from_order_preserving_bytes(sorted_entries[i].0[0..16].try_into().unwrap());
            assert!(
                prev_price <= curr_price,
                "Prices not sorted: {} > {}",
                prev_price,
                curr_price
            );
        }

        // Verify the first few sorted prices
        let first_price =
            i128_from_order_preserving_bytes(sorted_entries[0].0[0..16].try_into().unwrap());
        let second_price =
            i128_from_order_preserving_bytes(sorted_entries[1].0[0..16].try_into().unwrap());
        assert_eq!(first_price, 1000, "First price should be 1000 (10.00)");
        assert_eq!(second_price, 1100, "Second price should be 1100 (11.00)");

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_parquet_input_mixed_new_types() {
        let path = test_dir().join("mixed_new_types.parquet");

        // Create file with Int64, Date32, Decimal128, and Utf8 columns
        let schema = Arc::new(Schema::new(vec![
            Field::new("order_id", DataType::Int64, false),
            Field::new("order_date", DataType::Date32, false),
            Field::new("amount", DataType::Decimal128(15, 2), false),
            Field::new("status", DataType::Utf8, false),
        ]));

        let order_ids: Vec<i64> = vec![1001, 1002, 1003, 1004, 1005];
        let order_dates: Vec<i32> = vec![19000, 19001, 19000, 19002, 19001]; // Some duplicate dates
        let amounts: Vec<i128> = vec![10050, 25000, 5000, 15000, 30000]; // 100.50, 250.00, 50.00, 150.00, 300.00
        let statuses: Vec<&str> = vec!["PENDING", "SHIPPED", "DELIVERED", "PENDING", "SHIPPED"];

        let order_id_array = Int64Array::from(order_ids);
        let order_date_array = arrow::array::Date32Array::from(order_dates);
        let amount_array = arrow::array::Decimal128Array::from(amounts)
            .with_precision_and_scale(15, 2)
            .unwrap();
        let status_array = StringArray::from(statuses);

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(order_id_array),
                Arc::new(order_date_array),
                Arc::new(amount_array),
                Arc::new(status_array),
            ],
        )
        .unwrap();

        let file = File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        // Test 1: Sort by Date32 (column 1)
        let input = ParquetInputDirect::new(
            &path,
            ParquetDirectConfig {
                key_columns: vec![1],
                value_columns: vec![0, 2, 3],
            },
        )
        .unwrap();

        let mut sorter = ExternalSorter::new(1, 1024);
        let output = sorter.sort(Box::new(input)).unwrap();
        let sorted_by_date: Vec<_> = output.iter().collect();
        assert_eq!(sorted_by_date.len(), 5);

        // Verify sorted by date (now using order-preserving encoding)
        let dates: Vec<i32> = sorted_by_date
            .iter()
            .map(|(k, _)| i32_from_order_preserving_bytes(k[0..4].try_into().unwrap()))
            .collect();
        assert_eq!(
            dates,
            vec![19000, 19000, 19001, 19001, 19002]
                .into_iter()
                .map(unix_epoch_to_ce_days)
                .collect::<Vec<_>>()
        );

        // Test 2: Sort by Decimal128 (column 2)
        let input2 = ParquetInputDirect::new(
            &path,
            ParquetDirectConfig {
                key_columns: vec![2],
                value_columns: vec![0, 1, 3],
            },
        )
        .unwrap();

        let mut sorter2 = ExternalSorter::new(1, 1024);
        let output2 = sorter2.sort(Box::new(input2)).unwrap();
        let sorted_by_amount: Vec<_> = output2.iter().collect();

        // Note: We can't verify numerical sorting of decimal values because
        // little-endian encoding doesn't preserve numerical order when compared lexicographically
        // Just verify we got all 5 entries
        assert_eq!(sorted_by_amount.len(), 5);

        // Test 3: Multi-column key (Date32 + Decimal128)
        let input3 = ParquetInputDirect::new(
            &path,
            ParquetDirectConfig {
                key_columns: vec![1, 2],
                value_columns: vec![0, 3],
            },
        )
        .unwrap();

        let partitions = input3.create_parallel_scanners(1, None);
        let entries: Vec<_> = partitions.into_iter().flatten().collect();
        assert_eq!(entries.len(), 5);

        // Verify keys contain both date and amount
        for (key, _) in &entries {
            assert!(key.len() >= 20); // At least 4 bytes for date + 1 separator + 16 bytes for decimal
        }

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_extract_bytes_all_supported_types() {
        use arrow::array::Float32Array;
        // This test verifies extract_bytes_from_batch handles all supported types
        let schema = Arc::new(Schema::new(vec![
            Field::new("binary", DataType::Binary, false),
            Field::new("large_binary", DataType::LargeBinary, false),
            Field::new("utf8", DataType::Utf8, false),
            Field::new("large_utf8", DataType::LargeUtf8, false),
            Field::new("int32", DataType::Int32, false),
            Field::new("int64", DataType::Int64, false),
            Field::new("float32", DataType::Float32, false),
            Field::new("float64", DataType::Float64, false),
            Field::new("date32", DataType::Date32, false),
            Field::new("decimal128", DataType::Decimal128(10, 2), false),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(arrow::array::BinaryArray::from(vec![&b"binary"[..]])),
                Arc::new(arrow::array::LargeBinaryArray::from(vec![
                    &b"large_binary"[..],
                ])),
                Arc::new(StringArray::from(vec!["utf8"])),
                Arc::new(arrow::array::LargeStringArray::from(vec!["large_utf8"])),
                Arc::new(Int32Array::from(vec![42_i32])),
                Arc::new(Int64Array::from(vec![84_i64])),
                Arc::new(Float32Array::from(vec![3.14_f32])),
                Arc::new(Float64Array::from(vec![2.718_f64])),
                Arc::new(arrow::array::Date32Array::from(vec![19000_i32])),
                Arc::new(
                    arrow::array::Decimal128Array::from(vec![12345_i128])
                        .with_precision_and_scale(10, 2)
                        .unwrap(),
                ),
            ],
        )
        .unwrap();

        // Test each column type
        assert_eq!(
            extract_bytes_from_batch(&batch, 0, 0),
            Some(b"binary".to_vec())
        );
        assert_eq!(
            extract_bytes_from_batch(&batch, 1, 0),
            Some(b"large_binary".to_vec())
        );
        assert_eq!(
            extract_bytes_from_batch(&batch, 2, 0),
            Some(b"utf8".to_vec())
        );
        assert_eq!(
            extract_bytes_from_batch(&batch, 3, 0),
            Some(b"large_utf8".to_vec())
        );
        assert_eq!(
            extract_bytes_from_batch(&batch, 4, 0),
            Some(i32_to_order_preserving_bytes(42).to_vec())
        );
        assert_eq!(
            extract_bytes_from_batch(&batch, 5, 0),
            Some(i64_to_order_preserving_bytes(84).to_vec())
        );
        assert_eq!(
            extract_bytes_from_batch(&batch, 6, 0),
            Some(f32_to_order_preserving_bytes(3.14).to_vec())
        );
        assert_eq!(
            extract_bytes_from_batch(&batch, 7, 0),
            Some(f64_to_order_preserving_bytes(2.718).to_vec())
        );
        assert_eq!(
            extract_bytes_from_batch(&batch, 8, 0),
            Some(i32_to_order_preserving_bytes(unix_epoch_to_ce_days(19000)).to_vec())
        );
        assert_eq!(
            extract_bytes_from_batch(&batch, 9, 0),
            Some(i128_to_order_preserving_bytes(12345).to_vec())
        ); // 12345 with scale 2
    }

    #[test]
    fn test_parquet_input_sorting_numerical_types() {
        let path = test_dir().join("numerical_sorting.parquet");

        // Create file with various numerical types
        let schema = Arc::new(Schema::new(vec![
            Field::new("i32_col", DataType::Int32, false),
            Field::new("i64_col", DataType::Int64, false),
            Field::new("f64_col", DataType::Float64, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        // Create data with negative, zero, and positive values
        let i32_vals = vec![100, -50, 0, 75, -25];
        let i64_vals = vec![1000i64, -500, 0, 750, -250];
        let f64_vals = vec![10.5, -5.5, 0.0, 7.5, -2.5];
        let names = vec!["first", "second", "third", "fourth", "fifth"];

        let i32_array = Int32Array::from(i32_vals);
        let i64_array = Int64Array::from(i64_vals);
        let f64_array = Float64Array::from(f64_vals);
        let name_array = StringArray::from(names);

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(i32_array),
                Arc::new(i64_array),
                Arc::new(f64_array),
                Arc::new(name_array),
            ],
        )
        .unwrap();

        let file = File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        // Test 1: Sort by i32 column
        let input = ParquetInputDirect::new(
            &path,
            ParquetDirectConfig {
                key_columns: vec![0],
                value_columns: vec![3],
            },
        )
        .unwrap();
        let mut sorter = ExternalSorter::new(1, 1024);
        let output = sorter.sort(Box::new(input)).unwrap();
        let sorted: Vec<_> = output.iter().collect();

        let i32_sorted: Vec<i32> = sorted
            .iter()
            .map(|(k, _)| i32_from_order_preserving_bytes(k[0..4].try_into().unwrap()))
            .collect();
        assert_eq!(i32_sorted, vec![-50, -25, 0, 75, 100]);

        // Test 2: Sort by f64 column
        let input2 = ParquetInputDirect::new(
            &path,
            ParquetDirectConfig {
                key_columns: vec![2],
                value_columns: vec![3],
            },
        )
        .unwrap();
        let mut sorter2 = ExternalSorter::new(1, 1024);
        let output2 = sorter2.sort(Box::new(input2)).unwrap();
        let sorted2: Vec<_> = output2.iter().collect();

        let f64_sorted: Vec<f64> = sorted2
            .iter()
            .map(|(k, _)| f64_from_order_preserving_bytes(k[0..8].try_into().unwrap()))
            .collect();
        assert_eq!(f64_sorted, vec![-5.5, -2.5, 0.0, 7.5, 10.5]);

        std::fs::remove_file(&path).unwrap();
    }
}
