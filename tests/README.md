# External Sorter Test Suite

This directory contains comprehensive tests for the external sort buffer implementation.

## Test Files

### 1. `external_sorter_tests.rs` - Core Functionality Tests
- **Basic sorting**: Simple 3-element sort
- **Large datasets**: 100K entries with single thread
- **Small buffers**: Forces external sorting with multiple runs
- **Multi-threading**: Tests with 1, 2, 4, and 8 threads
- **Duplicate keys**: Extensive duplicate key handling with stability verification
- **Variable sizes**: Different key and value sizes
- **Binary data**: Tests with null bytes and binary content
- **Unicode**: Tests with international characters and emojis
- **Edge cases**: Empty keys/values, exact buffer boundaries
- **Error handling**: Entry too large for buffer
- **Concurrency**: Multiple sorters running simultaneously
- **Performance**: Basic performance scaling tests
- **Merge efficiency**: Tests with many runs (2500+)

### 2. `external_sorter_stress_tests.rs` - Stress and Edge Case Tests
- **Very large dataset**: 1M entries (ignored by default, run with `--ignored`)
- **Pathological distributions**: Same prefixes, all identical keys, reverse order
- **Memory pressure**: Large values (10KB-1MB each)
- **Extreme thread counts**: Up to 64 threads
- **Buffer size variations**: From 64 bytes to 8KB
- **Concurrent massive sorts**: Multiple large sorts simultaneously
- **Alternating sizes**: Mix of tiny and large entries
- **Random binary keys**: Completely random key/value data
- **Progressive key lengths**: Keys from 1 to 100 characters
- **File handle stress**: 20 concurrent sorts
- **Maximum value sizes**: 1MB values

### 3. `external_sorter_integration_tests.rs` - Real-World Scenarios
- **CSV data**: Timestamp-based sorting of log-like data
- **JSON keys**: Sorting JSON documents by serialized form
- **URL sorting**: Domain and path-based sorting with duplicates
- **Word frequency**: Simulating word count sorting
- **IP addresses**: Network log sorting scenarios
- **Genomic data**: DNA sequence sorting
- **File paths**: Directory structure aware sorting
- **Composite keys**: Multi-field sorting (timestamp + user_id)
- **Log entries**: 50K realistic log entries with timestamps
- **Case sensitivity**: ASCII ordering verification

## Running Tests

```bash
# Run all tests
cargo test

# Run specific test file
cargo test --test external_sorter_tests

# Run specific test
cargo test --test external_sorter_tests test_duplicate_keys

# Run stress tests including ignored ones
cargo test --test external_sorter_stress_tests -- --ignored

# Run with output
cargo test --test external_sorter_tests -- --nocapture
```

## Test Coverage

The test suite covers:

1. **Correctness**
   - Proper sorting order for various data types
   - Preservation of all entries
   - Stability for equal keys
   - Unicode and binary data handling

2. **Performance**
   - Scaling with data size
   - Multi-threading efficiency
   - Memory usage patterns
   - Many runs merging

3. **Edge Cases**
   - Empty input
   - Single entry
   - Entries larger than buffer (causes panic)
   - Extreme key/value sizes
   - Buffer boundary conditions

4. **Concurrency**
   - Thread safety
   - Multiple concurrent sorts
   - Various thread counts

5. **Real-World Usage**
   - Log file sorting
   - CSV/TSV data
   - JSON documents
   - Network logs
   - File system paths

## Known Limitations

1. The current implementation panics when a single entry exceeds the buffer size
2. Zero threads causes division by zero (minimum 1 thread required)
3. Temporary files are not automatically cleaned up after sorting

## Performance Notes

- Small buffers (< 1KB) create many runs and slow down merging
- Optimal buffer size depends on data characteristics
- More threads help with run generation but not merging (currently single-threaded)
- Direct I/O requires page-aligned operations which may impact small sorts