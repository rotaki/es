use es::{ExternalSorter, Input, Sorter};

fn main() {
    println!("=== External Sort ===\n");

    // Configuration
    let num_threads = 4;
    let buffer_size = 1024 * 1024; // 1MB buffer

    // Create sorter
    let mut sorter = ExternalSorter::new(num_threads, buffer_size);

    // Create test data
    let mut data = Vec::new();

    // Add some sample data
    data.push((b"user:003".to_vec(), b"Alice".to_vec()));
    data.push((b"user:001".to_vec(), b"Bob".to_vec()));
    data.push((b"user:002".to_vec(), b"Charlie".to_vec()));

    // Add more data to demonstrate external sorting
    for i in 0..1000 {
        let key = format!("item:{:04}", (i * 7) % 1000);
        let value = format!("Value {}", i);
        data.push((key.into_bytes(), value.into_bytes()));
    }

    println!("Sorting {} entries...\n", data.len());

    // Sort
    let input = Input { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    // Display first few results
    println!("First 10 sorted entries:");
    let results: Vec<_> = output.iter().take(10).collect();
    for (i, (key, value)) in results.iter().enumerate() {
        println!(
            "{:2}: {} => {}",
            i + 1,
            String::from_utf8_lossy(key),
            String::from_utf8_lossy(value)
        );
    }
}
