use es::{ExternalSorter, Input, Sorter};

fn main() {
    println!("=== External Sort Demo ===\n");

    // Create sorter with minimal configuration
    let num_threads = 4;
    let buffer_size = 1024 * 1024; // 1MB buffer
    let mut sorter = ExternalSorter::new(num_threads, buffer_size);

    // Generate test data
    println!("Generating test data...");
    let mut data = Vec::new();

    // Generate entries that need sorting
    let entries = vec![
        ("user:005", "Alice"),
        ("user:003", "Bob"),
        ("user:001", "Charlie"),
        ("user:004", "David"),
        ("user:002", "Eve"),
        ("timestamp:2024-01-03", "Event C"),
        ("timestamp:2024-01-01", "Event A"),
        ("timestamp:2024-01-02", "Event B"),
        ("session:zzz", "Last session"),
        ("session:aaa", "First session"),
        ("session:mmm", "Middle session"),
    ];

    for (key, value) in entries {
        data.push((key.as_bytes().to_vec(), value.as_bytes().to_vec()));
    }

    // Add more data to trigger external sorting
    for i in 0..1000 {
        let key = format!("data:{:04}", (i * 7) % 1000);
        let value = format!("Value {}", i);
        data.push((key.into_bytes(), value.into_bytes()));
    }

    println!("Generated {} entries", data.len());

    // Sort
    println!("\nSorting...");
    let input = Input { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    // Show results
    println!("\nFirst 20 sorted entries:");
    let results: Vec<_> = output.iter().take(20).collect();
    for (i, (key, value)) in results.iter().enumerate() {
        println!(
            "{:2}: {} => {}",
            i + 1,
            String::from_utf8_lossy(key),
            String::from_utf8_lossy(value)
        );
    }

    println!("\nSort complete!");
}
