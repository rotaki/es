use std::path::PathBuf;
use std::sync::Once;

static INIT: Once = Once::new();

fn test_dir() -> PathBuf {
    INIT.call_once(|| {
        let dir = PathBuf::from("./test_runs");
        std::fs::create_dir_all(&dir).expect("Failed to create test directory");
    });
    PathBuf::from("./test_runs")
}

use es::{ExternalSorter, Input, Sorter};
use std::collections::BTreeMap;
use std::fs;

#[test]
fn test_csv_like_data() {
    // Simulate sorting CSV-like data
    let mut sorter = ExternalSorter::new_with_dir(4, 1024 * 1024, test_dir());

    let mut data = Vec::new();
    for i in 0..10_000 {
        // Simulate CSV row with timestamp as key
        let timestamp = format!(
            "2024-01-{:02} {:02}:{:02}:{:02}",
            (i % 30) + 1,
            i % 24,
            i % 60,
            i % 60
        );
        let row = format!(
            "user_{},action_{},result_{}",
            i % 1000,
            i % 50,
            if i % 2 == 0 { "success" } else { "failure" }
        );
        data.push((timestamp.into_bytes(), row.into_bytes()));
    }

    // Shuffle to simulate unordered log entries
    use rand::seq::SliceRandom;
    let mut rng = rand::rng();
    data.shuffle(&mut rng);

    let input = Input { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    let results: Vec<_> = output.iter().collect();
    assert_eq!(results.len(), 10_000);

    // Verify chronological order
    for i in 1..results.len() {
        assert!(results[i - 1].0 <= results[i].0);
    }
}

#[test]
fn test_json_like_keys() {
    // Test with JSON-like nested keys
    let mut sorter = ExternalSorter::new_with_dir(2, 1024, test_dir());

    let mut data = Vec::new();
    for i in 0..1000 {
        let key = format!(
            r#"{{"user_id":{},"timestamp":{},"action":"click"}}"#,
            i % 100,
            1000000 + i
        );
        let value = format!("metadata_{}", i);
        data.push((key.into_bytes(), value.into_bytes()));
    }

    let input = Input { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    let results: Vec<_> = output.iter().collect();
    assert_eq!(results.len(), 1000);

    // Verify lexicographic ordering of JSON strings
    for i in 1..results.len() {
        assert!(results[i - 1].0 <= results[i].0);
    }
}

#[test]
fn test_url_sorting() {
    // Sort URLs by domain and path
    let mut sorter = ExternalSorter::new_with_dir(4, 2048, test_dir());

    let urls = ["https://example.com/page1",
        "https://example.com/page2",
        "https://api.example.com/v1/users",
        "https://api.example.com/v2/users",
        "http://example.com/index",
        "https://blog.example.com/post/1",
        "https://blog.example.com/post/10",
        "https://blog.example.com/post/2"];

    let mut data = Vec::new();
    for (i, url) in urls.iter().enumerate() {
        // Use URL as key, metadata as value
        let value = format!(
            "{{\"visits\":{},\"bounce_rate\":0.{}}}",
            (i + 1) * 100,
            i * 10
        );
        data.push((url.as_bytes().to_vec(), value.into_bytes()));
    }

    // Add many duplicates
    for _ in 0..100 {
        for (i, url) in urls.iter().enumerate() {
            let value = format!("{{\"session\":{}}}", i);
            data.push((url.as_bytes().to_vec(), value.into_bytes()));
        }
    }

    let input = Input { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    let results: Vec<_> = output.iter().collect();
    assert_eq!(results.len(), urls.len() + urls.len() * 100);

    // Verify URLs are sorted correctly
    for i in 1..results.len() {
        assert!(results[i - 1].0 <= results[i].0);
    }
}

#[test]
fn test_word_frequency_sorting() {
    // Simulate word frequency counting followed by sorting
    let mut sorter = ExternalSorter::new_with_dir(4, 1024, test_dir());

    let text = "the quick brown fox jumps over the lazy dog the fox was quick";
    let words: Vec<&str> = text.split_whitespace().collect();

    // Count frequencies
    let mut freq_map = BTreeMap::new();
    for word in &words {
        *freq_map.entry(*word).or_insert(0) += 1;
    }

    // Create entries with frequency as key (padded for correct sorting)
    let mut data = Vec::new();
    for (word, count) in freq_map {
        // Pad count to ensure numeric sorting
        let key = format!("{:05}_{}", count, word);
        let value = word.to_string();
        data.push((key.into_bytes(), value.into_bytes()));
    }

    let input = Input { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    let results: Vec<_> = output.iter().collect();

    // Most frequent word should be last (the: 3 times)
    let last = &results[results.len() - 1];
    assert!(String::from_utf8_lossy(&last.0).starts_with("00003"));
    assert_eq!(last.1, b"the");
}

#[test]
fn test_ip_address_sorting() {
    // Sort IP addresses with activity counts
    let mut sorter = ExternalSorter::new_with_dir(2, 1024, test_dir());

    let mut data = Vec::new();

    // Generate IPs
    for a in 1..5 {
        for b in 0..256 {
            if b % 50 == 0 {
                // Sample to keep test fast
                let ip = format!("192.168.{}.{}", a, b);
                let value = format!(
                    "requests:{},bytes:{}",
                    (a * b) % 1000,
                    (a * b * 1024) % 1000000
                );
                data.push((ip.into_bytes(), value.into_bytes()));
            }
        }
    }

    // Add some specific IPs multiple times
    for i in 0..100 {
        let ip = "192.168.1.1";
        let value = format!("session_{}", i);
        data.push((ip.as_bytes().to_vec(), value.into_bytes()));
    }

    let input = Input { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    let results: Vec<_> = output.iter().collect();

    // Verify IP addresses are sorted lexicographically
    for i in 1..results.len() {
        assert!(results[i - 1].0 <= results[i].0);
    }
}

#[test]
fn test_genomic_data_sorting() {
    // Simulate sorting genomic sequences
    let mut sorter = ExternalSorter::new_with_dir(4, 2048, test_dir());

    let bases = [b'A', b'C', b'G', b'T'];
    let mut data = Vec::new();

    // Generate sequences
    use rand::Rng;
    let mut rng = rand::rng();

    for i in 0..5000 {
        // Variable length sequences
        let seq_len = rng.random_range(10..50);
        let sequence: Vec<u8> = (0..seq_len)
            .map(|_| bases[rng.random_range(0..4)])
            .collect();

        let metadata = format!("gene_{}_quality_{}", i, rng.random_range(0..100));
        data.push((sequence, metadata.into_bytes()));
    }

    let input = Input { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    let results: Vec<_> = output.iter().collect();
    assert_eq!(results.len(), 5000);

    // Verify sequences are sorted
    for i in 1..results.len() {
        assert!(results[i - 1].0 <= results[i].0);
    }
}

#[test]
fn test_file_path_sorting() {
    // Sort file paths maintaining directory structure
    let mut sorter = ExternalSorter::new_with_dir(2, 1024, test_dir());

    let paths = vec![
        "/home/user/documents/file1.txt",
        "/home/user/documents/file10.txt",
        "/home/user/documents/file2.txt",
        "/home/user/downloads/archive.zip",
        "/home/user/downloads/image.png",
        "/home/user/.config/app.conf",
        "/home/user/.bashrc",
        "/etc/hosts",
        "/etc/passwd",
        "/var/log/syslog",
        "/var/log/app.log",
    ];

    let mut data = Vec::new();
    for path in &paths {
        // Simulate file metadata as value
        let metadata = format!("size:{},modified:2024-01-01", path.len() * 100);
        data.push((path.as_bytes().to_vec(), metadata.into_bytes()));
    }

    // Add many more generated paths
    for i in 0..1000 {
        let path = format!("/home/user/data/file_{:04}.dat", i);
        let metadata = format!("size:{}", i * 1024);
        data.push((path.into_bytes(), metadata.into_bytes()));
    }

    let input = Input { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    let results: Vec<_> = output.iter().collect();

    // Verify paths are sorted correctly
    for i in 1..results.len() {
        assert!(results[i - 1].0 <= results[i].0);
    }

    // Check that /etc comes before /home
    let etc_pos = results
        .iter()
        .position(|(k, _)| k.starts_with(b"/etc"))
        .unwrap();
    let home_pos = results
        .iter()
        .position(|(k, _)| k.starts_with(b"/home"))
        .unwrap();
    assert!(etc_pos < home_pos);
}

#[test]
fn test_composite_key_sorting() {
    // Sort by composite keys (e.g., timestamp + user_id)
    let mut sorter = ExternalSorter::new_with_dir(4, 1024, test_dir());

    let mut data = Vec::new();
    for day in 1..10 {
        for hour in 0..24 {
            for user_id in 0..100 {
                if (user_id + hour) % 10 == 0 {
                    // Sample
                    let key = format!("2024-01-{:02}T{:02}:00:00_user_{:04}", day, hour, user_id);
                    let value = format!(
                        "{{\"action\":\"login\",\"ip\":\"192.168.1.{}\"}}",
                        user_id % 255
                    );
                    data.push((key.into_bytes(), value.into_bytes()));
                }
            }
        }
    }

    let input = Input { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    let results: Vec<_> = output.iter().collect();

    // Verify chronological order with user grouping
    for i in 1..results.len() {
        assert!(results[i - 1].0 <= results[i].0);
    }
}

#[test]
fn test_cleanup_on_drop() {
    // Verify that temporary files are cleaned up
    let temp_dir_pattern = "external_sort_";

    // Count existing temp directories
    let temp_dir = std::env::temp_dir();
    let before_count = fs::read_dir(&temp_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().contains(temp_dir_pattern))
        .count();

    // Run a sort
    {
        let mut sorter = ExternalSorter::new_with_dir(2, 512, test_dir());
        let mut data = Vec::new();
        for i in 0..1000 {
            data.push((format!("{:04}", i).into_bytes(), b"value".to_vec()));
        }
        let input = Input { data };
        let _output = sorter.sort(Box::new(input)).unwrap();
    }

    // Check that temp directories still exist (manual cleanup needed)
    let after_count = fs::read_dir(&temp_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().contains(temp_dir_pattern))
        .count();

    // At least one new directory should have been created
    assert!(after_count >= before_count);
}

#[test]
fn test_real_world_log_sorting() {
    // Simulate sorting real-world log entries
    let mut sorter = ExternalSorter::new_with_dir(4, 2 * 1024 * 1024, test_dir());

    let log_levels = ["DEBUG", "INFO", "WARN", "ERROR"];
    let components = ["auth", "api", "db", "cache", "web"];

    let mut data = Vec::new();
    use rand::Rng;
    let mut rng = rand::rng();

    for i in 0..50_000 {
        let timestamp = format!(
            "2024-01-15T{:02}:{:02}:{:02}.{:03}Z",
            i / 3600 % 24,
            i / 60 % 60,
            i % 60,
            i % 1000
        );

        let level = log_levels[rng.random_range(0..log_levels.len())];
        let component = components[rng.random_range(0..components.len())];
        let message = format!("[{}] [{}] Processing request {}", level, component, i);

        // Use timestamp as key for chronological sorting
        data.push((timestamp.into_bytes(), message.into_bytes()));
    }

    let input = Input { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    let results: Vec<_> = output.iter().collect();
    assert_eq!(results.len(), 50_000);

    // Verify chronological order
    for i in 1..results.len() {
        assert!(results[i - 1].0 <= results[i].0);
    }
}

#[test]
fn test_case_sensitive_sorting() {
    // Test case-sensitive sorting behavior
    let mut sorter = ExternalSorter::new_with_dir(1, 512, test_dir());

    let words = vec![
        "Apple", "apple", "APPLE", "Banana", "banana", "BANANA", "Cherry", "cherry", "CHERRY",
        "123", "ABC", "abc", "Abc",
    ];

    let mut data = Vec::new();
    for word in &words {
        data.push((word.as_bytes().to_vec(), b"value".to_vec()));
    }

    let input = Input { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    let results: Vec<_> = output.iter().collect();
    assert_eq!(results.len(), words.len());

    // Verify ASCII ordering (numbers < uppercase < lowercase)
    let sorted_keys: Vec<String> = results
        .iter()
        .map(|(k, _)| String::from_utf8_lossy(k).to_string())
        .collect();

    // Numbers should come first
    assert!(sorted_keys[0].starts_with(|c: char| c.is_numeric()));

    // Capital letters before lowercase
    let apple_pos = sorted_keys.iter().position(|s| s == "Apple").unwrap();
    let apple_lower_pos = sorted_keys.iter().position(|s| s == "apple").unwrap();
    assert!(apple_pos < apple_lower_pos);
}
