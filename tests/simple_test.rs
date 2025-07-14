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

#[test]
fn test_basic_sort() {
    let mut sorter = ExternalSorter::new_with_dir(2, 1024, test_dir());

    let data = vec![
        (b"c".to_vec(), b"3".to_vec()),
        (b"a".to_vec(), b"1".to_vec()),
        (b"b".to_vec(), b"2".to_vec()),
    ];

    let input = Input { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    let results: Vec<_> = output.iter().collect();
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].0, b"a");
    assert_eq!(results[1].0, b"b");
    assert_eq!(results[2].0, b"c");
}

#[test]
fn test_external_sort() {
    // Use small buffer to force external sorting
    let mut sorter = ExternalSorter::new_with_dir(2, 512, test_dir());

    let mut data = Vec::new();
    for i in (0..100).rev() {
        let key = format!("key_{:03}", i);
        let value = format!("value_{}", i);
        data.push((key.into_bytes(), value.into_bytes()));
    }

    let input = Input { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    // Verify sorted
    let results: Vec<_> = output.iter().collect();
    assert_eq!(results.len(), 100);
    for i in 0..100 {
        let expected_key = format!("key_{:03}", i);
        assert_eq!(results[i].0, expected_key.as_bytes());
    }
}

#[test]
fn test_empty_input() {
    let mut sorter = ExternalSorter::new_with_dir(2, 1024, test_dir());

    let input = Input { data: vec![] };
    let output = sorter.sort(Box::new(input)).unwrap();

    let results: Vec<_> = output.iter().collect();
    assert_eq!(results.len(), 0);
}
