use std::path::PathBuf;
use std::sync::Once;

static INIT: Once = Once::new();

pub fn test_dir() -> PathBuf {
    INIT.call_once(|| {
        let dir = PathBuf::from("./test_runs");
        std::fs::create_dir_all(&dir).expect("Failed to create test directory");
    });
    PathBuf::from("./test_runs")
}
