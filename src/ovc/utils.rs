use crate::rand::small_thread_rng;
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};

pub fn random_string(length: usize) -> String {
    let mut rng = small_thread_rng();
    let mut s = String::with_capacity(length);
    for _ in 0..length {
        s.push(rng.sample(rand::distr::Alphanumeric) as char);
    }
    s
}

pub fn random_vec<T>(length: usize) -> Vec<T>
where
    StandardUniform: Distribution<T>,
{
    let mut rng = small_thread_rng();
    let mut vec = Vec::with_capacity(length);
    for _ in 0..length {
        vec.push(rng.random::<T>());
    }
    vec
}

pub fn generate_vecu8_run(num_entries: usize, key_length: usize) -> Vec<Vec<u8>> {
    let mut rng = small_thread_rng();
    let mut runs = Vec::with_capacity(num_entries);
    for _ in 0..num_entries {
        let mut run = Vec::with_capacity(key_length);
        for _ in 0..key_length {
            run.push(rng.random::<u8>());
        }
        runs.push(run);
    }
    runs.sort_unstable();
    runs
}

pub fn generate_vecu32_run(num_entries: usize, key_length: usize) -> Vec<Vec<u32>> {
    let mut rng = small_thread_rng();
    let mut runs = Vec::with_capacity(num_entries);
    for _ in 0..num_entries {
        let mut run = Vec::with_capacity(key_length);
        for _ in 0..key_length {
            run.push(rng.random::<u32>());
        }
        runs.push(run);
    }
    runs.sort_unstable();
    runs
}

pub fn generate_string_run(num_entries: usize, string_length: usize) -> Vec<String> {
    let mut run = Vec::with_capacity(num_entries);
    for _ in 0..num_entries {
        run.push(random_string(string_length));
    }
    run.sort_unstable();
    run
}

pub fn generate_runs<T: Ord>(num_runs: usize, run_length: usize) -> Vec<Vec<T>>
where
    StandardUniform: Distribution<T>,
{
    let mut rng = small_thread_rng();
    let mut runs = Vec::new();
    for _ in 0..num_runs {
        let mut run = Vec::new();
        for _ in 0..run_length {
            run.push(rng.random::<T>());
        }
        run.sort_unstable();
        runs.push(run);
    }
    runs
}

pub fn generate_string_runs(
    num_runs: usize,
    num_entries_per_run: usize,
    string_length: usize,
) -> Vec<Vec<String>> {
    let mut runs = Vec::with_capacity(num_runs);
    for _ in 0..num_runs {
        let run = generate_string_run(num_entries_per_run, string_length);
        runs.push(run);
    }
    runs
}

pub fn generate_vec_runs(
    num_runs: usize,
    num_entries_per_run: usize,
    key_length: usize,
) -> Vec<Vec<Vec<u8>>> {
    let mut runs = Vec::with_capacity(num_runs);
    for _ in 0..num_runs {
        let run = generate_vecu8_run(num_entries_per_run, key_length);
        runs.push(run);
    }
    runs
}

pub fn generate_random_array<T>(run_length: usize) -> Vec<T>
where
    StandardUniform: Distribution<T>,
{
    let mut rng = small_thread_rng();
    let mut run = Vec::with_capacity(run_length);
    for _ in 0..run_length {
        run.push(rng.random::<T>());
    }
    run
}

// Specialized function for Vec<String>
pub fn generate_random_string_array(run_length: usize, string_length: usize) -> Vec<String> {
    let mut run = Vec::with_capacity(run_length);
    for _ in 0..run_length {
        let string = random_string(string_length);
        run.push(string);
    }
    run
}

pub fn insertion_sort<T: PartialEq + PartialOrd>(mut run: Vec<T>) -> Vec<T> {
    for i in 1..run.len() {
        let mut j = i;
        while j > 0 && run[j - 1] > run[j] {
            run.swap(j - 1, j);
            j -= 1;
        }
    }
    run
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_string() {
        let length = 10;
        let s = random_string(length);
        assert_eq!(s.len(), length);
    }

    #[test]
    fn test_insertion_sort() {
        let mut run = generate_random_array::<i32>(100);
        let mut expected = run.clone();
        run = insertion_sort(run);
        expected.sort();
        assert_eq!(run, expected);
    }
}
