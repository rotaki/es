//! RW-lock fairness arbitration for the NVMe during run formation (Myung §3.3).
//!
//! Semantics:
//!   - Many readers can hold the lock concurrently (preserves NVMe read
//!     parallelism during the read step of all threads).
//!   - A single writer at a time (serializes write bursts to collapse the
//!     write-side starvation variance the SSD controller introduces).
//!   - Readers and writers are mutually exclusive.
//!   - A FIFO semaphore preserves writer order so no thread falls behind
//!     indefinitely.
//!
//! The arbitration is toggleable so we can A/B it on our hardware and
//! reproduce Myung's 34% makespan measurement (or whatever value the Intel
//! 760p exhibits).

use std::sync::{Arc, Condvar, Mutex};

/// Shared handle passed to each worker thread.
#[derive(Clone)]
pub struct DeviceArbiter {
    inner: Arc<Inner>,
}

struct Inner {
    enabled: bool,
    state: Mutex<State>,
    cv: Condvar,
}

#[derive(Debug)]
struct State {
    readers: usize,
    writer_active: bool,
    /// Next ticket to assign to an arriving writer.
    next_writer_ticket: u64,
    /// Ticket of the writer at the head of the FIFO (one that may proceed when
    /// readers drain and no writer is active).
    next_writer_to_serve: u64,
}

impl DeviceArbiter {
    pub fn new(enabled: bool) -> Self {
        DeviceArbiter {
            inner: Arc::new(Inner {
                enabled,
                state: Mutex::new(State {
                    readers: 0,
                    writer_active: false,
                    next_writer_ticket: 0,
                    next_writer_to_serve: 0,
                }),
                cv: Condvar::new(),
            }),
        }
    }

    pub fn enabled(&self) -> bool {
        self.inner.enabled
    }

    /// Acquire the shared (read) lock. Returned guard releases on drop.
    pub fn read(&self) -> ReadGuard<'_> {
        if !self.inner.enabled {
            return ReadGuard { arb: None };
        }
        let mut st = self.inner.state.lock().unwrap();
        while st.writer_active {
            st = self.inner.cv.wait(st).unwrap();
        }
        st.readers += 1;
        ReadGuard { arb: Some(self) }
    }

    /// Acquire the exclusive (write) lock with FIFO ordering.
    pub fn write(&self) -> WriteGuard<'_> {
        if !self.inner.enabled {
            return WriteGuard { arb: None };
        }
        let ticket;
        {
            let mut st = self.inner.state.lock().unwrap();
            ticket = st.next_writer_ticket;
            st.next_writer_ticket += 1;
            while st.writer_active || st.readers > 0 || st.next_writer_to_serve != ticket {
                st = self.inner.cv.wait(st).unwrap();
            }
            st.writer_active = true;
        }
        WriteGuard { arb: Some(self) }
    }
}

pub struct ReadGuard<'a> {
    arb: Option<&'a DeviceArbiter>,
}

impl Drop for ReadGuard<'_> {
    fn drop(&mut self) {
        if let Some(arb) = self.arb {
            let mut st = arb.inner.state.lock().unwrap();
            st.readers -= 1;
            if st.readers == 0 {
                arb.inner.cv.notify_all();
            }
        }
    }
}

pub struct WriteGuard<'a> {
    arb: Option<&'a DeviceArbiter>,
}

impl Drop for WriteGuard<'_> {
    fn drop(&mut self) {
        if let Some(arb) = self.arb {
            let mut st = arb.inner.state.lock().unwrap();
            st.writer_active = false;
            st.next_writer_to_serve += 1;
            arb.inner.cv.notify_all();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;
    use std::time::Duration;

    #[test]
    fn disabled_is_no_op() {
        let a = DeviceArbiter::new(false);
        // Acquiring should not block.
        let _r = a.read();
        let _w = a.write();
    }

    #[test]
    fn writer_exclusive_with_readers() {
        let a = DeviceArbiter::new(true);
        let concurrent_writers = Arc::new(AtomicUsize::new(0));
        let max_concurrent = Arc::new(AtomicUsize::new(0));
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let a = a.clone();
                let cw = concurrent_writers.clone();
                let mx = max_concurrent.clone();
                thread::spawn(move || {
                    for _ in 0..10 {
                        if i % 2 == 0 {
                            let _g = a.read();
                            thread::sleep(Duration::from_millis(1));
                        } else {
                            let _g = a.write();
                            let now = cw.fetch_add(1, Ordering::SeqCst) + 1;
                            mx.fetch_max(now, Ordering::SeqCst);
                            thread::sleep(Duration::from_millis(1));
                            cw.fetch_sub(1, Ordering::SeqCst);
                        }
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(max_concurrent.load(Ordering::SeqCst), 1);
    }
}
