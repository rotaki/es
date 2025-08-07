use std::{cmp::Ordering, usize};

use crate::ovc::{
    entry::SentinelValue, offset_value_coding::OVCTrait, offset_value_coding_u64::OVCU64,
};

#[derive(Clone, PartialEq, Eq)]
// Do not implement Ord or PartialOrd for OVCKeyValuePair
// Use `compare_and_update` method for ordering
pub struct OVCKeyValuePair {
    ovc: OVCU64,
    key: Vec<u8>,
    value: Vec<u8>,
}

impl std::fmt::Debug for OVCKeyValuePair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_early_fence() {
            write!(f, "[EarlyFence]")
        } else if self.is_late_fence() {
            write!(f, "[LateFence]")
        } else {
            write!(f, "{} -> {:?}", self.ovc, self.key)
        }
    }
}

impl std::fmt::Display for OVCKeyValuePair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_early_fence() {
            write!(f, "[EarlyFence]")
        } else if self.is_late_fence() {
            write!(f, "[LateFence]")
        } else {
            write!(f, "{} -> {:?}", self.ovc, self.key)
        }
    }
}

impl From<(Vec<u8>, Vec<u8>)> for OVCKeyValuePair {
    fn from((key, value): (Vec<u8>, Vec<u8>)) -> Self {
        let ovc = OVCU64::initial_value();
        Self { ovc, key, value }
    }
}

impl From<(OVCU64, Vec<u8>, Vec<u8>)> for OVCKeyValuePair {
    fn from((ovc, key, value): (OVCU64, Vec<u8>, Vec<u8>)) -> Self {
        Self { ovc, key, value }
    }
}

impl OVCKeyValuePair {
    pub fn new(key: Vec<u8>, value: Vec<u8>) -> Self {
        let ovc = OVCU64::initial_value();
        Self { ovc, key, value }
    }

    pub fn new_with_ovc(ovc: OVCU64, key: Vec<u8>, value: Vec<u8>) -> Self {
        Self { ovc, key, value }
    }

    pub fn get_ovc(&self) -> OVCU64 {
        self.ovc
    }

    pub fn get_key(&self) -> &[u8] {
        &self.key
    }

    pub fn get_value(&self) -> &[u8] {
        &self.value
    }

    pub fn take(self) -> (OVCU64, Vec<u8>, Vec<u8>) {
        (self.ovc, self.key, self.value)
    }
}

impl OVCTrait for OVCKeyValuePair {
    fn update(&mut self, prev: &Self) {
        debug_assert!(self.key >= prev.key);

        const CHUNK_SIZE: usize = 6;

        let min_len = self.key.len().min(prev.key.len());

        // Hybrid approach: byte-by-byte comparison with chunk-aligned OVC creation
        for i in 0..min_len {
            match self.key[i].cmp(&prev.key[i]) {
                Ordering::Less => {
                    panic!("This val should be greater than or equal to prev val");
                }
                Ordering::Greater => {
                    // Found difference at byte i, align OVC to chunk boundary
                    let aligned_offset = (i / CHUNK_SIZE) * CHUNK_SIZE;
                    let chunk_end = (aligned_offset + CHUNK_SIZE).min(self.key.len());
                    self.ovc =
                        OVCU64::normal_value(&self.key[aligned_offset..chunk_end], aligned_offset);
                    return;
                }
                Ordering::Equal => continue,
            }
        }

        // If we reach here, prev is a prefix of self or they're equal
        match self.key.len().cmp(&prev.key.len()) {
            Ordering::Greater => {
                // If self is longer, align the offset to chunk boundary
                let aligned_offset = (prev.key.len() / CHUNK_SIZE) * CHUNK_SIZE;
                let chunk_end = (aligned_offset + CHUNK_SIZE).min(self.key.len());
                self.ovc =
                    OVCU64::normal_value(&self.key[aligned_offset..chunk_end], aligned_offset);
            }
            Ordering::Equal => {
                self.ovc = OVCU64::duplicate_value();
            }
            Ordering::Less => {
                panic!(
                    "This val should be longer than or equal to prev val if prefixes are common"
                );
            }
        }
    }

    fn compare_and_update(&mut self, other: &mut Self) -> Ordering {
        let ovc_a = &mut self.ovc;
        let ovc_b = &mut other.ovc;

        match ovc_a.cmp(&ovc_b) {
            Ordering::Equal => {
                if ovc_a.is_early_fence() || ovc_a.is_late_fence() || ovc_a.is_duplicate_value() {
                    return Ordering::Equal;
                }

                const CHUNK_SIZE: usize = 6;

                let offset = ovc_a.offset();
                let start_index = (!ovc_a.is_initial_value() as usize) * (offset + CHUNK_SIZE);

                let val_a = &self.key;
                let val_b = &other.key;
                let min_len = val_a.len().min(val_b.len());

                for i in start_index..min_len {
                    match val_a[i].cmp(&val_b[i]) {
                        Ordering::Equal => continue,
                        Ordering::Less => {
                            // Found difference at byte i, align OVC to chunk boundary
                            let aligned_offset = (i / CHUNK_SIZE) * CHUNK_SIZE;
                            let chunk_end = (aligned_offset + CHUNK_SIZE).min(val_b.len());
                            *ovc_b = OVCU64::normal_value(
                                &val_b[aligned_offset..chunk_end],
                                aligned_offset,
                            );
                            return Ordering::Less;
                        }
                        Ordering::Greater => {
                            // Found difference at byte i, align OVC to chunk boundary
                            let aligned_offset = (i / CHUNK_SIZE) * CHUNK_SIZE;
                            let chunk_end = (aligned_offset + CHUNK_SIZE).min(val_a.len());
                            *ovc_a = OVCU64::normal_value(
                                &val_a[aligned_offset..chunk_end],
                                aligned_offset,
                            );
                            return Ordering::Greater;
                        }
                    }
                }

                // If we reach here, the prefixes are equal
                match val_a.len().cmp(&val_b.len()) {
                    Ordering::Greater => {
                        // This is longer than the other
                        // Align to chunk boundary where the difference starts
                        let aligned_offset = (val_b.len() / CHUNK_SIZE) * CHUNK_SIZE;
                        let chunk_end = (aligned_offset + CHUNK_SIZE).min(val_a.len());
                        *ovc_a =
                            OVCU64::normal_value(&val_a[aligned_offset..chunk_end], aligned_offset);
                        Ordering::Greater
                    }
                    Ordering::Less => {
                        // The other is longer than this
                        // Align to chunk boundary where the difference starts
                        let aligned_offset = (val_a.len() / CHUNK_SIZE) * CHUNK_SIZE;
                        let chunk_end = (aligned_offset + CHUNK_SIZE).min(val_b.len());
                        *ovc_b =
                            OVCU64::normal_value(&val_b[aligned_offset..chunk_end], aligned_offset);
                        Ordering::Less
                    }
                    Ordering::Equal => {
                        *ovc_a = OVCU64::duplicate_value();
                        Ordering::Equal
                    }
                }
            }
            ord => ord,
        }
    }
}

impl SentinelValue for OVCKeyValuePair {
    fn early_fence() -> Self {
        Self {
            ovc: OVCU64::early_fence(),
            key: Vec::new(),
            value: Vec::new(),
        }
    }

    fn late_fence() -> Self {
        Self {
            ovc: OVCU64::late_fence(),
            key: Vec::new(),
            value: Vec::new(),
        }
    }

    fn is_early_fence(&self) -> bool {
        self.ovc.is_early_fence()
    }

    fn is_late_fence(&self) -> bool {
        self.ovc.is_late_fence()
    }
}
