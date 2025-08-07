use std::{cmp::Ordering, usize};

use crate::ovc::{
    entry::SentinelValue,
    offset_value_coding::{OVCFlag, OVCTrait},
};

pub fn encode_run_with_ovc64(normalized_run: &[Vec<u8>]) -> Vec<OVCEntry64> {
    let mut result = Vec::with_capacity(normalized_run.len());

    for i in 0..normalized_run.len() {
        let mut new_entry = OVCEntry64::new(normalized_run[i].clone());

        if i > 0 {
            new_entry.update(&result[i - 1]);
        } else {
            // The first entry is always a initial value
            new_entry.ovc = OVCU64::initial_value();
        }

        result.push(new_entry);
    }

    result
}

pub fn encode_runs_with_ovc64(normalized_runs: &[Vec<Vec<u8>>]) -> Vec<Vec<OVCEntry64>> {
    normalized_runs
        .iter()
        .map(|run| encode_run_with_ovc64(run))
        .collect::<Vec<_>>()
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Ord, PartialOrd)]
pub struct OVCU64(u64);

impl OVCU64 {
    // Memory layout (64 bits total):
    // ┌───────┬─────────────────────┬────────────────────────────────────────────────┐
    // │ flags │ arity_minus_offset  │                     value                      │
    // └───────┴─────────────────────┴────────────────────────────────────────────────┘
    //  bits 61-63      bits 48-60                      bits 0-47
    //  (3 bits)        (13 bits)                       (48 bits)
    //
    // Fields:
    // • value (bits 0-47): The actual value stored at this offset (6 bytes)
    //
    // • arity_minus_offset (bits 48-60): Encodes the distance from a reference point
    //   Calculated as: 0x1FFF - (actual_index_difference)
    //   This inverted encoding may be used for efficient comparisons
    //   Max value: 8191 (0x1FFF)
    //
    // • flags (bits 61-63): Identifies the type of this entry
    //   - 0: EARLY_FENCE     (boundary marker before valid range)
    //   - 1: DUPLICATE_VALUE (repeated/duplicate entry)
    //   - 2: NORMAL_VALUE    (standard entry)
    //   - 3: INITIAL_VALUE   (first/default value)
    //   - 4: LATE_FENCE      (boundary marker after valid range)

    const VALUE_MASK: u64 = 0x0000_FFFF_FFFF_FFFF; // 48 bits
    const ARITY_MASK: u64 = 0x1FFF; // 13 bits
    const ARITY_MAX: u16 = 0x1FFF; // 8191
    const ARITY_SHIFT: u32 = 48; // shift for arity_minus_offset (bits 48-60)
    const FLAG_MASK: u64 = 0x7; // 3 bits
    const FLAG_SHIFT: u32 = 61; // shift for flags (bits 61-63)

    pub fn new(value: &[u8], offset: usize, flag: OVCFlag) -> Self {
        debug_assert!(value.len() <= 6, "Value must be at most 6 bytes long");
        let mut value_u64 = 0_u64;
        for i in 0..value.len() {
            value_u64 |= (value[i] as u64) << (8 * (5 - i));
        }

        // Calculate arity_minus_offset (13 bits max)
        let arity_minus_offset = (Self::ARITY_MAX as usize)
            .saturating_sub(offset)
            .min(Self::ARITY_MAX as usize) as u64;

        // Combine all parts
        let packed = value_u64
            | (arity_minus_offset << Self::ARITY_SHIFT)
            | ((flag.as_u8() as u64) << Self::FLAG_SHIFT);

        Self(packed)
    }

    pub fn early_fence() -> Self {
        let packed = 0_u64 | (OVCFlag::EarlyFence.as_u8() as u64) << Self::FLAG_SHIFT;
        Self(packed)
    }

    pub fn is_early_fence(&self) -> bool {
        self.flag() == OVCFlag::EarlyFence
    }

    pub fn late_fence() -> Self {
        let packed = 0_u64 | (OVCFlag::LateFence.as_u8() as u64) << Self::FLAG_SHIFT;
        Self(packed)
    }

    pub fn is_late_fence(&self) -> bool {
        self.flag() == OVCFlag::LateFence
    }

    pub fn duplicate_value() -> Self {
        let packed = 0_u64 | (OVCFlag::DuplicateValue.as_u8() as u64) << Self::FLAG_SHIFT;
        Self(packed)
    }

    pub fn is_duplicate_value(&self) -> bool {
        self.flag() == OVCFlag::DuplicateValue
    }

    pub fn normal_value(val: &[u8], offset: usize) -> Self {
        let packed = Self::new(val, offset, OVCFlag::NormalValue);
        packed
    }

    pub fn is_normal_value(&self) -> bool {
        self.flag() == OVCFlag::NormalValue
    }

    pub fn initial_value() -> Self {
        let packed = 0_u64 | (OVCFlag::InitialValue.as_u8() as u64) << Self::FLAG_SHIFT;
        Self(packed)
    }

    pub fn is_initial_value(&self) -> bool {
        self.flag() == OVCFlag::InitialValue
    }

    pub fn value(&self) -> [u8; 6] {
        let val = self.0.to_be_bytes();
        val[2..8].try_into().unwrap()
    }

    pub fn value_as_u64(&self) -> u64 {
        self.0 & Self::VALUE_MASK
    }

    pub fn arity_minus_offset(&self) -> u16 {
        ((self.0 >> Self::ARITY_SHIFT) & Self::ARITY_MASK) as u16
    }

    pub fn offset(&self) -> usize {
        Self::ARITY_MAX as usize - self.arity_minus_offset() as usize
    }

    pub fn flag(&self) -> OVCFlag {
        let flag_bits = ((self.0 >> Self::FLAG_SHIFT) & Self::FLAG_MASK) as u8;
        OVCFlag::from_u8(flag_bits).unwrap_or(OVCFlag::EarlyFence)
    }

    pub fn to_le_bytes(&self) -> [u8; 8] {
        self.0.to_le_bytes()
    }

    pub fn from_le_bytes(bytes: [u8; 8]) -> Self {
        let packed = u64::from_le_bytes(bytes);
        Self(packed)
    }

    pub fn to_be_bytes(&self) -> [u8; 8] {
        self.0.to_be_bytes()
    }

    pub fn from_be_bytes(bytes: [u8; 8]) -> Self {
        let packed = u64::from_be_bytes(bytes);
        Self(packed)
    }
}

impl std::fmt::Debug for OVCU64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = self.value();

        let offset = self.offset();
        let flag = self.flag();
        let flag_str = match flag {
            OVCFlag::EarlyFence => "EarlyFence",
            OVCFlag::DuplicateValue => "DuplicateValue",
            OVCFlag::NormalValue => "NormalValue",
            OVCFlag::InitialValue => "InitialValue",
            OVCFlag::LateFence => "LateFence",
        };
        write!(
            f,
            "OVC {{ value: {:?}, offset: {}, flag: {} }}",
            value, offset, flag_str
        )
    }
}

impl std::fmt::Display for OVCU64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = self.value();
        let value_as_u64 = self.value_as_u64();
        let offset = self.offset();
        let flag = self.flag();
        let flag_str = match flag {
            OVCFlag::EarlyFence => "[E]".to_string(),
            OVCFlag::DuplicateValue => format!("[D @ {}]", offset),
            OVCFlag::NormalValue => format!("[V:{:?}({}) @ {}]", value, value_as_u64, offset),
            OVCFlag::LateFence => "[L]".to_string(),
            OVCFlag::InitialValue => "[I]".to_string(),
        };
        write!(f, "{}", flag_str)
    }
}

#[derive(Clone, PartialEq, Eq)]
// Do not implement Ord or PartialOrd for OVCEntry
// Use `compare_and_update` method for ordering
pub struct OVCEntry64 {
    ovc: OVCU64,
    key: Vec<u8>,
}

impl std::fmt::Debug for OVCEntry64 {
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

impl std::fmt::Display for OVCEntry64 {
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

impl From<Vec<u8>> for OVCEntry64 {
    fn from(key: Vec<u8>) -> Self {
        let ovc = OVCU64::initial_value();
        Self { ovc, key }
    }
}

impl OVCEntry64 {
    pub fn new(key: Vec<u8>) -> Self {
        let ovc = OVCU64::initial_value();
        Self { ovc, key }
    }

    pub fn get_ovc(&self) -> OVCU64 {
        self.ovc
    }

    pub fn get_key(&self) -> &[u8] {
        &self.key
    }
}

impl OVCTrait for OVCEntry64 {
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

impl SentinelValue for OVCEntry64 {
    fn early_fence() -> Self {
        Self {
            ovc: OVCU64::early_fence(),
            key: Vec::new(),
        }
    }

    fn late_fence() -> Self {
        Self {
            ovc: OVCU64::late_fence(),
            key: Vec::new(),
        }
    }

    fn is_early_fence(&self) -> bool {
        self.ovc.is_early_fence()
    }

    fn is_late_fence(&self) -> bool {
        self.ovc.is_late_fence()
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use crate::ovc::utils::random_string;

    use super::*;

    #[test]
    fn test_simple_ovc() {
        let base = vec![1, 2, 3, 4];
        let duplicate = vec![1, 2, 3, 4];
        let longer = vec![1, 2, 3, 4, 0];

        let mut ovc_base = OVCEntry64::new(base);
        let mut ovc_duplicate = OVCEntry64::new(duplicate);
        let mut ovc_longer = OVCEntry64::new(longer);

        assert_eq!(
            ovc_duplicate.compare_and_update(&mut ovc_base),
            Ordering::Equal
        );
        assert_eq!(
            ovc_longer.compare_and_update(&mut ovc_base),
            Ordering::Greater
        );

        println!("Base OVC: {:?}", ovc_base);
        println!("Duplicate 4: {:?}", ovc_duplicate);
        println!("Longer OVC: {:?}", ovc_longer);
        assert!(ovc_duplicate.get_ovc() < ovc_longer.get_ovc());
        assert_eq!(ovc_duplicate.get_ovc(), OVCU64::duplicate_value());
    }

    #[test]
    fn test_run_encoding() {
        let run = vec![
            vec![1, 2, 3],
            vec![1, 2, 3],
            vec![1, 2, 3, 0],
            vec![1, 2, 3, 0, 0],
        ];

        let ovc_run = encode_run_with_ovc64(&run);
        assert_eq!(ovc_run.len(), 4);
        assert!(ovc_run[0].get_ovc().is_initial_value());
        assert!(ovc_run[1].get_ovc().is_duplicate_value());
        assert!(
            ovc_run[2].get_ovc().offset() == 0
                && ovc_run[2].get_ovc().value() == [1, 2, 3, 0, 0, 0]
        );
        assert!(
            ovc_run[3].get_ovc().offset() == 0
                && ovc_run[3].get_ovc().value() == [1, 2, 3, 0, 0, 0]
        );

        println!("OvC Run");
        for entry in &ovc_run {
            println!("{:?}", entry);
        }
    }

    #[test]
    fn test_update() {
        let base = vec![9, 1, 3];
        let this = vec![9, 1, 4];
        let ovc_base = OVCEntry64::new(base);
        let mut ovc = OVCEntry64::new(this);

        ovc.update(&ovc_base);
        assert!(ovc.get_ovc().offset() == 0 && ovc.get_ovc().value() == [9, 1, 4, 0, 0, 0]);
    }

    #[test]
    fn test_order_different_ovcs() {
        let base = vec![1, 2, 3, 4, 5];
        let ovc_base = OVCEntry64::new(base);

        // [NOTE] Value with sort order smaller than base does not work!!!
        // let val1 = vec![1, 1, 1, 1, 1];
        // let mut ovc1 = OVCEntry::new(val1);
        // ovc1.update(&ovc_base);

        let val2 = vec![1, 2, 3, 4, 5];
        let mut ovc2 = OVCEntry64::new(val2);
        ovc2.update(&ovc_base);

        let val3 = vec![1, 2, 3, 5, 5];
        let mut ovc3 = OVCEntry64::new(val3);
        ovc3.update(&ovc_base);

        let val4 = vec![1, 2, 7, 5, 5];
        let mut ovc4 = OVCEntry64::new(val4);
        ovc4.update(&ovc_base);

        let val5 = vec![1, 3, 5, 5, 5];
        let mut ovc5 = OVCEntry64::new(val5);
        ovc5.update(&ovc_base);

        let val6 = vec![2, 5, 5, 5, 5];
        let mut ovc6 = OVCEntry64::new(val6);
        ovc6.update(&ovc_base);

        let mut vec = vec![
            ovc6.clone(),
            ovc5.clone(),
            ovc4.clone(),
            ovc3.clone(),
            ovc2.clone(),
        ];
        vec.sort_by_key(|entry| entry.get_ovc());

        println!("Sorted OVCs:");
        for entry in &vec {
            println!("{:?}, code: {:?}", entry, entry.get_ovc());
        }

        let expected = vec![ovc2, ovc3, ovc4, ovc5, ovc6];

        assert_eq!(vec, expected);
    }

    #[test]
    fn test_order_same_ovcs_different_keys() {
        let base = vec![1, 2, 3, 4, 5];
        let ovc_base = OVCEntry64::new(base);

        let val1 = vec![1, 2, 3, 6, 1];
        let mut ovc1 = OVCEntry64::new(val1);
        ovc1.update(&ovc_base);

        let val2 = vec![1, 2, 3, 6, 2];
        let mut ovc2 = OVCEntry64::new(val2);
        ovc2.update(&ovc_base);

        let val3 = vec![1, 2, 3, 6, 3];
        let mut ovc3 = OVCEntry64::new(val3);
        ovc3.update(&ovc_base);

        let val4 = vec![1, 2, 3, 6, 4];
        let mut ovc4 = OVCEntry64::new(val4);
        ovc4.update(&ovc_base);

        let val5 = vec![1, 2, 3, 6, 5];
        let mut ovc5 = OVCEntry64::new(val5);
        ovc5.update(&ovc_base);

        let mut vec = vec![
            ovc5.clone(),
            ovc4.clone(),
            ovc3.clone(),
            ovc2.clone(),
            ovc1.clone(),
        ];

        vec.sort_by(|a, b| {
            let a_ovc = a.get_ovc();
            let b_ovc = b.get_ovc();
            match a_ovc.cmp(&b_ovc) {
                Ordering::Equal => {
                    // If the OVCs are equal, compare the values
                    let a_val = a.get_key();
                    let b_val = b.get_key();
                    a_val.cmp(b_val)
                }
                ord => ord,
            }
        });

        let expected = vec![ovc1, ovc2, ovc3, ovc4, ovc5];

        assert_eq!(vec, expected);
    }

    #[test]
    fn test_order_same_ovcs_same_keys() {
        let base = vec![1, 2, 3, 4, 5];
        let ovc_base = OVCEntry64::new(base);

        let val1 = vec![1, 2, 3, 4, 6];
        let mut ovc1 = OVCEntry64::new(val1);
        ovc1.update(&ovc_base);

        let val2 = vec![1, 2, 3, 4, 6];
        let mut ovc2 = OVCEntry64::new(val2);
        ovc2.update(&ovc_base);

        assert_eq!(ovc1, ovc2);
    }

    #[test]
    fn test_order_different_lengths() {
        let base = vec![1, 2, 3];
        let ovc_base = OVCEntry64::new(base);

        let test_vals = [
            vec![1, 2, 3],
            vec![1, 2, 3, 0],
            vec![1, 2, 3, 0, 0],
            vec![1, 2, 3, 4],
            vec![1, 2, 3, 4, 5],
            vec![1, 2, 3, 4, 5, 6],
            vec![1, 2, 3, 4, 5, 6, 0],
        ];

        let mut ovcs: Vec<OVCEntry64> = test_vals
            .iter()
            .map(|val| {
                let mut ovc = OVCEntry64::new(val.clone());
                ovc.update(&ovc_base);
                ovc
            })
            .collect();

        ovcs.sort_by(|a, b| {
            let a_ovc = a.get_ovc();
            let b_ovc = b.get_ovc();
            match a_ovc.cmp(&b_ovc) {
                Ordering::Equal => {
                    // If the OVCs are equal, compare the values
                    let a_val = a.get_key();
                    let b_val = b.get_key();
                    a_val.cmp(b_val)
                }
                ord => ord,
            }
        });

        println!("Sorted OVCs:");
        for entry in &ovcs {
            println!("{:?}", entry);
        }

        // Check that the OVCs are sorted correctly
        for i in 0..test_vals.len() {
            assert_eq!(ovcs[i].get_key(), test_vals[i]);
        }
    }

    #[test]
    fn test_ovc_with_normalize() {
        use rand::seq::SliceRandom;
        let mut rng = rand::rng();

        let run_length = 100;
        let original_run = (0..run_length)
            .map(|_| random_string(rng.random_range(1..=10)))
            .collect::<Vec<_>>();

        // Normalize the run. This is sorted in the same order as the original run.
        let mut normalized_run = original_run
            .iter()
            .map(|s| s.as_bytes().to_vec())
            .collect::<Vec<_>>();

        // Make sure that all elements are unique
        normalized_run.sort();

        // Create the offset value codings with base as the smallest element
        let ovc_base = OVCEntry64::new(normalized_run[0].clone());
        let mut ovcs = normalized_run
            .iter()
            .map(|s| OVCEntry64::new(s.clone()))
            .collect::<Vec<_>>();
        ovcs.iter_mut().for_each(|ovc| ovc.update(&ovc_base));

        // Randomize the order of the offset value codings
        ovcs.shuffle(&mut rng);

        // Sort the offset value codings
        ovcs.sort_by(|a, b| {
            let a_ovc = a.get_ovc();
            let b_ovc = b.get_ovc();
            match a_ovc.cmp(&b_ovc) {
                Ordering::Equal => {
                    // If the OVCs are equal, compare the values
                    let a_val = a.get_key();
                    let b_val = b.get_key();
                    a_val.cmp(b_val)
                }
                ord => ord,
            }
        });

        // Print the sorted OVCs
        println!("Sorted OVCs:");
        for entry in &ovcs {
            println!("{:?}", entry);
        }

        // Check that the order of the offset value codings is the same as the normalized run
        for (ovc, expected) in ovcs.iter().zip(normalized_run.iter()) {
            // Dereference the pointer to get the value
            assert_eq!(ovc.get_key(), expected);
        }
    }

    #[test]
    fn test_compare_and_update_different_ovcs() {
        let base = vec![1, 2, 3, 4, 5];
        let ovc_base = OVCEntry64::new(base);

        // Different OVCs relative to base, different keys
        let val1 = vec![1, 2, 3, 6, 1];
        let mut ovc1 = OVCEntry64::new(val1);
        ovc1.update(&ovc_base);
        let val2 = vec![1, 2, 4, 6, 2];
        let mut ovc2 = OVCEntry64::new(val2);
        ovc2.update(&ovc_base);

        assert!(ovc1.get_ovc().offset() == 0 && ovc1.get_ovc().value() == [1, 2, 3, 6, 1, 0]);
        assert!(ovc2.get_ovc().offset() == 0 && ovc2.get_ovc().value() == [1, 2, 4, 6, 2, 0]);

        let res = OVCEntry64::compare_and_update(&mut ovc1, &mut ovc2);
        assert_eq!(res, Ordering::Less);
        assert!(ovc1.get_ovc().offset() == 0 && ovc1.get_ovc().value() == [1, 2, 3, 6, 1, 0]);
        assert!(ovc2.get_ovc().offset() == 0 && ovc2.get_ovc().value() == [1, 2, 4, 6, 2, 0]);

        let res = OVCEntry64::compare_and_update(&mut ovc2, &mut ovc1);
        assert_eq!(res, Ordering::Greater);
        assert!(ovc1.get_ovc().offset() == 0 && ovc1.get_ovc().value() == [1, 2, 3, 6, 1, 0]);
        assert!(ovc2.get_ovc().offset() == 0 && ovc2.get_ovc().value() == [1, 2, 4, 6, 2, 0]);
    }

    #[test]
    fn test_compare_and_update_same_ovcs_same_keys() {
        let base = vec![1, 2, 3, 4, 5];
        let ovc_base = OVCEntry64::new(base);

        // Same OVCs relative to base, same keys
        let val1 = vec![1, 2, 3, 4, 6];
        let mut ovc1 = OVCEntry64::new(val1);
        ovc1.update(&ovc_base);
        let val2 = vec![1, 2, 3, 4, 6];
        let mut ovc2 = OVCEntry64::new(val2);
        ovc2.update(&ovc_base);

        let ovc = ovc1.get_ovc();
        assert!(ovc.offset() == 0 && ovc.value() == [1, 2, 3, 4, 6, 0]);
        let ovc = ovc2.get_ovc();
        assert!(ovc.offset() == 0 && ovc.value() == [1, 2, 3, 4, 6, 0]);

        let res = ovc1.compare_and_update(&mut ovc2);
        assert!(res == Ordering::Equal);

        // The OVCs should remain the same
        let ovc = ovc1.get_ovc();
        assert!(ovc.is_duplicate_value());
        let ovc = ovc2.get_ovc();
        assert!(ovc.offset() == 0 && ovc.value() == [1, 2, 3, 4, 6, 0]);
    }

    #[test]
    fn test_compare_and_update_same_ovcs_different_keys() {
        let base = vec![1, 2, 3, 4, 5];
        let ovc_base = OVCEntry64::new(base);

        // Same OVCs relative to base but different values
        let val1 = vec![1, 2, 3, 6, 1];
        let mut ovc1 = OVCEntry64::new(val1);
        ovc1.update(&ovc_base);
        let val2 = vec![1, 2, 3, 6, 2];
        let mut ovc2 = OVCEntry64::new(val2);
        ovc2.update(&ovc_base);

        let ovc = ovc1.get_ovc();
        assert!(ovc.offset() == 0 && ovc.value() == [1, 2, 3, 6, 1, 0]);
        let ovc = ovc2.get_ovc();
        assert!(ovc.offset() == 0 && ovc.value() == [1, 2, 3, 6, 2, 0]);

        let res = ovc1.compare_and_update(&mut ovc2);
        assert!(res == Ordering::Less);
        let ovc = ovc1.get_ovc();
        assert!(ovc.offset() == 0 && ovc.value() == [1, 2, 3, 6, 1, 0]);
        let ovc = ovc2.get_ovc();
        assert!(ovc.offset() == 0 && ovc.value() == [1, 2, 3, 6, 2, 0]);

        // ovc1.reset();
        ovc1.update(&ovc_base);
        // ovc2.reset();
        ovc2.update(&ovc_base);

        let ovc = ovc1.get_ovc();
        assert!(ovc.offset() == 0 && ovc.value() == [1, 2, 3, 6, 1, 0]);
        let ovc = ovc2.get_ovc();
        assert!(ovc.offset() == 0 && ovc.value() == [1, 2, 3, 6, 2, 0]);

        let res = OVCEntry64::compare_and_update(&mut ovc2, &mut ovc1);
        assert!(res == Ordering::Greater);
        let ovc = ovc1.get_ovc();
        assert!(ovc.offset() == 0 && ovc.value() == [1, 2, 3, 6, 1, 0]);
        let ovc = ovc2.get_ovc();
        assert!(ovc.offset() == 0 && ovc.value() == [1, 2, 3, 6, 2, 0]);
    }

    #[test]
    fn test_compare_and_update_same_ovcs_different_keys_long() {
        let base = vec![1, 2, 3, 4, 5];
        let ovc_base = OVCEntry64::new(base);

        // Same OVCs relative to base but different values
        let val1 = vec![1, 2, 3, 6, 1, 7, 8];
        let mut ovc1 = OVCEntry64::new(val1);
        ovc1.update(&ovc_base);
        let val2 = vec![1, 2, 3, 6, 1, 7, 9];
        let mut ovc2 = OVCEntry64::new(val2);
        ovc2.update(&ovc_base);

        let ovc = ovc1.get_ovc();
        assert!(ovc.offset() == 0 && ovc.value() == [1, 2, 3, 6, 1, 7]);
        let ovc = ovc2.get_ovc();
        assert!(ovc.offset() == 0 && ovc.value() == [1, 2, 3, 6, 1, 7]);

        let res = ovc1.compare_and_update(&mut ovc2);
        assert!(res == Ordering::Less);
        let ovc = ovc1.get_ovc();
        assert!(ovc.offset() == 0 && ovc.value() == [1, 2, 3, 6, 1, 7]);
        let ovc = ovc2.get_ovc();
        assert!(ovc.offset() == 6 && ovc.value() == [9, 0, 0, 0, 0, 0]);

        // Reset and update again
        // ovc1.reset();
        ovc1.update(&ovc_base);
        // ovc2.reset();
        ovc2.update(&ovc_base);

        let ovc = ovc1.get_ovc();
        assert!(ovc.offset() == 0 && ovc.value() == [1, 2, 3, 6, 1, 7]);
        let ovc = ovc2.get_ovc();
        assert!(ovc.offset() == 0 && ovc.value() == [1, 2, 3, 6, 1, 7]);

        let res = OVCEntry64::compare_and_update(&mut ovc2, &mut ovc1);
        assert!(res == Ordering::Greater);
        let ovc = ovc1.get_ovc();
        assert!(ovc.offset() == 0 && ovc.value() == [1, 2, 3, 6, 1, 7]);
        let ovc = ovc2.get_ovc();
        assert!(ovc.offset() == 6 && ovc.value() == [9, 0, 0, 0, 0, 0]);
    }
}
