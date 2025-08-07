use std::cmp::Ordering;

use crate::ovc::entry::SentinelValue;

pub trait OVCTrait: SentinelValue {
    fn update(&mut self, prev: &Self);
    fn compare_and_update(&mut self, other: &mut Self) -> Ordering;
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum OVCFlag {
    EarlyFence = 0,
    DuplicateValue = 1,
    NormalValue = 2,
    InitialValue = 3,
    LateFence = 4,
}

impl OVCFlag {
    pub fn as_u8(&self) -> u8 {
        *self as u8
    }

    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(OVCFlag::EarlyFence),
            1 => Some(OVCFlag::DuplicateValue),
            2 => Some(OVCFlag::NormalValue),
            3 => Some(OVCFlag::InitialValue),
            4 => Some(OVCFlag::LateFence),
            _ => None,
        }
    }
}
