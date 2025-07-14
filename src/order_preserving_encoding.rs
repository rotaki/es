//! Order-preserving byte encoding for numerical types
//!
//! This module provides functions to convert numerical values to byte arrays
//! that preserve the original ordering when compared lexicographically.

/// Convert i32 to order-preserving bytes
pub fn i32_to_order_preserving_bytes(val: i32) -> [u8; 4] {
    // XOR with sign bit to make negative numbers sort before positive
    let bits = (val as u32) ^ (1 << 31);
    bits.to_be_bytes()
}

/// Convert i64 to order-preserving bytes
pub fn i64_to_order_preserving_bytes(val: i64) -> [u8; 8] {
    // XOR with sign bit to make negative numbers sort before positive
    let bits = (val as u64) ^ (1 << 63);
    bits.to_be_bytes()
}

/// Convert f32 to order-preserving bytes
pub fn f32_to_order_preserving_bytes(val: f32) -> [u8; 4] {
    let mut val_bits = val.to_bits();
    let sign = (val_bits >> 31) as u8;
    if sign == 1 {
        // Negative number so flip all the bits including the sign bit
        val_bits = !val_bits;
    } else {
        // Positive number. To distinguish between positive and negative numbers,
        // we flip the sign bit.
        val_bits ^= 1 << 31;
    }
    val_bits.to_be_bytes()
}

/// Convert f64 to order-preserving bytes
pub fn f64_to_order_preserving_bytes(val: f64) -> [u8; 8] {
    let mut val_bits = val.to_bits();
    let sign = (val_bits >> 63) as u8;
    if sign == 1 {
        // Negative number so flip all the bits including the sign bit
        val_bits = !val_bits;
    } else {
        // Positive number. To distinguish between positive and negative numbers,
        // we flip the sign bit.
        val_bits ^= 1 << 63;
    }
    val_bits.to_be_bytes()
}

/// Convert i128 to order-preserving bytes
pub fn i128_to_order_preserving_bytes(val: i128) -> [u8; 16] {
    // XOR with sign bit to make negative numbers sort before positive
    let bits = (val as u128) ^ (1 << 127);
    bits.to_be_bytes()
}

/// Decode i32 from order-preserving bytes
pub fn i32_from_order_preserving_bytes(bytes: [u8; 4]) -> i32 {
    let bits = u32::from_be_bytes(bytes) ^ (1 << 31);
    bits as i32
}

/// Decode i64 from order-preserving bytes
pub fn i64_from_order_preserving_bytes(bytes: [u8; 8]) -> i64 {
    let bits = u64::from_be_bytes(bytes) ^ (1 << 63);
    bits as i64
}

/// Decode f32 from order-preserving bytes
pub fn f32_from_order_preserving_bytes(bytes: [u8; 4]) -> f32 {
    let mut bits = u32::from_be_bytes(bytes);
    if bits & (1 << 31) != 0 {
        // Positive number, flip sign bit back
        bits ^= 1 << 31;
    } else {
        // Negative number, flip all bits back
        bits = !bits;
    }
    f32::from_bits(bits)
}

/// Decode f64 from order-preserving bytes
pub fn f64_from_order_preserving_bytes(bytes: [u8; 8]) -> f64 {
    let mut bits = u64::from_be_bytes(bytes);
    if bits & (1 << 63) != 0 {
        // Positive number, flip sign bit back
        bits ^= 1 << 63;
    } else {
        // Negative number, flip all bits back
        bits = !bits;
    }
    f64::from_bits(bits)
}

/// Decode i128 from order-preserving bytes
pub fn i128_from_order_preserving_bytes(bytes: [u8; 16]) -> i128 {
    let bits = u128::from_be_bytes(bytes) ^ (1 << 127);
    bits as i128
}

/// Decode a Vec<u8> as an i32 (assumes order-preserving encoding)
pub fn decode_i32(bytes: &[u8]) -> Result<i32, String> {
    if bytes.len() != 4 {
        return Err(format!("Expected 4 bytes for i32, got {}", bytes.len()));
    }
    let array: [u8; 4] = bytes.try_into().unwrap();
    Ok(i32_from_order_preserving_bytes(array))
}

/// Decode a Vec<u8> as an i64 (assumes order-preserving encoding)
pub fn decode_i64(bytes: &[u8]) -> Result<i64, String> {
    if bytes.len() != 8 {
        return Err(format!("Expected 8 bytes for i64, got {}", bytes.len()));
    }
    let array: [u8; 8] = bytes.try_into().unwrap();
    Ok(i64_from_order_preserving_bytes(array))
}

/// Decode a Vec<u8> as an f32 (assumes order-preserving encoding)
pub fn decode_f32(bytes: &[u8]) -> Result<f32, String> {
    if bytes.len() != 4 {
        return Err(format!("Expected 4 bytes for f32, got {}", bytes.len()));
    }
    let array: [u8; 4] = bytes.try_into().unwrap();
    Ok(f32_from_order_preserving_bytes(array))
}

/// Decode a Vec<u8> as an f64 (assumes order-preserving encoding)
pub fn decode_f64(bytes: &[u8]) -> Result<f64, String> {
    if bytes.len() != 8 {
        return Err(format!("Expected 8 bytes for f64, got {}", bytes.len()));
    }
    let array: [u8; 8] = bytes.try_into().unwrap();
    Ok(f64_from_order_preserving_bytes(array))
}

/// Decode a Vec<u8> as an i128 (assumes order-preserving encoding)
pub fn decode_i128(bytes: &[u8]) -> Result<i128, String> {
    if bytes.len() != 16 {
        return Err(format!("Expected 16 bytes for i128, got {}", bytes.len()));
    }
    let array: [u8; 16] = bytes.try_into().unwrap();
    Ok(i128_from_order_preserving_bytes(array))
}

/// Decode a Vec<u8> as a Date32 (days since epoch, encoded as i32)
pub fn decode_date32(bytes: &[u8]) -> Result<i32, String> {
    decode_i32(bytes)
}

/// Format decoded Date32 as YYYY-MM-DD string
pub fn format_date32(days_from_ce: i32) -> String {
    use chrono::NaiveDate;
    match NaiveDate::from_num_days_from_ce_opt(days_from_ce) {
        Some(date) => date.format("%Y-%m-%d").to_string(),
        None => format!("Invalid date: {} days from CE", days_from_ce),
    }
}

/// Decode a Vec<u8> as a UTF-8 string
pub fn decode_utf8(bytes: &[u8]) -> Result<String, String> {
    String::from_utf8(bytes.to_vec()).map_err(|e| format!("Invalid UTF-8: {}", e))
}

/// Generic decoder that takes data type information
pub fn decode_bytes(bytes: &[u8], data_type: &str) -> Result<String, String> {
    match data_type {
        "i32" | "Int32" => decode_i32(bytes).map(|v| v.to_string()),
        "i64" | "Int64" => decode_i64(bytes).map(|v| v.to_string()),
        "f32" | "Float32" => decode_f32(bytes).map(|v| v.to_string()),
        "f64" | "Float64" => decode_f64(bytes).map(|v| v.to_string()),
        "i128" | "Int128" | "Decimal128" => decode_i128(bytes).map(|v| v.to_string()),
        "date32" | "Date32" => decode_date32(bytes).map(format_date32),
        "utf8" | "Utf8" | "string" | "String" => decode_utf8(bytes),
        _ => Ok(format!("<{} bytes>", bytes.len())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_i32_encoding() {
        // Test that negative numbers sort before positive
        let neg = i32_to_order_preserving_bytes(-10);
        let zero = i32_to_order_preserving_bytes(0);
        let pos = i32_to_order_preserving_bytes(10);

        assert!(neg < zero);
        assert!(zero < pos);

        // Test round-trip
        assert_eq!(i32_from_order_preserving_bytes(neg), -10);
        assert_eq!(i32_from_order_preserving_bytes(zero), 0);
        assert_eq!(i32_from_order_preserving_bytes(pos), 10);
    }

    #[test]
    fn test_f64_encoding() {
        // Test that negative numbers sort before positive
        let neg = f64_to_order_preserving_bytes(-10.5);
        let zero = f64_to_order_preserving_bytes(0.0);
        let pos = f64_to_order_preserving_bytes(10.5);

        assert!(neg < zero);
        assert!(zero < pos);

        // Test round-trip
        assert_eq!(f64_from_order_preserving_bytes(neg), -10.5);
        assert_eq!(f64_from_order_preserving_bytes(zero), 0.0);
        assert_eq!(f64_from_order_preserving_bytes(pos), 10.5);
    }

    #[test]
    fn test_vec_decoding() {
        // Test decoding from Vec<u8>
        let i32_bytes = i32_to_order_preserving_bytes(42).to_vec();
        assert_eq!(decode_i32(&i32_bytes).unwrap(), 42);

        let i64_bytes = i64_to_order_preserving_bytes(-100).to_vec();
        assert_eq!(decode_i64(&i64_bytes).unwrap(), -100);

        let f32_bytes = f32_to_order_preserving_bytes(3.14).to_vec();
        assert!((decode_f32(&f32_bytes).unwrap() - 3.14).abs() < 0.001);

        let f64_bytes = f64_to_order_preserving_bytes(-2.718).to_vec();
        assert!((decode_f64(&f64_bytes).unwrap() - (-2.718)).abs() < 0.001);
    }

    #[test]
    fn test_generic_decoder() {
        let i32_bytes = i32_to_order_preserving_bytes(42).to_vec();
        assert_eq!(decode_bytes(&i32_bytes, "i32").unwrap(), "42");
        assert_eq!(decode_bytes(&i32_bytes, "Int32").unwrap(), "42");

        let utf8_bytes = b"hello world".to_vec();
        assert_eq!(decode_bytes(&utf8_bytes, "utf8").unwrap(), "hello world");
        assert_eq!(decode_bytes(&utf8_bytes, "String").unwrap(), "hello world");

        // Test date formatting
        // Calculate correct days_from_ce for 2024-01-01
        use chrono::{Datelike, NaiveDate};
        let date = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let days_from_ce = date.num_days_from_ce();
        let date_bytes = i32_to_order_preserving_bytes(days_from_ce).to_vec();
        assert_eq!(decode_bytes(&date_bytes, "Date32").unwrap(), "2024-01-01");
    }

    #[test]
    fn test_order_preserving_encoding() {
        // Test i32 encoding
        let i32_values = vec![-1000, -1, 0, 1, 1000];
        let mut i32_encoded: Vec<_> = i32_values
            .iter()
            .map(|&v| i32_to_order_preserving_bytes(v))
            .collect();
        i32_encoded.sort();
        let i32_decoded: Vec<_> = i32_encoded
            .iter()
            .map(|b| i32_from_order_preserving_bytes(*b))
            .collect();
        assert_eq!(i32_decoded, vec![-1000, -1, 0, 1, 1000]);

        // Test i64 encoding
        let i64_values = vec![-1000000, -1, 0, 1, 1000000];
        let mut i64_encoded: Vec<_> = i64_values
            .iter()
            .map(|&v| i64_to_order_preserving_bytes(v))
            .collect();
        i64_encoded.sort();
        let i64_decoded: Vec<_> = i64_encoded
            .iter()
            .map(|b| i64_from_order_preserving_bytes(*b))
            .collect();
        assert_eq!(i64_decoded, vec![-1000000, -1, 0, 1, 1000000]);

        // Test i128 encoding
        let i128_values = vec![-1000000000, -1, 0, 1, 1000000000];
        let mut i128_encoded: Vec<_> = i128_values
            .iter()
            .map(|&v| i128_to_order_preserving_bytes(v))
            .collect();
        i128_encoded.sort();
        let i128_decoded: Vec<_> = i128_encoded
            .iter()
            .map(|b| i128_from_order_preserving_bytes(*b))
            .collect();
        assert_eq!(i128_decoded, vec![-1000000000, -1, 0, 1, 1000000000]);

        // Test f64 encoding with special values
        let f64_values = vec![
            f64::NEG_INFINITY,
            -1000.0,
            -1.0,
            -0.0,
            0.0,
            1.0,
            1000.0,
            f64::INFINITY,
            f64::NAN,
        ];
        let mut f64_encoded: Vec<_> = f64_values
            .iter()
            .map(|&v| f64_to_order_preserving_bytes(v))
            .collect();
        f64_encoded.sort();
        let f64_decoded: Vec<_> = f64_encoded
            .iter()
            .map(|b| f64_from_order_preserving_bytes(*b))
            .collect();

        // Check order (NaN comparison is tricky)
        assert_eq!(f64_decoded[0], f64::NEG_INFINITY);
        assert_eq!(f64_decoded[1], -1000.0);
        assert_eq!(f64_decoded[2], -1.0);
        assert!(f64_decoded[3] == -0.0 || f64_decoded[3] == 0.0); // -0.0 and 0.0 might be equal
        assert!(f64_decoded[4] == -0.0 || f64_decoded[4] == 0.0);
        assert_eq!(f64_decoded[5], 1.0);
        assert_eq!(f64_decoded[6], 1000.0);
        assert_eq!(f64_decoded[7], f64::INFINITY);
        assert!(f64_decoded[8].is_nan());
    }
}
