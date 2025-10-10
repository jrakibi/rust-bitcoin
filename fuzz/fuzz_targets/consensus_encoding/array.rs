// SPDX-License-Identifier: CC0-1.0

//! Fuzz test for Array encoding/decoding.


use honggfuzz::fuzz;

// Import the Decoder and Encoder traits so we can call their methods
use consensus_encoding::{Decoder, Encoder};

/// Test decoding/encoding for a 4-byte array.
/// This catches bugs in the ArrayDecoder/ArrayEncoder for small arrays.
fn test_array_4(data: &[u8]) {
    // Create a decoder for a 4-byte array
    let mut decoder = consensus_encoding::ArrayDecoder::<4>::new();
    let mut remaining = data;
    
    // Try to push bytes
    let push_result = decoder.push_bytes(&mut remaining);
    
    match push_result {
        Err(_) => {
            // Decoding failed - that's OK!
        }
        Ok(needs_more) => {
            if needs_more {
                let _ = decoder.end();
                return;
            }
            
            // Try to finalize
            match decoder.end() {
                Err(_) => {
                    // Failed - that's OK
                }
                Ok(array) => {
                    // Success! Encode it back.
                    let encoder = consensus_encoding::ArrayEncoder::without_length_prefix(array);
                    
                    // Manually encode
                    let mut encoded = Vec::new();
                    let mut enc = encoder;
                    while let Some(chunk) = enc.current_chunk() {
                        encoded.extend_from_slice(chunk);
                        enc.advance();
                    }

                    // Round-trip check: Should get back the first 4 bytes
                    let consumed = data.len() - remaining.len();
                    assert_eq!(
                        &encoded[..],
                        &data[..consumed],
                        "Round-trip failed for 4-byte array"
                    );
                }
            }
        }
    }
}

/// Test decoding/encoding for a 32-byte array.

fn test_array_32(data: &[u8]) {
    let mut decoder = consensus_encoding::ArrayDecoder::<32>::new();
    let mut remaining = data;
    
    let push_result = decoder.push_bytes(&mut remaining);
    
    match push_result {
        Err(_) => {}
        Ok(needs_more) => {
            if needs_more {
                let _ = decoder.end();
                return;
            }
            
            if let Ok(array) = decoder.end() {
                let encoder = consensus_encoding::ArrayEncoder::without_length_prefix(array);
                
                let mut encoded = Vec::new();
                let mut enc = encoder;
                while let Some(chunk) = enc.current_chunk() {
                    encoded.extend_from_slice(chunk);
                    enc.advance();
                }

                let consumed = data.len() - remaining.len();
                assert_eq!(
                    &encoded[..],
                    &data[..consumed],
                    "Round-trip failed for 32-byte array"
                );
            }
        }
    }
}


    let mut decoder = consensus_encoding::ArrayDecoder::<64>::new();
    let mut remaining = data;
    
    let push_result = decoder.push_bytes(&mut remaining);
    
    match push_result {
        Err(_) => {}
        Ok(needs_more) => {
            if needs_more {
                let _ = decoder.end();
                return;
            }
            
            if let Ok(array) = decoder.end() {
                let encoder = consensus_encoding::ArrayEncoder::without_length_prefix(array);
                
                let mut encoded = Vec::new();
                let mut enc = encoder;
                while let Some(chunk) = enc.current_chunk() {
                    encoded.extend_from_slice(chunk);
                    enc.advance();
                }

                let consumed = data.len() - remaining.len();
                assert_eq!(
                    &encoded[..],
                    &data[..consumed],
                    "Round-trip failed for 64-byte array"
                );
            }
        }
    }
}

/// Main test function that runs all array size tests.
fn do_test(data: &[u8]) {
    // Test multiple array sizes to get good coverage.
    // Each test is independent and handles its own errors.
    test_array_4(data);
    test_array_32(data);
    test_array_64(data);
}

fn main() {
    // Honggfuzz main loop - generates random bytes forever.
    loop {
        fuzz!(|data| {
            do_test(data);
        });
    }
}

#[cfg(all(test, fuzzing))]
mod tests {
    fn extend_vec_from_hex(hex: &str, out: &mut Vec<u8>) {
        let mut b = 0;
        for (idx, c) in hex.as_bytes().iter().enumerate() {
            b <<= 4;
            match *c {
                b'A'..=b'F' => b |= c - b'A' + 10,
                b'a'..=b'f' => b |= c - b'a' + 10,
                b'0'..=b'9' => b |= c - b'0',
                _ => panic!("Bad hex"),
            }
            if (idx & 1) == 1 {
                out.push(b);
                b = 0;
            }
        }
    }

    #[test]
    fn test_four_bytes() {
        // Test with exactly 4 bytes
        let mut a = Vec::new();
        extend_vec_from_hex("deadbeef", &mut a);
        super::do_test(&a);
    }

    #[test]
    fn test_32_bytes() {
        // Test with 32 bytes (a hash-sized array)
        let mut a = Vec::new();
        extend_vec_from_hex(
            "0000000000000000000000000000000000000000000000000000000000000000",
            &mut a,
        );
        super::do_test(&a);
    }

    #[test]
    fn test_insufficient_bytes() {
        // Test with only 2 bytes - should fail to decode larger arrays
        let mut a = Vec::new();
        extend_vec_from_hex("abcd", &mut a);
        super::do_test(&a);
    }
}

