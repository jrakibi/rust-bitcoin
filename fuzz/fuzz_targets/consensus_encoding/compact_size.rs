// SPDX-License-Identifier: CC0-1.0

//! Fuzz test for CompactSize encoding/decoding.

use honggfuzz::fuzz;

use consensus_encoding::{Decoder, Encoder};


fn do_test(data: &[u8]) {
    // Create a decoder and try to decode the random bytes as a CompactSize
    let mut decoder = consensus_encoding::CompactSizeDecoder::default();
    let mut remaining = data;
    
    // Push bytes into the decoder
    let push_result = decoder.push_bytes(&mut remaining);
    
    match push_result {
        Err(_) => {
        }
        Ok(needs_more) => {
            // If push succeeded but needs more data, we can't complete decoding
            if needs_more {
                // Decoder needs more bytes but we've run out of data
                // Try to end anyway - this should fail gracefully
                let _ = decoder.end();
                return;
            }
            
            // Decoding succeeded! Try to finalize.
            match decoder.end() {
                Err(_) => {
                    // Failed at finalization - that's OK
                }
                Ok(value) => {
                    let encoder = consensus_encoding::CompactSizeEncoder::new(value);
                    
                    let mut encoded = Vec::new();
                    let mut enc = encoder;
                    while let Some(chunk) = enc.current_chunk() {
                        encoded.extend_from_slice(chunk);
                        enc.advance();
                    }

                    // CRITICAL CHECK: The round-trip must match!
                    let consumed_bytes = data.len() - remaining.len();
                    assert_eq!(
                        &encoded[..],
                        &data[..consumed_bytes],
                        "Round-trip encoding/decoding failed: decoded {}, but re-encoded as {:?} instead of {:?}",
                        value,
                        encoded,
                        &data[..consumed_bytes]
                    );
                }
            }
        }
    }
}

fn main() {
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
    fn test_compact_size_zero() {
        // CompactSize encoding of 0 is just one byte: 0x00
        let mut a = Vec::new();
        extend_vec_from_hex("00", &mut a);
        super::do_test(&a);
    }

    #[test]
    fn test_compact_size_fd() {
        // CompactSize encoding using 0xFD prefix (2-byte encoding)
        // 0xFD 0xFD 0x00 = 253 in CompactSize format
        let mut a = Vec::new();
        extend_vec_from_hex("fdfd00", &mut a);
        super::do_test(&a);
    }
}

