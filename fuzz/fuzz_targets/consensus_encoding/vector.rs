// SPDX-License-Identifier: CC0-1.0

use honggfuzz::fuzz;

use consensus_encoding::{Decoder, Encoder};


fn do_test(data: &[u8]) {
    // Create a ByteVecDecoder and try to decode the random bytes
    let mut decoder = consensus_encoding::ByteVecDecoder::new();
    let mut remaining = data;
    
    // Push bytes into the decoder
    let push_result = decoder.push_bytes(&mut remaining);
    
    match push_result {
        Err(_) => {

        }
        Ok(needs_more) => {
            if needs_more {
                let _ = decoder.end();
                return;
            }
            
            match decoder.end() {
                Err(_) => {
                    // Failed at finalization - that's OK
                }
                Ok(decoded_vec) => {
                    // Decoding succeeded! Now encode it back.
                    let encoder = consensus_encoding::BytesEncoder::with_length_prefix(&decoded_vec);
                    
                    // Manually encode by iterating through chunks
                    let mut encoded = Vec::new();
                    let mut enc = encoder;
                    while let Some(chunk) = enc.current_chunk() {
                        encoded.extend_from_slice(chunk);
                        enc.advance();
                    }


                    // If this assertion fails, there's a mismatch between encoder/decoder.
                    let consumed = data.len() - remaining.len();
                    assert_eq!(
                        &encoded[..],
                        &data[..consumed],
                        "Round-trip failed for vector of length {}: encoded {:?} != original {:?}",
                        decoded_vec.len(),
                        encoded,
                        &data[..consumed]
                    );
                }
            }
        }
    }
}

fn main() {
    // Honggfuzz main loop
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
    fn test_empty_vector() {
        // Empty vector: just 0x00 (CompactSize for length 0)
        let mut a = Vec::new();
        extend_vec_from_hex("00", &mut a);
        super::do_test(&a);
    }

    #[test]
    fn test_single_byte_vector() {
        // Vector with 1 byte: 0x01 (length) followed by 0xFF (data)
        let mut a = Vec::new();
        extend_vec_from_hex("01ff", &mut a);
        super::do_test(&a);
    }

    #[test]
    fn test_vector_with_data() {
        // Vector with 5 bytes of data
        // 0x05 = length 5
        // 0xDE 0xAD 0xBE 0xEF 0x00 = the data
        let mut a = Vec::new();
        extend_vec_from_hex("05deadbeef00", &mut a);
        super::do_test(&a);
    }

    #[test]
    fn test_incomplete_vector() {
        // Length says 10 bytes, but only 2 bytes of data follow
        // This should fail gracefully
        let mut a = Vec::new();
        extend_vec_from_hex("0aabcd", &mut a);
        super::do_test(&a);
    }

    #[test]
    fn test_vector_with_fd_prefix() {
        // Vector with 253+ bytes needs 0xFD prefix
        // 0xFD 0xFD 0x00 = length 253 (in CompactSize format)
        // followed by 253 bytes of data
        let mut a = Vec::new();
        extend_vec_from_hex("fdfd00", &mut a);
        // Add 253 bytes of 0xFF
        for _ in 0..253 {
            a.push(0xFF);
        }
        super::do_test(&a);
    }
}

