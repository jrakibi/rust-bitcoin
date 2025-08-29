// SPDX-License-Identifier: CC0-1.0

//! Sha3-256 from the family of hashes based on the Keccak permutation function.

// The Keccak permutation function is defined by five functions and a state array of N-bits,
// commonly 1600 bits. These 1600 bits are often divided into a 5x5 matrix of `u64` little endian
// numbers, commonly called a "lane." A single Keccak round comprises of a theta, rho, pi, chi, and iota step. Each of these
// steps are easily performed on a lane. Keccakf1600 is a function that performs a number of Keccak rounds, each defined with
// a different round-constant.
//
// Sha3-256 is a hash function that accepts arbitrary data and alters a Keccak state via the
// KeccakF1600 function. Data is chunked into fixed-sizes slices, called padded messages, each with
// a size of "bitrate" or "rate" for short. Each padded message block is XOR'd into parts of the
// state array, followed by a call of Keccakf1600. To pad the final message block, a
// domain-specific identifier is appended to the message (`0x06`), followed by an XOR of the last
// byte with `0x80`.
//
// To read this file, follow the example code: https://keccak.team/keccak_specs_summary.html
// For a detailed specification: https://keccak.team/files/Keccak-reference-3.0.pdf

crate::internal_macros::hash_type_no_default! {
    256,
    false,
    "Output of the SHA3-256 hash function."
}
// 1600 bits are divided into 25, 64-bit "lanes"
const NUM_LANES: usize = 25;
// Let the word size be 64. Let 2^l = 64, then l is 6. In Keccak, the number of rounds is 12 +
// 2l
const NUM_ROUNDS: usize = 24;
// The number of rows or columns
const B: usize = 5;
// The number of bytes "absorbed" into the state-array per message block.
const RATE: usize = 136;
// The number of lanes a message block may be divided into.
const RATE_LANES: usize = 17;

// These create non-linear relations between rounds to avoid timing analysis.
const ROUND_CONSTANTS: [u64; NUM_ROUNDS] = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808A,
    0x8000000080008000,
    0x000000000000808B,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008A,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000A,
    0x000000008000808B,
    0x800000000000008B,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800A,
    0x800000008000000A,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
];

const ROTATION_OFFSETS: [[u32; 5]; 5] = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
];

// A `x` and `y` index into a flattened matrix.
#[inline(always)]
const fn ind(x: usize, y: usize) -> usize {
    x + B * y
}

// A flattened 5x5 matrix of little-endian `u64`.
#[derive(Clone, Default)]
struct KeccakState([u64; NUM_LANES]);

// A row-column labeled output of the current state.
impl ::core::fmt::Debug for KeccakState {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for y in 0..B {
            for x in 0..B {
                writeln!(f, "[{},{}]: {:016x}", x, y, self.lane(x, y).to_le())?;
            }
        }
        Ok(())
    }
}

impl KeccakState {
    const fn new() -> Self {
        Self([0u64; NUM_LANES])
    }

    #[inline(always)]
    fn assign(&mut self, x: usize, y: usize, val: u64) {
        self.0[ind(x, y)] = val;
    }

    #[inline(always)]
    const fn lane(&self, x: usize, y: usize) -> u64 {
        self.0[ind(x, y)]
    }

    #[inline(always)]
    const fn row_xor(&self, x: usize) -> u64 {
        self.0[ind(x, 0)]
            ^ self.0[ind(x, 1)]
            ^ self.0[ind(x, 2)]
            ^ self.0[ind(x, 3)]
            ^ self.0[ind(x, 4)]
    }

    #[inline(always)]
    fn xor_assign(&mut self, x: usize, y: usize, val: u64) {
        self.0[ind(x, y)] ^= val
    }

    #[inline(always)]
    const fn chi(&self, x: usize, y: usize) -> u64 {
        self.lane(x, y) ^ (!self.lane((x + 1) % B, y) & self.lane((x + 2) % B, y))
    }
}

fn keccak_round(state: &mut KeccakState, round_constant: u64) {
    // Theta
    let mut c: [u64; B] = [0; B];
    (0..B).for_each(|x| {
        c[x] = state.row_xor(x);
    });
    let mut d: [u64; B] = [0; B];
    (0..B).for_each(|x| {
        // Avoid an underflow here with a mod trick
        d[x] = c[(x + B - 1) % B] ^ (c[(x + 1) % B].rotate_left(1));
    });
    (0..B).for_each(|x| {
        (0..B).for_each(|y| {
            state.xor_assign(x, y, d[x]);
        });
    });

    // Rho and Pi combined
    let mut b = KeccakState::default();
    (0..B).for_each(|x| {
        (0..B).for_each(|y| {
            let offset = ROTATION_OFFSETS[x][y];
            let val = state.lane(x, y).rotate_left(offset);
            let b_y = ((2 * x) + (3 * y)) % B;
            b.assign(y, b_y, val);
        });
    });

    // Chi
    (0..B).for_each(|x| {
        (0..B).for_each(|y| {
            state.assign(x, y, b.chi(x, y));
        });
    });

    // Iota
    state.xor_assign(0, 0, round_constant);
}

fn keccakf1600(state: &mut KeccakState) {
    for c in ROUND_CONSTANTS {
        keccak_round(state, c);
    }
}

/// Engine to compute the Sha3-256 hash function.
#[derive(Debug, Clone, Default)]
pub struct HashEngine {
    state: KeccakState,
    bytes_hashed: u64,
}

impl HashEngine {
    /// Construct a new Sha3-256 hash engine.
    pub const fn new() -> Self {
        Self { state: KeccakState::new(), bytes_hashed: 0 }
    }

    fn absorb(&mut self, block: [u8; RATE]) {
        for lane in 0..RATE_LANES {
            let x = lane % 5;
            let y = lane / 5;
            let mut pad_block = [0u8; 8];
            pad_block.copy_from_slice(&block[8 * lane..8 * lane + 8]);
            let shuffle = u64::from_le_bytes(pad_block);
            self.state.xor_assign(x, y, shuffle);
        }
    }

    fn input(&mut self, mut inp: &[u8]) {
        while inp.len().ge(&RATE) {
            let mut block = [0u8; RATE];
            block.copy_from_slice(&inp[..RATE]);
            self.bytes_hashed += RATE as u64;
            self.absorb(block);
            keccakf1600(&mut self.state);
            inp = &inp[RATE..];
        }
        let mut final_block = [0u8; RATE];
        final_block[..inp.len()].copy_from_slice(inp);
        self.bytes_hashed += inp.len() as u64;
        final_block[inp.len()] = 0x06;
        final_block[RATE - 1] ^= 0x80;
        self.absorb(final_block);
        keccakf1600(&mut self.state);
    }

    fn finalize(self) -> Hash {
        let mut out = [0u8; 32];
        out[..8].copy_from_slice(&self.state.lane(0, 0).to_le_bytes());
        out[8..16].copy_from_slice(&self.state.lane(1, 0).to_le_bytes());
        out[16..24].copy_from_slice(&self.state.lane(2, 0).to_le_bytes());
        out[24..].copy_from_slice(&self.state.lane(3, 0).to_le_bytes());
        Hash(out)
    }
}

impl crate::HashEngine for HashEngine {
    type Hash = Hash;
    type Bytes = [u8; 32];
    const BLOCK_SIZE: usize = RATE;

    fn input(&mut self, data: &[u8]) {
        self.input(data);
    }

    fn n_bytes_hashed(&self) -> u64 {
        self.bytes_hashed
    }

    fn finalize(self) -> Self::Hash {
        self.finalize()
    }
}

#[cfg(test)]
#[cfg(feature = "alloc")]
#[cfg(feature = "hex")]
mod tests {
    use alloc::string::ToString;

    struct TestCase {
        input: &'static [u8],
        output: &'static str,
    }

    const TESTS: [TestCase; 7] = [
        TestCase {
            // One below the rate block size.
            input: b"The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy do",
            output: "943509b3daac78347687edb7c7c8d3684fd5721e55ee394abdb897dde661b048"
        },
        TestCase {
            // Exactly the rate block size.
            input: b"The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog",
            output: "53b56e8953fb3183aa4886cd79cd8a06b47b9200bfbab4a1b76fd4695efe2885"
        },
        TestCase {
            // One byte over the rate block size.
            input: b"The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog.",
            output: "a7d61692295cf5c9c4cd0c7e3359667239cc1990f0aa9564be33c7b371298adc"
        },
        TestCase {
            input: &[],
            output: "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a",
        },
        // The following are taken from the Cryptographic Algorithm Validation Program by NIST.
        // ref: https://csrc.nist.gov/projects/cryptographic-algorithm-validation-program/secure-hashing
        TestCase {
            input: &[0x5c, 0x56, 0xa6, 0xb1, 0x8c, 0x39, 0xe6, 0x6e, 0x1b, 0x7a, 0x99, 0x3a],
            output: "b697556cb30d6df448ee38b973cb6942559de4c2567b1556240188c55ec0841c",
        },
        TestCase {
            input: &[0x5c, 0x58, 0x9f, 0xc5, 0x4f, 0xef, 0xc4, 0xd6,
                     0xe2, 0x24, 0x9a, 0x36, 0x58, 0x3e, 0x19, 0x92,
                     0xfc, 0x6b, 0x8a, 0x9c, 0x07, 0x0e, 0x8e, 0x00,
                     0xc4, 0x5a, 0x63, 0x9a, 0xf2, 0x20, 0x63, 0xe6,
                     0x6a, 0xe5, 0xcd, 0xb8, 0x02, 0x38, 0xc8, 0x2d,
                     0xb0, 0x43, 0xa5, 0xe1, 0xf3, 0x9f, 0x65, 0x62,
                     0x6e, 0x6d, 0x7b, 0xe5, 0xd6, 0xa2, 0xd3, 0x38,
                     0x0f, 0xa2, 0x12, 0xf8, 0x92, 0x11, 0x20, 0x04,
                     0x12, 0xe5, 0xe4, 0x31, 0x5f, 0xc0, 0x4e, 0x40
            ],
            output: "18deba74e9d93ae7df93c6c316ef201bf5e3a661e68868e14d4f56264f5d858c",
        },
        TestCase {
            input: &[0xf6, 0x5c, 0x3a, 0xa1, 0xd9, 0x98, 0x1a, 0x84,
                     0xe4, 0x9f, 0xc8, 0x6d, 0x93, 0x8f, 0x3f, 0x75,
                     0x6f, 0x60, 0xe3, 0x85, 0x8d, 0x5e, 0x1f, 0x69,
                     0x57, 0xdd, 0x4d, 0x26, 0x8e, 0x28, 0xd6, 0x8e,
                     0x90, 0xba, 0x9a, 0x11, 0xd7, 0xb1, 0x92, 0xd6,
                     0xc3, 0x7f, 0xb3, 0x0b
            ],
            output: "3acbebf8eda9d3c99a6b6b666366c391e8200d55fd33ad8680734def1dc7ae85",
        }
    ];

    #[test]
    fn test() {
        for test in TESTS {
            let mut sha3 = super::HashEngine::new();
            sha3.input(test.input);
            let hash = crate::HashEngine::finalize(sha3);
            assert_eq!(hash.to_string(), test.output);
        }
    }
}
