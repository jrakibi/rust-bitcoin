// SPDX-License-Identifier: CC0-1.0

//! SHA256d implementation (double SHA256).

use internals::slice::SliceExt;

use crate::sha256;

crate::internal_macros::general_hash_type! {
    256,
    true,
    "Output of the SHA256d hash function."
}

impl Hash {
    /// Finalize a hash engine to produce a hash.
    pub fn from_engine(e: HashEngine) -> Self {
        let sha2 = sha256::Hash::from_engine(e.0);
        let sha2d = sha256::Hash::hash(sha2.as_byte_array());

        let mut ret = [0; 32];
        ret.copy_from_slice(sha2d.as_byte_array());
        Self(ret)
    }

    /// optimized version of `sha256d::Hash::hash()` for 64 byte inputs
    #[inline]
    pub fn hash_64(data: &[u8; 64]) -> Self { Self(hash_64(data)) }
}

#[allow(non_snake_case)]
const fn Ch(x: u32, y: u32, z: u32) -> u32 { z ^ (x & (y ^ z)) }
#[allow(non_snake_case)]
const fn Maj(x: u32, y: u32, z: u32) -> u32 { (x & y) | (z & (x | y)) }
#[allow(non_snake_case)]
const fn Sigma0(x: u32) -> u32 { x.rotate_left(30) ^ x.rotate_left(19) ^ x.rotate_left(10) }
#[allow(non_snake_case)]
const fn Sigma1(x: u32) -> u32 { x.rotate_left(26) ^ x.rotate_left(21) ^ x.rotate_left(7) }
const fn sigma0(x: u32) -> u32 { x.rotate_left(25) ^ x.rotate_left(14) ^ (x >> 3) }
const fn sigma1(x: u32) -> u32 { x.rotate_left(15) ^ x.rotate_left(13) ^ (x >> 10) }

const H: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

macro_rules! round(
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr, $h:expr, $k:expr, $w:expr) => {
        let t1 = $h.wrapping_add(Sigma1($e)).wrapping_add(Ch($e, $f, $g)).wrapping_add($k).wrapping_add($w);
        let t2 = Sigma0($a).wrapping_add(Maj($a, $b, $c));
        $d = $d.wrapping_add(t1);
        $h = t1.wrapping_add(t2);
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr, $h:expr, $k:expr, $w:expr, $w1:expr, $w2:expr, $w3:expr) => {
        $w = $w.wrapping_add(sigma1($w1)).wrapping_add($w2).wrapping_add(sigma0($w3));
        round!($a, $b, $c, $d, $e, $f, $g, $h, $k, $w);
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr, $h:expr, $kw:expr) => {
        let t1 = $h.wrapping_add(Sigma1($e)).wrapping_add(Ch($e, $f, $g)).wrapping_add($kw);
        let t2 = Sigma0($a).wrapping_add(Maj($a, $b, $c));
        $d = $d.wrapping_add(t1);
        $h = t1.wrapping_add(t2);
    };
);

/// Computes double SHA256 hash of  64 bytes
pub fn hash_64(data: &[u8; 64]) -> [u8; 32] {
    let mut w = [0u32; 16];
    for (w_val, chunk) in w.iter_mut().zip(data.bitcoin_as_chunks::<4>().0) {
        *w_val = u32::from_be_bytes(*chunk);
    }

    // Transform 1: Process the 64-byte input block
    let mut a = H[0];
    let mut b = H[1];
    let mut c = H[2];
    let mut d = H[3];
    let mut e = H[4];
    let mut f = H[5];
    let mut g = H[6];
    let mut h = H[7];

    // Rounds 0-15
    round!(a, b, c, d, e, f, g, h, 0x428a2f98, w[0]);
    round!(h, a, b, c, d, e, f, g, 0x71374491, w[1]);
    round!(g, h, a, b, c, d, e, f, 0xb5c0fbcf, w[2]);
    round!(f, g, h, a, b, c, d, e, 0xe9b5dba5, w[3]);
    round!(e, f, g, h, a, b, c, d, 0x3956c25b, w[4]);
    round!(d, e, f, g, h, a, b, c, 0x59f111f1, w[5]);
    round!(c, d, e, f, g, h, a, b, 0x923f82a4, w[6]);
    round!(b, c, d, e, f, g, h, a, 0xab1c5ed5, w[7]);
    round!(a, b, c, d, e, f, g, h, 0xd807aa98, w[8]);
    round!(h, a, b, c, d, e, f, g, 0x12835b01, w[9]);
    round!(g, h, a, b, c, d, e, f, 0x243185be, w[10]);
    round!(f, g, h, a, b, c, d, e, 0x550c7dc3, w[11]);
    round!(e, f, g, h, a, b, c, d, 0x72be5d74, w[12]);
    round!(d, e, f, g, h, a, b, c, 0x80deb1fe, w[13]);
    round!(c, d, e, f, g, h, a, b, 0x9bdc06a7, w[14]);
    round!(b, c, d, e, f, g, h, a, 0xc19bf174, w[15]);

    // Rounds 16-31
    round!(a, b, c, d, e, f, g, h, 0xe49b69c1, w[0], w[14], w[9], w[1]);
    round!(h, a, b, c, d, e, f, g, 0xefbe4786, w[1], w[15], w[10], w[2]);
    round!(g, h, a, b, c, d, e, f, 0x0fc19dc6, w[2], w[0], w[11], w[3]);
    round!(f, g, h, a, b, c, d, e, 0x240ca1cc, w[3], w[1], w[12], w[4]);
    round!(e, f, g, h, a, b, c, d, 0x2de92c6f, w[4], w[2], w[13], w[5]);
    round!(d, e, f, g, h, a, b, c, 0x4a7484aa, w[5], w[3], w[14], w[6]);
    round!(c, d, e, f, g, h, a, b, 0x5cb0a9dc, w[6], w[4], w[15], w[7]);
    round!(b, c, d, e, f, g, h, a, 0x76f988da, w[7], w[5], w[0], w[8]);
    round!(a, b, c, d, e, f, g, h, 0x983e5152, w[8], w[6], w[1], w[9]);
    round!(h, a, b, c, d, e, f, g, 0xa831c66d, w[9], w[7], w[2], w[10]);
    round!(g, h, a, b, c, d, e, f, 0xb00327c8, w[10], w[8], w[3], w[11]);
    round!(f, g, h, a, b, c, d, e, 0xbf597fc7, w[11], w[9], w[4], w[12]);
    round!(e, f, g, h, a, b, c, d, 0xc6e00bf3, w[12], w[10], w[5], w[13]);
    round!(d, e, f, g, h, a, b, c, 0xd5a79147, w[13], w[11], w[6], w[14]);
    round!(c, d, e, f, g, h, a, b, 0x06ca6351, w[14], w[12], w[7], w[15]);
    round!(b, c, d, e, f, g, h, a, 0x14292967, w[15], w[13], w[8], w[0]);

    // Rounds 32-47
    round!(a, b, c, d, e, f, g, h, 0x27b70a85, w[0], w[14], w[9], w[1]);
    round!(h, a, b, c, d, e, f, g, 0x2e1b2138, w[1], w[15], w[10], w[2]);
    round!(g, h, a, b, c, d, e, f, 0x4d2c6dfc, w[2], w[0], w[11], w[3]);
    round!(f, g, h, a, b, c, d, e, 0x53380d13, w[3], w[1], w[12], w[4]);
    round!(e, f, g, h, a, b, c, d, 0x650a7354, w[4], w[2], w[13], w[5]);
    round!(d, e, f, g, h, a, b, c, 0x766a0abb, w[5], w[3], w[14], w[6]);
    round!(c, d, e, f, g, h, a, b, 0x81c2c92e, w[6], w[4], w[15], w[7]);
    round!(b, c, d, e, f, g, h, a, 0x92722c85, w[7], w[5], w[0], w[8]);
    round!(a, b, c, d, e, f, g, h, 0xa2bfe8a1, w[8], w[6], w[1], w[9]);
    round!(h, a, b, c, d, e, f, g, 0xa81a664b, w[9], w[7], w[2], w[10]);
    round!(g, h, a, b, c, d, e, f, 0xc24b8b70, w[10], w[8], w[3], w[11]);
    round!(f, g, h, a, b, c, d, e, 0xc76c51a3, w[11], w[9], w[4], w[12]);
    round!(e, f, g, h, a, b, c, d, 0xd192e819, w[12], w[10], w[5], w[13]);
    round!(d, e, f, g, h, a, b, c, 0xd6990624, w[13], w[11], w[6], w[14]);
    round!(c, d, e, f, g, h, a, b, 0xf40e3585, w[14], w[12], w[7], w[15]);
    round!(b, c, d, e, f, g, h, a, 0x106aa070, w[15], w[13], w[8], w[0]);

    // Rounds 48-63
    round!(a, b, c, d, e, f, g, h, 0x19a4c116, w[0], w[14], w[9], w[1]);
    round!(h, a, b, c, d, e, f, g, 0x1e376c08, w[1], w[15], w[10], w[2]);
    round!(g, h, a, b, c, d, e, f, 0x2748774c, w[2], w[0], w[11], w[3]);
    round!(f, g, h, a, b, c, d, e, 0x34b0bcb5, w[3], w[1], w[12], w[4]);
    round!(e, f, g, h, a, b, c, d, 0x391c0cb3, w[4], w[2], w[13], w[5]);
    round!(d, e, f, g, h, a, b, c, 0x4ed8aa4a, w[5], w[3], w[14], w[6]);
    round!(c, d, e, f, g, h, a, b, 0x5b9cca4f, w[6], w[4], w[15], w[7]);
    round!(b, c, d, e, f, g, h, a, 0x682e6ff3, w[7], w[5], w[0], w[8]);
    round!(a, b, c, d, e, f, g, h, 0x748f82ee, w[8], w[6], w[1], w[9]);
    round!(h, a, b, c, d, e, f, g, 0x78a5636f, w[9], w[7], w[2], w[10]);
    round!(g, h, a, b, c, d, e, f, 0x84c87814, w[10], w[8], w[3], w[11]);
    round!(f, g, h, a, b, c, d, e, 0x8cc70208, w[11], w[9], w[4], w[12]);
    round!(e, f, g, h, a, b, c, d, 0x90befffa, w[12], w[10], w[5], w[13]);
    round!(d, e, f, g, h, a, b, c, 0xa4506ceb, w[13], w[11], w[6], w[14]);
    round!(c, d, e, f, g, h, a, b, 0xbef9a3f7, w[14], w[12], w[7], w[15]);
    round!(b, c, d, e, f, g, h, a, 0xc67178f2, w[15], w[13], w[8], w[0]);

    a = a.wrapping_add(H[0]);
    b = b.wrapping_add(H[1]);
    c = c.wrapping_add(H[2]);
    d = d.wrapping_add(H[3]);
    e = e.wrapping_add(H[4]);
    f = f.wrapping_add(H[5]);
    g = g.wrapping_add(H[6]);
    h = h.wrapping_add(H[7]);

    let t0 = a;
    let t1 = b;
    let t2 = c;
    let t3 = d;
    let t4 = e;
    let t5 = f;
    let t6 = g;
    let t7 = h;

    // Transform 2: process padding block
    round!(a, b, c, d, e, f, g, h, 0xc28a2f98u32);
    round!(h, a, b, c, d, e, f, g, 0x71374491u32);
    round!(g, h, a, b, c, d, e, f, 0xb5c0fbcfu32);
    round!(f, g, h, a, b, c, d, e, 0xe9b5dba5u32);
    round!(e, f, g, h, a, b, c, d, 0x3956c25bu32);
    round!(d, e, f, g, h, a, b, c, 0x59f111f1u32);
    round!(c, d, e, f, g, h, a, b, 0x923f82a4u32);
    round!(b, c, d, e, f, g, h, a, 0xab1c5ed5u32);
    round!(a, b, c, d, e, f, g, h, 0xd807aa98u32);
    round!(h, a, b, c, d, e, f, g, 0x12835b01u32);
    round!(g, h, a, b, c, d, e, f, 0x243185beu32);
    round!(f, g, h, a, b, c, d, e, 0x550c7dc3u32);
    round!(e, f, g, h, a, b, c, d, 0x72be5d74u32);
    round!(d, e, f, g, h, a, b, c, 0x80deb1feu32);
    round!(c, d, e, f, g, h, a, b, 0x9bdc06a7u32);
    round!(b, c, d, e, f, g, h, a, 0xc19bf374u32);
    round!(a, b, c, d, e, f, g, h, 0x649b69c1u32);
    round!(h, a, b, c, d, e, f, g, 0xf0fe4786u32);
    round!(g, h, a, b, c, d, e, f, 0x0fe1edc6u32);
    round!(f, g, h, a, b, c, d, e, 0x240cf254u32);
    round!(e, f, g, h, a, b, c, d, 0x4fe9346fu32);
    round!(d, e, f, g, h, a, b, c, 0x6cc984beu32);
    round!(c, d, e, f, g, h, a, b, 0x61b9411eu32);
    round!(b, c, d, e, f, g, h, a, 0x16f988fau32);
    round!(a, b, c, d, e, f, g, h, 0xf2c65152u32);
    round!(h, a, b, c, d, e, f, g, 0xa88e5a6du32);
    round!(g, h, a, b, c, d, e, f, 0xb019fc65u32);
    round!(f, g, h, a, b, c, d, e, 0xb9d99ec7u32);
    round!(e, f, g, h, a, b, c, d, 0x9a1231c3u32);
    round!(d, e, f, g, h, a, b, c, 0xe70eeaa0u32);
    round!(c, d, e, f, g, h, a, b, 0xfdb1232bu32);
    round!(b, c, d, e, f, g, h, a, 0xc7353eb0u32);
    round!(a, b, c, d, e, f, g, h, 0x3069bad5u32);
    round!(h, a, b, c, d, e, f, g, 0xcb976d5fu32);
    round!(g, h, a, b, c, d, e, f, 0x5a0f118fu32);
    round!(f, g, h, a, b, c, d, e, 0xdc1eeefdu32);
    round!(e, f, g, h, a, b, c, d, 0x0a35b689u32);
    round!(d, e, f, g, h, a, b, c, 0xde0b7a04u32);
    round!(c, d, e, f, g, h, a, b, 0x58f4ca9du32);
    round!(b, c, d, e, f, g, h, a, 0xe15d5b16u32);
    round!(a, b, c, d, e, f, g, h, 0x007f3e86u32);
    round!(h, a, b, c, d, e, f, g, 0x37088980u32);
    round!(g, h, a, b, c, d, e, f, 0xa507ea32u32);
    round!(f, g, h, a, b, c, d, e, 0x6fab9537u32);
    round!(e, f, g, h, a, b, c, d, 0x17406110u32);
    round!(d, e, f, g, h, a, b, c, 0x0d8cd6f1u32);
    round!(c, d, e, f, g, h, a, b, 0xcdaa3b6du32);
    round!(b, c, d, e, f, g, h, a, 0xc0bbbe37u32);
    round!(a, b, c, d, e, f, g, h, 0x83613bdau32);
    round!(h, a, b, c, d, e, f, g, 0xdb48a363u32);
    round!(g, h, a, b, c, d, e, f, 0x0b02e931u32);
    round!(f, g, h, a, b, c, d, e, 0x6fd15ca7u32);
    round!(e, f, g, h, a, b, c, d, 0x521afacau32);
    round!(d, e, f, g, h, a, b, c, 0x31338431u32);
    round!(c, d, e, f, g, h, a, b, 0x6ed41a95u32);
    round!(b, c, d, e, f, g, h, a, 0x6d437890u32);
    round!(a, b, c, d, e, f, g, h, 0xc39c91f2u32);
    round!(h, a, b, c, d, e, f, g, 0x9eccabbdu32);
    round!(g, h, a, b, c, d, e, f, 0xb5c9a0e6u32);
    round!(f, g, h, a, b, c, d, e, 0x532fb63cu32);
    round!(e, f, g, h, a, b, c, d, 0xd2c741c6u32);
    round!(d, e, f, g, h, a, b, c, 0x07237ea3u32);
    round!(c, d, e, f, g, h, a, b, 0xa4954b68u32);
    round!(b, c, d, e, f, g, h, a, 0x4c191d76u32);

    a = a.wrapping_add(t0);
    b = b.wrapping_add(t1);
    c = c.wrapping_add(t2);
    d = d.wrapping_add(t3);
    e = e.wrapping_add(t4);
    f = f.wrapping_add(t5);
    g = g.wrapping_add(t6);
    h = h.wrapping_add(t7);

    // Transform 3: SHA256 of the 32 byte result
    w[0] = a;
    w[1] = b;
    w[2] = c;
    w[3] = d;
    w[4] = e;
    w[5] = f;
    w[6] = g;
    w[7] = h;
    w[8] = 0x80000000;
    w[9] = 0;
    w[10] = 0;
    w[11] = 0;
    w[12] = 0;
    w[13] = 0;
    w[14] = 0;
    w[15] = 0x00000100;

    a = H[0];
    b = H[1];
    c = H[2];
    d = H[3];
    e = H[4];
    f = H[5];
    g = H[6];
    h = H[7];

    // Rounds 0-15
    round!(a, b, c, d, e, f, g, h, 0x428a2f98, w[0]);
    round!(h, a, b, c, d, e, f, g, 0x71374491, w[1]);
    round!(g, h, a, b, c, d, e, f, 0xb5c0fbcf, w[2]);
    round!(f, g, h, a, b, c, d, e, 0xe9b5dba5, w[3]);
    round!(e, f, g, h, a, b, c, d, 0x3956c25b, w[4]);
    round!(d, e, f, g, h, a, b, c, 0x59f111f1, w[5]);
    round!(c, d, e, f, g, h, a, b, 0x923f82a4, w[6]);
    round!(b, c, d, e, f, g, h, a, 0xab1c5ed5, w[7]);
    round!(a, b, c, d, e, f, g, h, 0xd807aa98, w[8]);
    round!(h, a, b, c, d, e, f, g, 0x12835b01, w[9]);
    round!(g, h, a, b, c, d, e, f, 0x243185be, w[10]);
    round!(f, g, h, a, b, c, d, e, 0x550c7dc3, w[11]);
    round!(e, f, g, h, a, b, c, d, 0x72be5d74, w[12]);
    round!(d, e, f, g, h, a, b, c, 0x80deb1fe, w[13]);
    round!(c, d, e, f, g, h, a, b, 0x9bdc06a7, w[14]);
    round!(b, c, d, e, f, g, h, a, 0xc19bf174, w[15]);

    // Rounds 16-31
    round!(a, b, c, d, e, f, g, h, 0xe49b69c1, w[0], w[14], w[9], w[1]);
    round!(h, a, b, c, d, e, f, g, 0xefbe4786, w[1], w[15], w[10], w[2]);
    round!(g, h, a, b, c, d, e, f, 0x0fc19dc6, w[2], w[0], w[11], w[3]);
    round!(f, g, h, a, b, c, d, e, 0x240ca1cc, w[3], w[1], w[12], w[4]);
    round!(e, f, g, h, a, b, c, d, 0x2de92c6f, w[4], w[2], w[13], w[5]);
    round!(d, e, f, g, h, a, b, c, 0x4a7484aa, w[5], w[3], w[14], w[6]);
    round!(c, d, e, f, g, h, a, b, 0x5cb0a9dc, w[6], w[4], w[15], w[7]);
    round!(b, c, d, e, f, g, h, a, 0x76f988da, w[7], w[5], w[0], w[8]);
    round!(a, b, c, d, e, f, g, h, 0x983e5152, w[8], w[6], w[1], w[9]);
    round!(h, a, b, c, d, e, f, g, 0xa831c66d, w[9], w[7], w[2], w[10]);
    round!(g, h, a, b, c, d, e, f, 0xb00327c8, w[10], w[8], w[3], w[11]);
    round!(f, g, h, a, b, c, d, e, 0xbf597fc7, w[11], w[9], w[4], w[12]);
    round!(e, f, g, h, a, b, c, d, 0xc6e00bf3, w[12], w[10], w[5], w[13]);
    round!(d, e, f, g, h, a, b, c, 0xd5a79147, w[13], w[11], w[6], w[14]);
    round!(c, d, e, f, g, h, a, b, 0x06ca6351, w[14], w[12], w[7], w[15]);
    round!(b, c, d, e, f, g, h, a, 0x14292967, w[15], w[13], w[8], w[0]);

    // Rounds 32-47
    round!(a, b, c, d, e, f, g, h, 0x27b70a85, w[0], w[14], w[9], w[1]);
    round!(h, a, b, c, d, e, f, g, 0x2e1b2138, w[1], w[15], w[10], w[2]);
    round!(g, h, a, b, c, d, e, f, 0x4d2c6dfc, w[2], w[0], w[11], w[3]);
    round!(f, g, h, a, b, c, d, e, 0x53380d13, w[3], w[1], w[12], w[4]);
    round!(e, f, g, h, a, b, c, d, 0x650a7354, w[4], w[2], w[13], w[5]);
    round!(d, e, f, g, h, a, b, c, 0x766a0abb, w[5], w[3], w[14], w[6]);
    round!(c, d, e, f, g, h, a, b, 0x81c2c92e, w[6], w[4], w[15], w[7]);
    round!(b, c, d, e, f, g, h, a, 0x92722c85, w[7], w[5], w[0], w[8]);
    round!(a, b, c, d, e, f, g, h, 0xa2bfe8a1, w[8], w[6], w[1], w[9]);
    round!(h, a, b, c, d, e, f, g, 0xa81a664b, w[9], w[7], w[2], w[10]);
    round!(g, h, a, b, c, d, e, f, 0xc24b8b70, w[10], w[8], w[3], w[11]);
    round!(f, g, h, a, b, c, d, e, 0xc76c51a3, w[11], w[9], w[4], w[12]);
    round!(e, f, g, h, a, b, c, d, 0xd192e819, w[12], w[10], w[5], w[13]);
    round!(d, e, f, g, h, a, b, c, 0xd6990624, w[13], w[11], w[6], w[14]);
    round!(c, d, e, f, g, h, a, b, 0xf40e3585, w[14], w[12], w[7], w[15]);
    round!(b, c, d, e, f, g, h, a, 0x106aa070, w[15], w[13], w[8], w[0]);

    // Rounds 48-63
    round!(a, b, c, d, e, f, g, h, 0x19a4c116, w[0], w[14], w[9], w[1]);
    round!(h, a, b, c, d, e, f, g, 0x1e376c08, w[1], w[15], w[10], w[2]);
    round!(g, h, a, b, c, d, e, f, 0x2748774c, w[2], w[0], w[11], w[3]);
    round!(f, g, h, a, b, c, d, e, 0x34b0bcb5, w[3], w[1], w[12], w[4]);
    round!(e, f, g, h, a, b, c, d, 0x391c0cb3, w[4], w[2], w[13], w[5]);
    round!(d, e, f, g, h, a, b, c, 0x4ed8aa4a, w[5], w[3], w[14], w[6]);
    round!(c, d, e, f, g, h, a, b, 0x5b9cca4f, w[6], w[4], w[15], w[7]);
    round!(b, c, d, e, f, g, h, a, 0x682e6ff3, w[7], w[5], w[0], w[8]);
    round!(a, b, c, d, e, f, g, h, 0x748f82ee, w[8], w[6], w[1], w[9]);
    round!(h, a, b, c, d, e, f, g, 0x78a5636f, w[9], w[7], w[2], w[10]);
    round!(g, h, a, b, c, d, e, f, 0x84c87814, w[10], w[8], w[3], w[11]);
    round!(f, g, h, a, b, c, d, e, 0x8cc70208, w[11], w[9], w[4], w[12]);
    round!(e, f, g, h, a, b, c, d, 0x90befffa, w[12], w[10], w[5], w[13]);
    round!(d, e, f, g, h, a, b, c, 0xa4506ceb, w[13], w[11], w[6], w[14]);
    round!(c, d, e, f, g, h, a, b, 0xbef9a3f7, w[14], w[12], w[7], w[15]);
    round!(b, c, d, e, f, g, h, a, 0xc67178f2, w[15], w[13], w[8], w[0]);

    let mut output = [0u8; 32];
    for (chunk, val) in output.bitcoin_as_chunks_mut::<4>().0.iter_mut().zip([
        a.wrapping_add(H[0]),
        b.wrapping_add(H[1]),
        c.wrapping_add(H[2]),
        d.wrapping_add(H[3]),
        e.wrapping_add(H[4]),
        f.wrapping_add(H[5]),
        g.wrapping_add(H[6]),
        h.wrapping_add(H[7]),
    ]) {
        *chunk = val.to_be_bytes();
    }

    output
}

/// Engine to compute SHA256d hash function.
#[derive(Debug, Clone)]
pub struct HashEngine(sha256::HashEngine);

impl HashEngine {
    /// Constructs a new SHA256d hash engine.
    pub const fn new() -> Self { Self(sha256::HashEngine::new()) }
}

impl Default for HashEngine {
    fn default() -> Self { Self::new() }
}

impl crate::HashEngine for HashEngine {
    type Hash = Hash;
    type Bytes = [u8; 32];
    const BLOCK_SIZE: usize = 64; // Same as sha256::HashEngine::BLOCK_SIZE;

    fn input(&mut self, data: &[u8]) { self.0.input(data) }
    fn n_bytes_hashed(&self) -> u64 { self.0.n_bytes_hashed() }
    fn finalize(self) -> Self::Hash { Hash::from_engine(self) }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)] // whether this is used depends on features
    use crate::sha256d;

    #[test]
    #[cfg(feature = "alloc")]
    #[cfg(feature = "hex")]
    fn test() {
        use alloc::string::ToString;

        use crate::{sha256, HashEngine};

        #[derive(Clone)]
        struct Test {
            input: &'static str,
            output: [u8; 32],
            output_str: &'static str,
        }

        #[rustfmt::skip]
        let tests = [
            // Test vector copied out of rust-bitcoin
            Test {
                input: "",
                output: [
                    0x5d, 0xf6, 0xe0, 0xe2, 0x76, 0x13, 0x59, 0xd3,
                    0x0a, 0x82, 0x75, 0x05, 0x8e, 0x29, 0x9f, 0xcc,
                    0x03, 0x81, 0x53, 0x45, 0x45, 0xf5, 0x5c, 0xf4,
                    0x3e, 0x41, 0x98, 0x3f, 0x5d, 0x4c, 0x94, 0x56,
                ],
                output_str: "56944c5d3f98413ef45cf54545538103cc9f298e0575820ad3591376e2e0f65d",
            },
        ];

        for test in tests {
            // Hash through high-level API, check hex encoding/decoding
            let hash = sha256d::Hash::hash(test.input.as_bytes());
            assert_eq!(hash, test.output_str.parse::<sha256d::Hash>().expect("parse hex"));
            assert_eq!(hash.as_byte_array(), &test.output);
            assert_eq!(hash.to_string(), test.output_str);

            // Hash through engine, checking that we can input byte by byte
            let mut engine = sha256d::Hash::engine();
            for ch in test.input.as_bytes() {
                engine.input(&[*ch]);
            }
            let manual_hash = sha256d::Hash::from_engine(engine);
            assert_eq!(hash, manual_hash);

            // Hash by computing a sha256 then `hash_again`ing it
            let sha2_hash = sha256::Hash::hash(test.input.as_bytes());
            let sha2d_hash = sha2_hash.hash_again();
            assert_eq!(hash, sha2d_hash);

            assert_eq!(hash.to_byte_array(), test.output);
        }
    }

    #[test]
    #[cfg(feature = "alloc")]
    #[cfg(feature = "hex")]
    fn fmt_roundtrips() {
        use alloc::format;

        let hash = sha256d::Hash::hash(b"some arbitrary bytes");
        let hex = format!("{}", hash);
        let roundtrip = hex.parse::<sha256d::Hash>().expect("failed to parse hex");
        assert_eq!(roundtrip, hash)
    }

    #[test]
    #[cfg(feature = "serde")]
    fn sha256_serde() {
        use serde_test::{assert_tokens, Configure, Token};

        #[rustfmt::skip]
        static HASH_BYTES: [u8; 32] = [
            0xef, 0x53, 0x7f, 0x25, 0xc8, 0x95, 0xbf, 0xa7,
            0x82, 0x52, 0x65, 0x29, 0xa9, 0xb6, 0x3d, 0x97,
            0xaa, 0x63, 0x15, 0x64, 0xd5, 0xd7, 0x89, 0xc2,
            0xb7, 0x65, 0x44, 0x8c, 0x86, 0x35, 0xfb, 0x6c,
        ];

        let hash = sha256d::Hash::from_byte_array(HASH_BYTES);
        assert_tokens(&hash.compact(), &[Token::BorrowedBytes(&HASH_BYTES[..])]);
        assert_tokens(
            &hash.readable(),
            &[Token::Str("6cfb35868c4465b7c289d7d5641563aa973db6a929655282a7bf95c8257f53ef")],
        );
    }
}
