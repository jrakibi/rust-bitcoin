// SPDX-License-Identifier: CC0-1.0

#![allow(clippy::unreadable_literal)]
#![allow(clippy::cast_ptr_alignment)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::many_single_char_names)]

#[cfg(all(target_arch = "aarch64", any(feature = "std", feature = "cpufeatures")))]
use core::arch::aarch64::{
    vaddq_u32, vld1q_u32, vreinterpretq_u32_u8, vreinterpretq_u8_u32, vrev32q_u8, vsha256h2q_u32,
    vsha256hq_u32, vsha256su0q_u32, vsha256su1q_u32, vst1q_u32,
};
#[cfg(all(target_arch = "x86", any(feature = "std", feature = "cpufeatures")))]
use core::arch::x86::{
    __m128i, _mm_add_epi32, _mm_alignr_epi8, _mm_blend_epi16, _mm_loadu_si128, _mm_set_epi64x,
    _mm_sha256msg1_epu32, _mm_sha256msg2_epu32, _mm_sha256rnds2_epu32, _mm_shuffle_epi32,
    _mm_shuffle_epi8, _mm_storeu_si128,
};
#[cfg(all(target_arch = "x86_64", any(feature = "std", feature = "cpufeatures")))]
use core::arch::x86_64::{
    __m128i, _mm_add_epi32, _mm_alignr_epi8, _mm_blend_epi16, _mm_loadu_si128, _mm_set_epi64x,
    _mm_sha256msg1_epu32, _mm_sha256msg2_epu32, _mm_sha256rnds2_epu32, _mm_shuffle_epi32,
    _mm_shuffle_epi8, _mm_storeu_si128,
};

use internals::slice::SliceExt;

use super::{HashEngine, Midstate, BLOCK_SIZE};
use crate::sha256d;

#[cfg(all(feature = "cpufeatures", target_arch = "aarch64"))]
// cpufeatures crate internally uses `u8::max_value()` which will be deprecated.
// See: https://docs.rs/cpufeatures/0.2.17/src/cpufeatures/lib.rs.html#161
#[allow(deprecated_in_future)]
mod cpuid_sha256_aarch64 {
    cpufeatures::new!(inner, "sha2");
    pub fn get() -> bool { inner::get() }
}
#[cfg(all(feature = "cpufeatures", any(target_arch = "x86", target_arch = "x86_64")))]
// cpufeatures crate internally uses `u8::max_value()` which will be deprecated.
// See: https://docs.rs/cpufeatures/0.2.17/src/cpufeatures/lib.rs.html#161
#[allow(deprecated_in_future)]
mod cpuid_sha256_x86 {
    cpufeatures::new!(inner, "sha", "sse2", "ssse3", "sse4.1");
    pub fn get() -> bool { inner::get() }
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

#[cfg(feature = "small-hash")]
#[macro_use]
mod small_hash {
    use super::{sigma0, sigma1, Ch, Maj, Sigma0, Sigma1};

    #[rustfmt::skip]
    #[allow(clippy::too_many_arguments)]
    pub(super) const fn round(a: u32, b: u32, c: u32, d: u32, e: u32,
                              f: u32, g: u32, h: u32, k: u32, w: u32) -> (u32, u32) {
        let t1 =
            h.wrapping_add(Sigma1(e)).wrapping_add(Ch(e, f, g)).wrapping_add(k).wrapping_add(w);
        let t2 = Sigma0(a).wrapping_add(Maj(a, b, c));
        (d.wrapping_add(t1), t1.wrapping_add(t2))
    }
    #[rustfmt::skip]
    #[allow(clippy::too_many_arguments)]
    pub(super) const fn later_round(a: u32, b: u32, c: u32, d: u32, e: u32,
                                    f: u32, g: u32, h: u32, k: u32, w: u32,
                                    w1: u32, w2: u32, w3: u32,
    ) -> (u32, u32, u32) {
        let w = w.wrapping_add(sigma1(w1)).wrapping_add(w2).wrapping_add(sigma0(w3));
        let (d, h) = round(a, b, c, d, e, f, g, h, k, w);
        (d, h, w)
    }

    macro_rules! round(
        // first round
        ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr, $h:expr, $k:expr, $w:expr) => (
            let updates = small_hash::round($a, $b, $c, $d, $e, $f, $g, $h, $k, $w);
            $d = updates.0;
            $h = updates.1;
        );
        // later rounds we reassign $w before doing the first-round computation
        ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr, $h:expr, $k:expr, $w:expr, $w1:expr, $w2:expr, $w3:expr) => (
            let updates = small_hash::later_round($a, $b, $c, $d, $e, $f, $g, $h, $k, $w, $w1, $w2, $w3);
            $d = updates.0;
            $h = updates.1;
            $w = updates.2;
        )
    );
}

#[cfg(not(feature = "small-hash"))]
#[macro_use]
mod fast_hash {
    macro_rules! round(
        // first round
        ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr, $h:expr, $k:expr, $w:expr) => (
            let t1 = $h.wrapping_add(Sigma1($e)).wrapping_add(Ch($e, $f, $g)).wrapping_add($k).wrapping_add($w);
            let t2 = Sigma0($a).wrapping_add(Maj($a, $b, $c));
            $d = $d.wrapping_add(t1);
            $h = t1.wrapping_add(t2);
        );
        // later rounds we reassign $w before doing the first-round computation
        ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr, $h:expr, $k:expr, $w:expr, $w1:expr, $w2:expr, $w3:expr) => (
            $w = $w.wrapping_add(sigma1($w1)).wrapping_add($w2).wrapping_add(sigma0($w3));
            round!($a, $b, $c, $d, $e, $f, $g, $h, $k, $w);
        )
    );
}

impl Midstate {
    #[allow(clippy::identity_op)] // more readable
    const fn read_u32(bytes: &[u8], index: usize) -> u32 {
        ((bytes[index + 0] as u32) << 24)
            | ((bytes[index + 1] as u32) << 16)
            | ((bytes[index + 2] as u32) << 8)
            | ((bytes[index + 3] as u32) << 0)
    }

    const fn copy_w(bytes: &[u8], index: usize) -> [u32; 16] {
        let mut w = [0u32; 16];
        let mut i = 0;
        while i < 16 {
            w[i] = Self::read_u32(bytes, index + i * 4);
            i += 1;
        }
        w
    }

    pub(super) const fn compute_midstate_unoptimized(bytes: &[u8], finalize: bool) -> Self {
        let mut state = [
            0x6a09e667u32,
            0xbb67ae85,
            0x3c6ef372,
            0xa54ff53a,
            0x510e527f,
            0x9b05688c,
            0x1f83d9ab,
            0x5be0cd19,
        ];

        let num_chunks = (bytes.len() + 9).div_ceil(64);
        let mut chunk = 0;
        #[allow(clippy::precedence)]
        while chunk < num_chunks {
            if !finalize && chunk + 1 == num_chunks {
                break;
            }
            let mut w = if chunk * 64 + 64 <= bytes.len() {
                Self::copy_w(bytes, chunk * 64)
            } else {
                let mut buf = [0; 64];
                let mut i = 0;
                let offset = chunk * 64;
                while offset + i < bytes.len() {
                    buf[i] = bytes[offset + i];
                    i += 1;
                }
                if (bytes.len() % 64 <= 64 - 9) || (chunk + 2 == num_chunks) {
                    buf[i] = 0x80;
                }
                #[allow(clippy::identity_op)] // more readable
                #[allow(clippy::erasing_op)]
                if chunk + 1 == num_chunks {
                    let bit_len = bytes.len() as u64 * 8;
                    buf[64 - 8] = ((bit_len >> 8 * 7) & 0xFF) as u8;
                    buf[64 - 7] = ((bit_len >> 8 * 6) & 0xFF) as u8;
                    buf[64 - 6] = ((bit_len >> 8 * 5) & 0xFF) as u8;
                    buf[64 - 5] = ((bit_len >> 8 * 4) & 0xFF) as u8;
                    buf[64 - 4] = ((bit_len >> 8 * 3) & 0xFF) as u8;
                    buf[64 - 3] = ((bit_len >> 8 * 2) & 0xFF) as u8;
                    buf[64 - 2] = ((bit_len >> 8 * 1) & 0xFF) as u8;
                    buf[64 - 1] = ((bit_len >> 8 * 0) & 0xFF) as u8;
                }
                Self::copy_w(&buf, 0)
            };
            chunk += 1;

            let mut a = state[0];
            let mut b = state[1];
            let mut c = state[2];
            let mut d = state[3];
            let mut e = state[4];
            let mut f = state[5];
            let mut g = state[6];
            let mut h = state[7];

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

            state[0] = state[0].wrapping_add(a);
            state[1] = state[1].wrapping_add(b);
            state[2] = state[2].wrapping_add(c);
            state[3] = state[3].wrapping_add(d);
            state[4] = state[4].wrapping_add(e);
            state[5] = state[5].wrapping_add(f);
            state[6] = state[6].wrapping_add(g);
            state[7] = state[7].wrapping_add(h);
        }
        let mut output = [0u8; 32];
        let mut i = 0;
        #[allow(clippy::identity_op)] // more readable
        while i < 8 {
            output[i * 4 + 0] = (state[i + 0] >> 24) as u8;
            output[i * 4 + 1] = (state[i + 0] >> 16) as u8;
            output[i * 4 + 2] = (state[i + 0] >> 8) as u8;
            output[i * 4 + 3] = (state[i + 0] >> 0) as u8;
            i += 1;
        }
        Self { bytes: output, bytes_hashed: bytes.len() as u64 }
    }
}

impl HashEngine {
    pub(super) fn process_blocks(state: &mut [u32; 8], blocks: &[u8]) {
        #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if std::is_x86_feature_detected!("sse4.1")
                && std::is_x86_feature_detected!("sha")
                && std::is_x86_feature_detected!("sse2")
                && std::is_x86_feature_detected!("ssse3")
            {
                for block in blocks.chunks_exact(BLOCK_SIZE) {
                    unsafe { Self::process_block_simd_x86_intrinsics(state, block) };
                }
                return;
            }
        }

        #[cfg(all(feature = "cpufeatures", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if cpuid_sha256_x86::get() {
                for block in blocks.chunks_exact(BLOCK_SIZE) {
                    unsafe { Self::process_block_simd_x86_intrinsics(state, block) };
                }
                return;
            }
        }

        #[cfg(all(feature = "std", target_arch = "aarch64"))]
        {
            if std::arch::is_aarch64_feature_detected!("sha2") {
                for block in blocks.chunks_exact(BLOCK_SIZE) {
                    unsafe { Self::process_block_simd_arm_intrinsics(state, block) };
                }
                return;
            }
        }

        #[cfg(all(feature = "cpufeatures", target_arch = "aarch64"))]
        {
            if cpuid_sha256_aarch64::get() {
                for block in blocks.chunks_exact(BLOCK_SIZE) {
                    unsafe { Self::process_block_simd_arm_intrinsics(state, block) };
                }
                return;
            }
        }

        // fallback implementation without using any intrinsics
        Self::software_process_block(state, blocks);
    }

    pub(crate) fn sha256d_64(outputs: &mut [[u8; 32]], inputs: &[[u8; 64]]) {
        assert_eq!(outputs.len(), inputs.len());
        let mut i = 0;
        let count = inputs.len();

        // x86_64 dispatch: SHA-NI > AVX2 > SSE4.1
        // When SHA-NI is available, use 2-way hardware SHA (faster per-block than
        // software 4/8-way). Otherwise fall through to AVX2 8-way, then SSE4.1 4-way.
        #[cfg(all(feature = "std", target_arch = "x86_64"))]
        {
            let has_sha_ni = std::is_x86_feature_detected!("sha")
                && std::is_x86_feature_detected!("sse4.1")
                && std::is_x86_feature_detected!("sse2")
                && std::is_x86_feature_detected!("ssse3");

            if has_sha_ni {
                while count - i >= 2 {
                    let out = <&mut [[u8; 32]; 2]>::try_from(&mut outputs[i..i + 2]).unwrap();
                    let inp = <&[[u8; 64]; 2]>::try_from(&inputs[i..i + 2]).unwrap();
                    unsafe { Self::sha256d_64_x86_shani_2way(out, inp) };
                    i += 2;
                }
            } else {
                if std::is_x86_feature_detected!("avx2") {
                    while count - i >= 8 {
                        let out =
                            <&mut [[u8; 32]; 8]>::try_from(&mut outputs[i..i + 8]).unwrap();
                        let inp = <&[[u8; 64]; 8]>::try_from(&inputs[i..i + 8]).unwrap();
                        unsafe { Self::sha256d_64_avx2_8way(out, inp) };
                        i += 8;
                    }
                }
                if std::is_x86_feature_detected!("sse4.1") {
                    while count - i >= 4 {
                        let out =
                            <&mut [[u8; 32]; 4]>::try_from(&mut outputs[i..i + 4]).unwrap();
                        let inp = <&[[u8; 64]; 4]>::try_from(&inputs[i..i + 4]).unwrap();
                        unsafe { Self::sha256d_64_sse41_4way(out, inp) };
                        i += 4;
                    }
                }
            }
        }

        #[cfg(all(feature = "cpufeatures", target_arch = "x86_64"))]
        {
            if cpuid_sha256_x86::get() {
                while count - i >= 2 {
                    let out = <&mut [[u8; 32]; 2]>::try_from(&mut outputs[i..i + 2]).unwrap();
                    let inp = <&[[u8; 64]; 2]>::try_from(&inputs[i..i + 2]).unwrap();
                    unsafe { Self::sha256d_64_x86_shani_2way(out, inp) };
                    i += 2;
                }
            }
            // TODO: cpufeatures dispatch for AVX2/SSE4.1
        }

        // 2-way ARM SHA2
        #[cfg(all(feature = "std", target_arch = "aarch64"))]
        {
            if std::arch::is_aarch64_feature_detected!("sha2") {
                while count - i >= 2 {
                    let out = <&mut [[u8; 32]; 2]>::try_from(&mut outputs[i..i + 2]).unwrap();
                    let inp = <&[[u8; 64]; 2]>::try_from(&inputs[i..i + 2]).unwrap();
                    unsafe { Self::sha256d_64_arm_2way(out, inp) };
                    i += 2;
                }
            }
        }

        #[cfg(all(feature = "cpufeatures", target_arch = "aarch64"))]
        {
            if cpuid_sha256_aarch64::get() {
                while count - i >= 2 {
                    let out = <&mut [[u8; 32]; 2]>::try_from(&mut outputs[i..i + 2]).unwrap();
                    let inp = <&[[u8; 64]; 2]>::try_from(&inputs[i..i + 2]).unwrap();
                    unsafe { Self::sha256d_64_arm_2way(out, inp) };
                    i += 2;
                }
            }
        }

        // fallback
        while i < count {
            outputs[i] = sha256d::hash(&inputs[i]).to_byte_array();
            i += 1;
        }
    }

    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        any(feature = "std", feature = "cpufeatures")
    ))]
    #[target_feature(enable = "sha,sse2,ssse3,sse4.1")]
    unsafe fn process_block_simd_x86_intrinsics(state: &mut [u32; 8], block: &[u8]) {
        // Code translated and based on from
        // https://github.com/noloader/SHA-Intrinsics/blob/4899efc81d1af159c1fd955936c673139f35aea9/sha256-x86.c

        /* sha256-x86.c - Intel SHA extensions using C intrinsics  */
        /*   Written and place in public domain by Jeffrey Walton  */
        /*   Based on code from Intel, and by Sean Gulley for      */
        /*   the miTLS project.                                    */

        // Variable names are also kept the same as in the original C code for easier comparison.
        let (mut state0, mut state1);
        let (mut msg, mut tmp);

        let (mut msg0, mut msg1, mut msg2, mut msg3);

        let (abef_save, cdgh_save);

        #[allow(non_snake_case)]
        let MASK: __m128i =
            _mm_set_epi64x(0x0c0d_0e0f_0809_0a0bu64 as i64, 0x0405_0607_0001_0203u64 as i64);

        let block_offset = 0;

        // Load initial values
        // CAST SAFETY: loadu_si128 documentation states that mem_addr does not
        // need to be aligned on any particular boundary.
        tmp = _mm_loadu_si128(state.as_ptr().add(0).cast::<__m128i>());
        state1 = _mm_loadu_si128(state.as_ptr().add(4).cast::<__m128i>());

        tmp = _mm_shuffle_epi32(tmp, 0xB1); // CDAB
        state1 = _mm_shuffle_epi32(state1, 0x1B); // EFGH
        state0 = _mm_alignr_epi8(tmp, state1, 8); // ABEF
        state1 = _mm_blend_epi16(state1, tmp, 0xF0); // CDGH

        // Process a single block
        {
            // Save current state
            abef_save = state0;
            cdgh_save = state1;

            // Rounds 0-3
            msg = _mm_loadu_si128(block.as_ptr().add(block_offset).cast::<__m128i>());
            msg0 = _mm_shuffle_epi8(msg, MASK);
            msg = _mm_add_epi32(
                msg0,
                _mm_set_epi64x(0xE9B5DBA5B5C0FBCFu64 as i64, 0x71374491428A2F98u64 as i64),
            );
            state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
            msg = _mm_shuffle_epi32(msg, 0x0E);
            state0 = _mm_sha256rnds2_epu32(state0, state1, msg);

            // Rounds 4-7
            msg1 = _mm_loadu_si128(block.as_ptr().add(block_offset + 16).cast::<__m128i>());
            msg1 = _mm_shuffle_epi8(msg1, MASK);
            msg = _mm_add_epi32(
                msg1,
                _mm_set_epi64x(0xAB1C5ED5923F82A4u64 as i64, 0x59F111F13956C25Bu64 as i64),
            );
            state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
            msg = _mm_shuffle_epi32(msg, 0x0E);
            state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
            msg0 = _mm_sha256msg1_epu32(msg0, msg1);

            // Rounds 8-11
            msg2 = _mm_loadu_si128(block.as_ptr().add(block_offset + 32).cast::<__m128i>());
            msg2 = _mm_shuffle_epi8(msg2, MASK);
            msg = _mm_add_epi32(
                msg2,
                _mm_set_epi64x(0x550C7DC3243185BEu64 as i64, 0x12835B01D807AA98u64 as i64),
            );
            state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
            msg = _mm_shuffle_epi32(msg, 0x0E);
            state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
            msg1 = _mm_sha256msg1_epu32(msg1, msg2);

            // Rounds 12-15
            msg3 = _mm_loadu_si128(block.as_ptr().add(block_offset + 48).cast::<__m128i>());
            msg3 = _mm_shuffle_epi8(msg3, MASK);
            msg = _mm_add_epi32(
                msg3,
                _mm_set_epi64x(0xC19BF1749BDC06A7u64 as i64, 0x80DEB1FE72BE5D74u64 as i64),
            );
            state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
            tmp = _mm_alignr_epi8(msg3, msg2, 4);
            msg0 = _mm_add_epi32(msg0, tmp);
            msg0 = _mm_sha256msg2_epu32(msg0, msg3);
            msg = _mm_shuffle_epi32(msg, 0x0E);
            state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
            msg2 = _mm_sha256msg1_epu32(msg2, msg3);

            // Rounds 16-19
            msg = _mm_add_epi32(
                msg0,
                _mm_set_epi64x(0x240CA1CC0FC19DC6u64 as i64, 0xEFBE4786E49B69C1u64 as i64),
            );
            state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
            tmp = _mm_alignr_epi8(msg0, msg3, 4);
            msg1 = _mm_add_epi32(msg1, tmp);
            msg1 = _mm_sha256msg2_epu32(msg1, msg0);
            msg = _mm_shuffle_epi32(msg, 0x0E);
            state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
            msg3 = _mm_sha256msg1_epu32(msg3, msg0);

            // Rounds 20-23
            msg = _mm_add_epi32(
                msg1,
                _mm_set_epi64x(0x76F988DA5CB0A9DCu64 as i64, 0x4A7484AA2DE92C6Fu64 as i64),
            );
            state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
            tmp = _mm_alignr_epi8(msg1, msg0, 4);
            msg2 = _mm_add_epi32(msg2, tmp);
            msg2 = _mm_sha256msg2_epu32(msg2, msg1);
            msg = _mm_shuffle_epi32(msg, 0x0E);
            state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
            msg0 = _mm_sha256msg1_epu32(msg0, msg1);

            // Rounds 24-27
            msg = _mm_add_epi32(
                msg2,
                _mm_set_epi64x(0xBF597FC7B00327C8u64 as i64, 0xA831C66D983E5152u64 as i64),
            );
            state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
            tmp = _mm_alignr_epi8(msg2, msg1, 4);
            msg3 = _mm_add_epi32(msg3, tmp);
            msg3 = _mm_sha256msg2_epu32(msg3, msg2);
            msg = _mm_shuffle_epi32(msg, 0x0E);
            state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
            msg1 = _mm_sha256msg1_epu32(msg1, msg2);

            // Rounds 28-31
            msg = _mm_add_epi32(
                msg3,
                _mm_set_epi64x(0x1429296706CA6351u64 as i64, 0xD5A79147C6E00BF3u64 as i64),
            );
            state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
            tmp = _mm_alignr_epi8(msg3, msg2, 4);
            msg0 = _mm_add_epi32(msg0, tmp);
            msg0 = _mm_sha256msg2_epu32(msg0, msg3);
            msg = _mm_shuffle_epi32(msg, 0x0E);
            state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
            msg2 = _mm_sha256msg1_epu32(msg2, msg3);

            // Rounds 32-35
            msg = _mm_add_epi32(
                msg0,
                _mm_set_epi64x(0x53380D134D2C6DFCu64 as i64, 0x2E1B213827B70A85u64 as i64),
            );
            state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
            tmp = _mm_alignr_epi8(msg0, msg3, 4);
            msg1 = _mm_add_epi32(msg1, tmp);
            msg1 = _mm_sha256msg2_epu32(msg1, msg0);
            msg = _mm_shuffle_epi32(msg, 0x0E);
            state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
            msg3 = _mm_sha256msg1_epu32(msg3, msg0);

            // Rounds 36-39
            msg = _mm_add_epi32(
                msg1,
                _mm_set_epi64x(0x92722C8581C2C92Eu64 as i64, 0x766A0ABB650A7354u64 as i64),
            );
            state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
            tmp = _mm_alignr_epi8(msg1, msg0, 4);
            msg2 = _mm_add_epi32(msg2, tmp);
            msg2 = _mm_sha256msg2_epu32(msg2, msg1);
            msg = _mm_shuffle_epi32(msg, 0x0E);
            state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
            msg0 = _mm_sha256msg1_epu32(msg0, msg1);

            // Rounds 40-43
            msg = _mm_add_epi32(
                msg2,
                _mm_set_epi64x(0xC76C51A3C24B8B70u64 as i64, 0xA81A664BA2BFE8A1u64 as i64),
            );
            state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
            tmp = _mm_alignr_epi8(msg2, msg1, 4);
            msg3 = _mm_add_epi32(msg3, tmp);
            msg3 = _mm_sha256msg2_epu32(msg3, msg2);
            msg = _mm_shuffle_epi32(msg, 0x0E);
            state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
            msg1 = _mm_sha256msg1_epu32(msg1, msg2);

            // Rounds 44-47
            msg = _mm_add_epi32(
                msg3,
                _mm_set_epi64x(0x106AA070F40E3585u64 as i64, 0xD6990624D192E819u64 as i64),
            );
            state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
            tmp = _mm_alignr_epi8(msg3, msg2, 4);
            msg0 = _mm_add_epi32(msg0, tmp);
            msg0 = _mm_sha256msg2_epu32(msg0, msg3);
            msg = _mm_shuffle_epi32(msg, 0x0E);
            state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
            msg2 = _mm_sha256msg1_epu32(msg2, msg3);

            // Rounds 48-51
            msg = _mm_add_epi32(
                msg0,
                _mm_set_epi64x(0x34B0BCB52748774Cu64 as i64, 0x1E376C0819A4C116u64 as i64),
            );
            state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
            tmp = _mm_alignr_epi8(msg0, msg3, 4);
            msg1 = _mm_add_epi32(msg1, tmp);
            msg1 = _mm_sha256msg2_epu32(msg1, msg0);
            msg = _mm_shuffle_epi32(msg, 0x0E);
            state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
            msg3 = _mm_sha256msg1_epu32(msg3, msg0);

            // Rounds 52-55
            msg = _mm_add_epi32(
                msg1,
                _mm_set_epi64x(0x682E6FF35B9CCA4Fu64 as i64, 0x4ED8AA4A391C0CB3u64 as i64),
            );
            state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
            tmp = _mm_alignr_epi8(msg1, msg0, 4);
            msg2 = _mm_add_epi32(msg2, tmp);
            msg2 = _mm_sha256msg2_epu32(msg2, msg1);
            msg = _mm_shuffle_epi32(msg, 0x0E);
            state0 = _mm_sha256rnds2_epu32(state0, state1, msg);

            // Rounds 56-59
            msg = _mm_add_epi32(
                msg2,
                _mm_set_epi64x(0x8CC7020884C87814u64 as i64, 0x78A5636F748F82EEu64 as i64),
            );
            state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
            tmp = _mm_alignr_epi8(msg2, msg1, 4);
            msg3 = _mm_add_epi32(msg3, tmp);
            msg3 = _mm_sha256msg2_epu32(msg3, msg2);
            msg = _mm_shuffle_epi32(msg, 0x0E);
            state0 = _mm_sha256rnds2_epu32(state0, state1, msg);

            // Rounds 60-63
            msg = _mm_add_epi32(
                msg3,
                _mm_set_epi64x(0xC67178F2BEF9A3F7u64 as i64, 0xA4506CEB90BEFFFAu64 as i64),
            );
            state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
            msg = _mm_shuffle_epi32(msg, 0x0E);
            state0 = _mm_sha256rnds2_epu32(state0, state1, msg);

            // Combine state
            state0 = _mm_add_epi32(state0, abef_save);
            state1 = _mm_add_epi32(state1, cdgh_save);
        }

        tmp = _mm_shuffle_epi32(state0, 0x1B); // FEBA
        state1 = _mm_shuffle_epi32(state1, 0xB1); // DCHG
        state0 = _mm_blend_epi16(tmp, state1, 0xF0); // DCBA
        state1 = _mm_alignr_epi8(state1, tmp, 8); // ABEF

        // Save state
        // CAST SAFETY: storeu_si128 documentation states that mem_addr does not
        // need to be aligned on any particular boundary.
        _mm_storeu_si128(state.as_mut_ptr().add(0).cast::<__m128i>(), state0);
        _mm_storeu_si128(state.as_mut_ptr().add(4).cast::<__m128i>(), state1);
    }

    #[cfg(all(target_arch = "aarch64", any(feature = "std", feature = "cpufeatures")))]
    #[target_feature(enable = "sha2")]
    unsafe fn process_block_simd_arm_intrinsics(state: &mut [u32; 8], block: &[u8]) {
        // Code translated and based on from
        // https://github.com/noloader/SHA-Intrinsics/blob/4e754bec921a9f281b69bd681ca0065763aa911c/sha256-arm.c

        /* sha256-arm.c - ARMv8 SHA extensions using C intrinsics     */
        /*   Written and placed in public domain by Jeffrey Walton    */
        /*   Based on code from ARM, and by Johannes Schneiders, Skip */
        /*   Hovsmith and Barry O'Rourke for the mbedTLS project.     */

        // SHA256 round constants
        #[rustfmt::skip]
        const K: [u32; 64] = [
            0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5,
            0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
            0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,
            0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
            0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC,
            0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
            0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7,
            0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
            0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13,
            0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
            0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3,
            0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
            0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5,
            0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
            0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,
            0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
        ];

        let (mut state0, mut state1);
        let (abcd_save, efgh_save);

        let (mut msg0, mut msg1, mut msg2, mut msg3);
        let (mut tmp0, mut tmp1, mut tmp2);

        // Load state
        state0 = vld1q_u32(state.as_ptr().add(0));
        state1 = vld1q_u32(state.as_ptr().add(4));

        // Save state
        abcd_save = state0;
        efgh_save = state1;

        // Load message
        msg0 = vld1q_u32(block.as_ptr().add(0).cast::<u32>());
        msg1 = vld1q_u32(block.as_ptr().add(16).cast::<u32>());
        msg2 = vld1q_u32(block.as_ptr().add(32).cast::<u32>());
        msg3 = vld1q_u32(block.as_ptr().add(48).cast::<u32>());

        // Reverse for little endian
        msg0 = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(msg0)));
        msg1 = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(msg1)));
        msg2 = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(msg2)));
        msg3 = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(msg3)));

        tmp0 = vaddq_u32(msg0, vld1q_u32(K.as_ptr().add(0x00)));

        // Rounds 0-3
        msg0 = vsha256su0q_u32(msg0, msg1);
        tmp2 = state0;
        tmp1 = vaddq_u32(msg1, vld1q_u32(K.as_ptr().add(0x04)));
        state0 = vsha256hq_u32(state0, state1, tmp0);
        state1 = vsha256h2q_u32(state1, tmp2, tmp0);
        msg0 = vsha256su1q_u32(msg0, msg2, msg3);

        // Rounds 4-7
        msg1 = vsha256su0q_u32(msg1, msg2);
        tmp2 = state0;
        tmp0 = vaddq_u32(msg2, vld1q_u32(K.as_ptr().add(0x08)));
        state0 = vsha256hq_u32(state0, state1, tmp1);
        state1 = vsha256h2q_u32(state1, tmp2, tmp1);
        msg1 = vsha256su1q_u32(msg1, msg3, msg0);

        // Rounds 8-11
        msg2 = vsha256su0q_u32(msg2, msg3);
        tmp2 = state0;
        tmp1 = vaddq_u32(msg3, vld1q_u32(K.as_ptr().add(0x0c)));
        state0 = vsha256hq_u32(state0, state1, tmp0);
        state1 = vsha256h2q_u32(state1, tmp2, tmp0);
        msg2 = vsha256su1q_u32(msg2, msg0, msg1);

        // Rounds 12-15
        msg3 = vsha256su0q_u32(msg3, msg0);
        tmp2 = state0;
        tmp0 = vaddq_u32(msg0, vld1q_u32(K.as_ptr().add(0x10)));
        state0 = vsha256hq_u32(state0, state1, tmp1);
        state1 = vsha256h2q_u32(state1, tmp2, tmp1);
        msg3 = vsha256su1q_u32(msg3, msg1, msg2);

        // Rounds 16-19
        msg0 = vsha256su0q_u32(msg0, msg1);
        tmp2 = state0;
        tmp1 = vaddq_u32(msg1, vld1q_u32(K.as_ptr().add(0x14)));
        state0 = vsha256hq_u32(state0, state1, tmp0);
        state1 = vsha256h2q_u32(state1, tmp2, tmp0);
        msg0 = vsha256su1q_u32(msg0, msg2, msg3);

        // Rounds 20-23
        msg1 = vsha256su0q_u32(msg1, msg2);
        tmp2 = state0;
        tmp0 = vaddq_u32(msg2, vld1q_u32(K.as_ptr().add(0x18)));
        state0 = vsha256hq_u32(state0, state1, tmp1);
        state1 = vsha256h2q_u32(state1, tmp2, tmp1);
        msg1 = vsha256su1q_u32(msg1, msg3, msg0);

        // Rounds 24-27
        msg2 = vsha256su0q_u32(msg2, msg3);
        tmp2 = state0;
        tmp1 = vaddq_u32(msg3, vld1q_u32(K.as_ptr().add(0x1c)));
        state0 = vsha256hq_u32(state0, state1, tmp0);
        state1 = vsha256h2q_u32(state1, tmp2, tmp0);
        msg2 = vsha256su1q_u32(msg2, msg0, msg1);

        // Rounds 28-31
        msg3 = vsha256su0q_u32(msg3, msg0);
        tmp2 = state0;
        tmp0 = vaddq_u32(msg0, vld1q_u32(K.as_ptr().add(0x20)));
        state0 = vsha256hq_u32(state0, state1, tmp1);
        state1 = vsha256h2q_u32(state1, tmp2, tmp1);
        msg3 = vsha256su1q_u32(msg3, msg1, msg2);

        // Rounds 32-35
        msg0 = vsha256su0q_u32(msg0, msg1);
        tmp2 = state0;
        tmp1 = vaddq_u32(msg1, vld1q_u32(K.as_ptr().add(0x24)));
        state0 = vsha256hq_u32(state0, state1, tmp0);
        state1 = vsha256h2q_u32(state1, tmp2, tmp0);
        msg0 = vsha256su1q_u32(msg0, msg2, msg3);

        // Rounds 36-39
        msg1 = vsha256su0q_u32(msg1, msg2);
        tmp2 = state0;
        tmp0 = vaddq_u32(msg2, vld1q_u32(K.as_ptr().add(0x28)));
        state0 = vsha256hq_u32(state0, state1, tmp1);
        state1 = vsha256h2q_u32(state1, tmp2, tmp1);
        msg1 = vsha256su1q_u32(msg1, msg3, msg0);

        // Rounds 40-43
        msg2 = vsha256su0q_u32(msg2, msg3);
        tmp2 = state0;
        tmp1 = vaddq_u32(msg3, vld1q_u32(K.as_ptr().add(0x2c)));
        state0 = vsha256hq_u32(state0, state1, tmp0);
        state1 = vsha256h2q_u32(state1, tmp2, tmp0);
        msg2 = vsha256su1q_u32(msg2, msg0, msg1);

        // Rounds 44-47
        msg3 = vsha256su0q_u32(msg3, msg0);
        tmp2 = state0;
        tmp0 = vaddq_u32(msg0, vld1q_u32(K.as_ptr().add(0x30)));
        state0 = vsha256hq_u32(state0, state1, tmp1);
        state1 = vsha256h2q_u32(state1, tmp2, tmp1);
        msg3 = vsha256su1q_u32(msg3, msg1, msg2);

        // Rounds 48-51
        tmp2 = state0;
        tmp1 = vaddq_u32(msg1, vld1q_u32(K.as_ptr().add(0x34)));
        state0 = vsha256hq_u32(state0, state1, tmp0);
        state1 = vsha256h2q_u32(state1, tmp2, tmp0);

        // Rounds 52-55
        tmp2 = state0;
        tmp0 = vaddq_u32(msg2, vld1q_u32(K.as_ptr().add(0x38)));
        state0 = vsha256hq_u32(state0, state1, tmp1);
        state1 = vsha256h2q_u32(state1, tmp2, tmp1);

        // Rounds 56-59
        tmp2 = state0;
        tmp1 = vaddq_u32(msg3, vld1q_u32(K.as_ptr().add(0x3c)));
        state0 = vsha256hq_u32(state0, state1, tmp0);
        state1 = vsha256h2q_u32(state1, tmp2, tmp0);

        // Rounds 60-63
        tmp2 = state0;
        state0 = vsha256hq_u32(state0, state1, tmp1);
        state1 = vsha256h2q_u32(state1, tmp2, tmp1);

        // Combine state
        state0 = vaddq_u32(state0, abcd_save);
        state1 = vaddq_u32(state1, efgh_save);

        // Save state
        vst1q_u32(state.as_mut_ptr().add(0), state0);
        vst1q_u32(state.as_mut_ptr().add(4), state1);
    }

    /// Computes SHA256d on two 64-byte inputs in parallel using x86 SHA-NI intrinsics.
    ///
    /// Based on Bitcoin Core's `sha256d64_x86_shani::Transform_2way`.
    // https://github.com/bitcoin/bitcoin/blob/master/src/crypto/sha256_x86_shani.cpp
    #[cfg(all(target_arch = "x86_64", any(feature = "std", feature = "cpufeatures")))]
    #[target_feature(enable = "sha,sse2,ssse3,sse4.1")]
    unsafe fn sha256d_64_x86_shani_2way(output: &mut [[u8; 32]; 2], input: &[[u8; 64]; 2]) {
        use core::arch::x86_64::*;

        #[allow(non_snake_case)]
        let MASK: __m128i =
            _mm_set_epi64x(0x0c0d_0e0f_0809_0a0bu64 as i64, 0x0405_0607_0001_0203u64 as i64);

        // Initial state in shuffled (ABEF/CDGH) form.
        #[allow(non_snake_case)]
        let INIT0: __m128i =
            _mm_set_epi64x(0x6a09e667bb67ae85u64 as i64, 0x510e527f9b05688cu64 as i64);
        #[allow(non_snake_case)]
        let INIT1: __m128i =
            _mm_set_epi64x(0x3c6ef372a54ff53au64 as i64, 0x1f83d9ab5be0cd19u64 as i64);

        // Precomputed W[i] + K[i] for the 2nd transform (padding block).
        #[rustfmt::skip]
        const MIDS: [[u64; 2]; 16] = [
            [0x71374491c28a2f98, 0xe9b5dba5b5c0fbcf],
            [0x59f111f13956c25b, 0xab1c5ed5923f82a4],
            [0x12835b01d807aa98, 0x550c7dc3243185be],
            [0x80deb1fe72be5d74, 0xc19bf3749bdc06a7],
            [0xf0fe4786649b69c1, 0x240cf2540fe1edc6],
            [0x6cc984be4fe9346f, 0x16f988fa61b9411e],
            [0xa88e5a6df2c65152, 0xb9d99ec7b019fc65],
            [0xe70eeaa09a1231c3, 0xc7353eb0fdb1232b],
            [0xcb976d5f3069bad5, 0xdc1eeefd5a0f118f],
            [0xde0b7a040a35b689, 0xe15d5b1658f4ca9d],
            [0x37088980007f3e86, 0x6fab9537a507ea32],
            [0x0d8cd6f117406110, 0xc0bbbe37cdaa3b6d],
            [0xdb48a36383613bda, 0x6fd15ca70b02e931],
            [0x31338431521afaca, 0x6d4378906ed41a95],
            [0x9eccabbdc39c91f2, 0x532fb63cb5c9a0e6],
            [0x07237ea3d2c741c6, 0x4c191d76a4954b68],
        ];

        // ---- Helper closures ----

        // QuadRound with just constants (no message): 4 SHA256 rounds using
        // precomputed W+K values packed as two u64.
        #[inline(always)]
        unsafe fn quad_round_k(
            s0: &mut __m128i, s1: &mut __m128i, k1: u64, k0: u64,
        ) {
            let msg = _mm_set_epi64x(k1 as i64, k0 as i64);
            *s1 = _mm_sha256rnds2_epu32(*s1, *s0, msg);
            *s0 = _mm_sha256rnds2_epu32(*s0, *s1, _mm_shuffle_epi32(msg, 0x0E));
        }

        // QuadRound with message + round constants.
        #[inline(always)]
        unsafe fn quad_round(
            s0: &mut __m128i, s1: &mut __m128i, m: __m128i, k1: u64, k0: u64,
        ) {
            let msg = _mm_add_epi32(m, _mm_set_epi64x(k1 as i64, k0 as i64));
            *s1 = _mm_sha256rnds2_epu32(*s1, *s0, msg);
            *s0 = _mm_sha256rnds2_epu32(*s0, *s1, _mm_shuffle_epi32(msg, 0x0E));
        }

        #[inline(always)]
        unsafe fn shift_message_a(m0: &mut __m128i, m1: __m128i) {
            *m0 = _mm_sha256msg1_epu32(*m0, m1);
        }

        #[inline(always)]
        unsafe fn shift_message_c(m0: __m128i, m1: __m128i, m2: &mut __m128i) {
            *m2 = _mm_sha256msg2_epu32(
                _mm_add_epi32(*m2, _mm_alignr_epi8(m1, m0, 4)),
                m1,
            );
        }

        #[inline(always)]
        unsafe fn shift_message_b(m0: &mut __m128i, m1: __m128i, m2: &mut __m128i) {
            shift_message_c(*m0, m1, m2);
            shift_message_a(m0, m1);
        }

        #[inline(always)]
        unsafe fn unshuffle(s0: &mut __m128i, s1: &mut __m128i) {
            let t1 = _mm_shuffle_epi32(*s0, 0x1B);
            let t2 = _mm_shuffle_epi32(*s1, 0xB1);
            *s0 = _mm_blend_epi16(t1, t2, 0xF0);
            *s1 = _mm_alignr_epi8(t2, t1, 0x08);
        }

        #[inline(always)]
        unsafe fn load(input: *const u8, mask: __m128i) -> __m128i {
            _mm_shuffle_epi8(_mm_loadu_si128(input as *const __m128i), mask)
        }

        #[inline(always)]
        unsafe fn save(out: *mut u8, s: __m128i, mask: __m128i) {
            _mm_storeu_si128(out as *mut __m128i, _mm_shuffle_epi8(s, mask));
        }

        // ---- Begin 2-way Transform ----

        // Load and byte-swap input messages for both lanes.
        let mut am0 = load(input[0].as_ptr(), MASK);
        let mut am1 = load(input[0].as_ptr().add(16), MASK);
        let mut am2 = load(input[0].as_ptr().add(32), MASK);
        let mut am3 = load(input[0].as_ptr().add(48), MASK);
        let mut bm0 = load(input[1].as_ptr(), MASK);
        let mut bm1 = load(input[1].as_ptr().add(16), MASK);
        let mut bm2 = load(input[1].as_ptr().add(32), MASK);
        let mut bm3 = load(input[1].as_ptr().add(48), MASK);

        // Initialize state (already in shuffled ABEF/CDGH form).
        let mut as0 = INIT0;
        let mut as1 = INIT1;
        let mut bs0 = INIT0;
        let mut bs1 = INIT1;

        // ---- Transform 1: SHA256 of 64-byte input block ----

        // Rounds 0-3
        quad_round(&mut as0, &mut as1, am0, 0xe9b5dba5b5c0fbcf, 0x71374491428a2f98);
        quad_round(&mut bs0, &mut bs1, bm0, 0xe9b5dba5b5c0fbcf, 0x71374491428a2f98);
        // Rounds 4-7
        quad_round(&mut as0, &mut as1, am1, 0xab1c5ed5923f82a4, 0x59f111f13956c25b);
        quad_round(&mut bs0, &mut bs1, bm1, 0xab1c5ed5923f82a4, 0x59f111f13956c25b);
        // Rounds 8-11
        shift_message_a(&mut am0, am1);
        quad_round(&mut as0, &mut as1, am2, 0x550c7dc3243185be, 0x12835b01d807aa98);
        quad_round(&mut bs0, &mut bs1, bm2, 0x550c7dc3243185be, 0x12835b01d807aa98);
        shift_message_a(&mut bm0, bm1);
        // Rounds 12-15
        shift_message_b(&mut am1, am2, &mut am0);
        quad_round(&mut as0, &mut as1, am3, 0xc19bf1749bdc06a7, 0x80deb1fe72be5d74);
        quad_round(&mut bs0, &mut bs1, bm3, 0xc19bf1749bdc06a7, 0x80deb1fe72be5d74);
        shift_message_b(&mut bm1, bm2, &mut bm0);
        // Rounds 16-19
        shift_message_b(&mut am2, am3, &mut am1);
        quad_round(&mut as0, &mut as1, am0, 0x240ca1cc0fc19dc6, 0xefbe4786e49b69c1);
        quad_round(&mut bs0, &mut bs1, bm0, 0x240ca1cc0fc19dc6, 0xefbe4786e49b69c1);
        shift_message_b(&mut bm2, bm3, &mut bm1);
        // Rounds 20-23
        shift_message_b(&mut am3, am0, &mut am2);
        quad_round(&mut as0, &mut as1, am1, 0x76f988da5cb0a9dc, 0x4a7484aa2de92c6f);
        quad_round(&mut bs0, &mut bs1, bm1, 0x76f988da5cb0a9dc, 0x4a7484aa2de92c6f);
        shift_message_b(&mut bm3, bm0, &mut bm2);
        // Rounds 24-27
        shift_message_b(&mut am0, am1, &mut am3);
        quad_round(&mut as0, &mut as1, am2, 0xbf597fc7b00327c8, 0xa831c66d983e5152);
        quad_round(&mut bs0, &mut bs1, bm2, 0xbf597fc7b00327c8, 0xa831c66d983e5152);
        shift_message_b(&mut bm0, bm1, &mut bm3);
        // Rounds 28-31
        shift_message_b(&mut am1, am2, &mut am0);
        quad_round(&mut as0, &mut as1, am3, 0x1429296706ca6351, 0xd5a79147c6e00bf3);
        quad_round(&mut bs0, &mut bs1, bm3, 0x1429296706ca6351, 0xd5a79147c6e00bf3);
        shift_message_b(&mut bm1, bm2, &mut bm0);
        // Rounds 32-35
        shift_message_b(&mut am2, am3, &mut am1);
        quad_round(&mut as0, &mut as1, am0, 0x53380d134d2c6dfc, 0x2e1b213827b70a85);
        quad_round(&mut bs0, &mut bs1, bm0, 0x53380d134d2c6dfc, 0x2e1b213827b70a85);
        shift_message_b(&mut bm2, bm3, &mut bm1);
        // Rounds 36-39
        shift_message_b(&mut am3, am0, &mut am2);
        quad_round(&mut as0, &mut as1, am1, 0x92722c8581c2c92e, 0x766a0abb650a7354);
        quad_round(&mut bs0, &mut bs1, bm1, 0x92722c8581c2c92e, 0x766a0abb650a7354);
        shift_message_b(&mut bm3, bm0, &mut bm2);
        // Rounds 40-43
        shift_message_b(&mut am0, am1, &mut am3);
        quad_round(&mut as0, &mut as1, am2, 0xc76c51a3c24b8b70, 0xa81a664ba2bfe8a1);
        quad_round(&mut bs0, &mut bs1, bm2, 0xc76c51a3c24b8b70, 0xa81a664ba2bfe8a1);
        shift_message_b(&mut bm0, bm1, &mut bm3);
        // Rounds 44-47
        shift_message_b(&mut am1, am2, &mut am0);
        quad_round(&mut as0, &mut as1, am3, 0x106aa070f40e3585, 0xd6990624d192e819);
        quad_round(&mut bs0, &mut bs1, bm3, 0x106aa070f40e3585, 0xd6990624d192e819);
        shift_message_b(&mut bm1, bm2, &mut bm0);
        // Rounds 48-51
        shift_message_b(&mut am2, am3, &mut am1);
        quad_round(&mut as0, &mut as1, am0, 0x34b0bcb52748774c, 0x1e376c0819a4c116);
        quad_round(&mut bs0, &mut bs1, bm0, 0x34b0bcb52748774c, 0x1e376c0819a4c116);
        shift_message_b(&mut bm2, bm3, &mut bm1);
        // Rounds 52-55
        shift_message_c(am0, am1, &mut am2);
        quad_round(&mut as0, &mut as1, am1, 0x682e6ff35b9cca4f, 0x4ed8aa4a391c0cb3);
        quad_round(&mut bs0, &mut bs1, bm1, 0x682e6ff35b9cca4f, 0x4ed8aa4a391c0cb3);
        shift_message_c(bm0, bm1, &mut bm2);
        // Rounds 56-59
        shift_message_c(am1, am2, &mut am3);
        quad_round(&mut as0, &mut as1, am2, 0x8cc7020884c87814, 0x78a5636f748f82ee);
        quad_round(&mut bs0, &mut bs1, bm2, 0x8cc7020884c87814, 0x78a5636f748f82ee);
        shift_message_c(bm1, bm2, &mut bm3);
        // Rounds 60-63
        quad_round(&mut as0, &mut as1, am3, 0xc67178f2bef9a3f7, 0xa4506ceb90befffa);
        quad_round(&mut bs0, &mut bs1, bm3, 0xc67178f2bef9a3f7, 0xa4506ceb90befffa);

        // Add initial state back.
        as0 = _mm_add_epi32(as0, INIT0);
        as1 = _mm_add_epi32(as1, INIT1);
        bs0 = _mm_add_epi32(bs0, INIT0);
        bs1 = _mm_add_epi32(bs1, INIT1);

        // ---- Transform 2: SHA256 of result with padding ----
        // The second block is the 256-bit hash + padding (precomputed as MIDS).

        // Save state from Transform 1 as the starting state for Transform 2.
        let as0_save = as0;
        let as1_save = as1;
        let bs0_save = bs0;
        let bs1_save = bs1;

        // All 16 rounds use precomputed MIDS (W+K values).
        for r in 0..16 {
            quad_round_k(
                &mut as0, &mut as1, MIDS[r][1], MIDS[r][0],
            );
            quad_round_k(
                &mut bs0, &mut bs1, MIDS[r][1], MIDS[r][0],
            );
        }

        // Add saved state.
        as0 = _mm_add_epi32(as0, as0_save);
        as1 = _mm_add_epi32(as1, as1_save);
        bs0 = _mm_add_epi32(bs0, bs0_save);
        bs1 = _mm_add_epi32(bs1, bs1_save);

        // ---- Transform 3: SHA256d final (double hash) ----
        // The input to Transform 3 is the 256-bit state from Transform 2,
        // unshuffled back to standard order, then treated as a 32-byte message
        // with standard SHA256 padding appended.

        // Unshuffle state back to standard ABCDEFGH order for use as message.
        unshuffle(&mut as0, &mut as1);
        unshuffle(&mut bs0, &mut bs1);

        // Message words for Transform 3:
        //   m0 = state[0..3] (ABCD)
        //   m1 = state[4..7] (EFGH)
        //   m2 = 0x80000000, 0, 0, 0  (padding start)
        //   m3 = 0, 0, 0, 0x100       (length = 256 bits, big-endian)
        am0 = as0;
        am1 = as1;
        am2 = _mm_set_epi64x(0x0, 0x80000000_00000000u64 as i64);
        am3 = _mm_set_epi64x(0x10000000000, 0x0);
        bm0 = bs0;
        bm1 = bs1;
        bm2 = am2;
        bm3 = am3;

        // Re-initialize state to SHA256 IV (in shuffled form).
        as0 = INIT0;
        as1 = INIT1;
        bs0 = INIT0;
        bs1 = INIT1;

        // Rounds 0-3
        quad_round(&mut as0, &mut as1, am0, 0xe9b5dba5b5c0fbcf, 0x71374491428a2f98);
        quad_round(&mut bs0, &mut bs1, bm0, 0xe9b5dba5b5c0fbcf, 0x71374491428a2f98);
        // Rounds 4-7
        quad_round(&mut as0, &mut as1, am1, 0xab1c5ed5923f82a4, 0x59f111f13956c25b);
        quad_round(&mut bs0, &mut bs1, bm1, 0xab1c5ed5923f82a4, 0x59f111f13956c25b);
        // Rounds 8-11
        shift_message_a(&mut am0, am1);
        quad_round(&mut as0, &mut as1, am2, 0x550c7dc3243185be, 0x12835b01d807aa98);
        quad_round(&mut bs0, &mut bs1, bm2, 0x550c7dc3243185be, 0x12835b01d807aa98);
        shift_message_a(&mut bm0, bm1);
        // Rounds 12-15
        shift_message_b(&mut am1, am2, &mut am0);
        quad_round(&mut as0, &mut as1, am3, 0xc19bf1749bdc06a7, 0x80deb1fe72be5d74);
        quad_round(&mut bs0, &mut bs1, bm3, 0xc19bf1749bdc06a7, 0x80deb1fe72be5d74);
        shift_message_b(&mut bm1, bm2, &mut bm0);
        // Rounds 16-19
        shift_message_b(&mut am2, am3, &mut am1);
        quad_round(&mut as0, &mut as1, am0, 0x240ca1cc0fc19dc6, 0xefbe4786e49b69c1);
        quad_round(&mut bs0, &mut bs1, bm0, 0x240ca1cc0fc19dc6, 0xefbe4786e49b69c1);
        shift_message_b(&mut bm2, bm3, &mut bm1);
        // Rounds 20-23
        shift_message_b(&mut am3, am0, &mut am2);
        quad_round(&mut as0, &mut as1, am1, 0x76f988da5cb0a9dc, 0x4a7484aa2de92c6f);
        quad_round(&mut bs0, &mut bs1, bm1, 0x76f988da5cb0a9dc, 0x4a7484aa2de92c6f);
        shift_message_b(&mut bm3, bm0, &mut bm2);
        // Rounds 24-27
        shift_message_b(&mut am0, am1, &mut am3);
        quad_round(&mut as0, &mut as1, am2, 0xbf597fc7b00327c8, 0xa831c66d983e5152);
        quad_round(&mut bs0, &mut bs1, bm2, 0xbf597fc7b00327c8, 0xa831c66d983e5152);
        shift_message_b(&mut bm0, bm1, &mut bm3);
        // Rounds 28-31
        shift_message_b(&mut am1, am2, &mut am0);
        quad_round(&mut as0, &mut as1, am3, 0x1429296706ca6351, 0xd5a79147c6e00bf3);
        quad_round(&mut bs0, &mut bs1, bm3, 0x1429296706ca6351, 0xd5a79147c6e00bf3);
        shift_message_b(&mut bm1, bm2, &mut bm0);
        // Rounds 32-35
        shift_message_b(&mut am2, am3, &mut am1);
        quad_round(&mut as0, &mut as1, am0, 0x53380d134d2c6dfc, 0x2e1b213827b70a85);
        quad_round(&mut bs0, &mut bs1, bm0, 0x53380d134d2c6dfc, 0x2e1b213827b70a85);
        shift_message_b(&mut bm2, bm3, &mut bm1);
        // Rounds 36-39
        shift_message_b(&mut am3, am0, &mut am2);
        quad_round(&mut as0, &mut as1, am1, 0x92722c8581c2c92e, 0x766a0abb650a7354);
        quad_round(&mut bs0, &mut bs1, bm1, 0x92722c8581c2c92e, 0x766a0abb650a7354);
        shift_message_b(&mut bm3, bm0, &mut bm2);
        // Rounds 40-43
        shift_message_b(&mut am0, am1, &mut am3);
        quad_round(&mut as0, &mut as1, am2, 0xc76c51a3c24b8b70, 0xa81a664ba2bfe8a1);
        quad_round(&mut bs0, &mut bs1, bm2, 0xc76c51a3c24b8b70, 0xa81a664ba2bfe8a1);
        shift_message_b(&mut bm0, bm1, &mut bm3);
        // Rounds 44-47
        shift_message_b(&mut am1, am2, &mut am0);
        quad_round(&mut as0, &mut as1, am3, 0x106aa070f40e3585, 0xd6990624d192e819);
        quad_round(&mut bs0, &mut bs1, bm3, 0x106aa070f40e3585, 0xd6990624d192e819);
        shift_message_b(&mut bm1, bm2, &mut bm0);
        // Rounds 48-51
        shift_message_b(&mut am2, am3, &mut am1);
        quad_round(&mut as0, &mut as1, am0, 0x34b0bcb52748774c, 0x1e376c0819a4c116);
        quad_round(&mut bs0, &mut bs1, bm0, 0x34b0bcb52748774c, 0x1e376c0819a4c116);
        shift_message_b(&mut bm2, bm3, &mut bm1);
        // Rounds 52-55
        shift_message_c(am0, am1, &mut am2);
        quad_round(&mut as0, &mut as1, am1, 0x682e6ff35b9cca4f, 0x4ed8aa4a391c0cb3);
        quad_round(&mut bs0, &mut bs1, bm1, 0x682e6ff35b9cca4f, 0x4ed8aa4a391c0cb3);
        shift_message_c(bm0, bm1, &mut bm2);
        // Rounds 56-59
        shift_message_c(am1, am2, &mut am3);
        quad_round(&mut as0, &mut as1, am2, 0x8cc7020884c87814, 0x78a5636f748f82ee);
        quad_round(&mut bs0, &mut bs1, bm2, 0x8cc7020884c87814, 0x78a5636f748f82ee);
        shift_message_c(bm1, bm2, &mut bm3);
        // Rounds 60-63
        quad_round(&mut as0, &mut as1, am3, 0xc67178f2bef9a3f7, 0xa4506ceb90befffa);
        quad_round(&mut bs0, &mut bs1, bm3, 0xc67178f2bef9a3f7, 0xa4506ceb90befffa);

        // Add initial state back.
        as0 = _mm_add_epi32(as0, INIT0);
        as1 = _mm_add_epi32(as1, INIT1);
        bs0 = _mm_add_epi32(bs0, INIT0);
        bs1 = _mm_add_epi32(bs1, INIT1);

        // Unshuffle to get standard ABCDEFGH order, then byte-swap and store.
        unshuffle(&mut as0, &mut as1);
        unshuffle(&mut bs0, &mut bs1);

        save(output[0].as_mut_ptr(), as0, MASK);
        save(output[0].as_mut_ptr().add(16), as1, MASK);
        save(output[1].as_mut_ptr(), bs0, MASK);
        save(output[1].as_mut_ptr().add(16), bs1, MASK);
    }

    /// Computes SHA256d on four 64-byte inputs in parallel using SSE4.1 intrinsics.
    ///
    /// Based on Bitcoin Core's `sha256d64_sse41::Transform_4way`.
    // https://github.com/bitcoin/bitcoin/blob/master/src/crypto/sha256_sse41.cpp
    #[cfg(all(target_arch = "x86_64", any(feature = "std", feature = "cpufeatures")))]
    #[target_feature(enable = "sse4.1")]
    unsafe fn sha256d_64_sse41_4way(output: &mut [[u8; 32]; 4], input: &[[u8; 64]; 4]) {
        use core::arch::x86_64::*;

        #[inline(always)]
        unsafe fn k(x: u32) -> __m128i { _mm_set1_epi32(x as i32) }
        #[inline(always)]
        unsafe fn add2(a: __m128i, b: __m128i) -> __m128i { _mm_add_epi32(a, b) }
        #[inline(always)]
        unsafe fn add3(a: __m128i, b: __m128i, c: __m128i) -> __m128i { add2(add2(a, b), c) }
        #[inline(always)]
        unsafe fn add4(a: __m128i, b: __m128i, c: __m128i, d: __m128i) -> __m128i {
            add2(add2(a, b), add2(c, d))
        }
        #[inline(always)]
        unsafe fn add5(
            a: __m128i,
            b: __m128i,
            c: __m128i,
            d: __m128i,
            e: __m128i,
        ) -> __m128i {
            add2(add3(a, b, c), add2(d, e))
        }
        #[inline(always)]
        unsafe fn xor2(a: __m128i, b: __m128i) -> __m128i { _mm_xor_si128(a, b) }
        #[inline(always)]
        unsafe fn xor3(a: __m128i, b: __m128i, c: __m128i) -> __m128i { xor2(xor2(a, b), c) }
        #[inline(always)]
        unsafe fn or(a: __m128i, b: __m128i) -> __m128i { _mm_or_si128(a, b) }
        #[inline(always)]
        unsafe fn and(a: __m128i, b: __m128i) -> __m128i { _mm_and_si128(a, b) }

        #[inline(always)]
        unsafe fn ch(x: __m128i, y: __m128i, z: __m128i) -> __m128i {
            xor2(z, and(x, xor2(y, z)))
        }
        #[inline(always)]
        unsafe fn maj(x: __m128i, y: __m128i, z: __m128i) -> __m128i {
            or(and(x, y), and(z, or(x, y)))
        }
        #[inline(always)]
        unsafe fn big_sigma0(x: __m128i) -> __m128i {
            xor3(
                or(_mm_srli_epi32(x, 2), _mm_slli_epi32(x, 30)),
                or(_mm_srli_epi32(x, 13), _mm_slli_epi32(x, 19)),
                or(_mm_srli_epi32(x, 22), _mm_slli_epi32(x, 10)),
            )
        }
        #[inline(always)]
        unsafe fn big_sigma1(x: __m128i) -> __m128i {
            xor3(
                or(_mm_srli_epi32(x, 6), _mm_slli_epi32(x, 26)),
                or(_mm_srli_epi32(x, 11), _mm_slli_epi32(x, 21)),
                or(_mm_srli_epi32(x, 25), _mm_slli_epi32(x, 7)),
            )
        }
        #[inline(always)]
        unsafe fn small_sigma0(x: __m128i) -> __m128i {
            xor3(
                or(_mm_srli_epi32(x, 7), _mm_slli_epi32(x, 25)),
                or(_mm_srli_epi32(x, 18), _mm_slli_epi32(x, 14)),
                _mm_srli_epi32(x, 3),
            )
        }
        #[inline(always)]
        unsafe fn small_sigma1(x: __m128i) -> __m128i {
            xor3(
                or(_mm_srli_epi32(x, 17), _mm_slli_epi32(x, 15)),
                or(_mm_srli_epi32(x, 19), _mm_slli_epi32(x, 13)),
                _mm_srli_epi32(x, 10),
            )
        }

        #[inline(always)]
        unsafe fn round(
            a: __m128i,
            b: __m128i,
            c: __m128i,
            d: &mut __m128i,
            e: __m128i,
            f: __m128i,
            g: __m128i,
            h: &mut __m128i,
            ki: __m128i,
        ) {
            let t1 = add4(*h, big_sigma1(e), ch(e, f, g), ki);
            let t2 = add2(big_sigma0(a), maj(a, b, c));
            *d = add2(*d, t1);
            *h = add2(t1, t2);
        }

        #[inline(always)]
        unsafe fn read4(base: *const u8, offset: usize, shuf: __m128i) -> __m128i {
            let w0 = (base.add(offset) as *const u32).read_unaligned();
            let w1 = (base.add(64 + offset) as *const u32).read_unaligned();
            let w2 = (base.add(128 + offset) as *const u32).read_unaligned();
            let w3 = (base.add(192 + offset) as *const u32).read_unaligned();
            let ret = _mm_set_epi32(w0 as i32, w1 as i32, w2 as i32, w3 as i32);
            _mm_shuffle_epi8(ret, shuf)
        }

        #[inline(always)]
        unsafe fn write4(base: *mut u8, offset: usize, v: __m128i, shuf: __m128i) {
            let v = _mm_shuffle_epi8(v, shuf);
            (base.add(offset) as *mut u32)
                .write_unaligned(_mm_extract_epi32::<3>(v) as u32);
            (base.add(32 + offset) as *mut u32)
                .write_unaligned(_mm_extract_epi32::<2>(v) as u32);
            (base.add(64 + offset) as *mut u32)
                .write_unaligned(_mm_extract_epi32::<1>(v) as u32);
            (base.add(96 + offset) as *mut u32)
                .write_unaligned(_mm_extract_epi32::<0>(v) as u32);
        }

        let shuf_mask = _mm_set_epi32(
            0x0C0D0E0Fu32 as i32,
            0x08090A0Bu32 as i32,
            0x04050607u32 as i32,
            0x00010203u32 as i32,
        );

        #[rustfmt::skip]
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
        ];

        #[rustfmt::skip]
        const MIDS: [u32; 64] = [
            0xc28a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf374,
            0x649b69c1, 0xf0fe4786, 0x0fe1edc6, 0x240cf254, 0x4fe9346f, 0x6cc984be, 0x61b9411e, 0x16f988fa,
            0xf2c65152, 0xa88e5a6d, 0xb019fc65, 0xb9d99ec7, 0x9a1231c3, 0xe70eeaa0, 0xfdb1232b, 0xc7353eb0,
            0x3069bad5, 0xcb976d5f, 0x5a0f118f, 0xdc1eeefd, 0x0a35b689, 0xde0b7a04, 0x58f4ca9d, 0xe15d5b16,
            0x007f3e86, 0x37088980, 0xa507ea32, 0x6fab9537, 0x17406110, 0x0d8cd6f1, 0xcdaa3b6d, 0xc0bbbe37,
            0x83613bda, 0xdb48a363, 0x0b02e931, 0x6fd15ca7, 0x521afaca, 0x31338431, 0x6ed41a95, 0x6d437890,
            0xc39c91f2, 0x9eccabbd, 0xb5c9a0e6, 0x532fb63c, 0xd2c741c6, 0x07237ea3, 0xa4954b68, 0x4c191d76,
        ];

        let inp = input.as_ptr() as *const u8;

        // SHA-256 initial hash value
        let iv0 = k(0x6a09e667);
        let iv1 = k(0xbb67ae85);
        let iv2 = k(0x3c6ef372);
        let iv3 = k(0xa54ff53a);
        let iv4 = k(0x510e527f);
        let iv5 = k(0x9b05688c);
        let iv6 = k(0x1f83d9ab);
        let iv7 = k(0x5be0cd19);

        // ---- Transform 1: SHA-256 of the 64-byte input blocks ----
        let mut a = iv0;
        let mut b = iv1;
        let mut c = iv2;
        let mut d = iv3;
        let mut e = iv4;
        let mut f = iv5;
        let mut g = iv6;
        let mut h = iv7;

        // Load message words
        let mut w0 = read4(inp, 4 * 0, shuf_mask);
        let mut w1 = read4(inp, 4 * 1, shuf_mask);
        let mut w2 = read4(inp, 4 * 2, shuf_mask);
        let mut w3 = read4(inp, 4 * 3, shuf_mask);
        let mut w4 = read4(inp, 4 * 4, shuf_mask);
        let mut w5 = read4(inp, 4 * 5, shuf_mask);
        let mut w6 = read4(inp, 4 * 6, shuf_mask);
        let mut w7 = read4(inp, 4 * 7, shuf_mask);
        let mut w8 = read4(inp, 4 * 8, shuf_mask);
        let mut w9 = read4(inp, 4 * 9, shuf_mask);
        let mut w10 = read4(inp, 4 * 10, shuf_mask);
        let mut w11 = read4(inp, 4 * 11, shuf_mask);
        let mut w12 = read4(inp, 4 * 12, shuf_mask);
        let mut w13 = read4(inp, 4 * 13, shuf_mask);
        let mut w14 = read4(inp, 4 * 14, shuf_mask);
        let mut w15 = read4(inp, 4 * 15, shuf_mask);

        // Rounds 0-15
        round(a, b, c, &mut d, e, f, g, &mut h, add2(k(K[0]), w0));
        round(h, a, b, &mut c, d, e, f, &mut g, add2(k(K[1]), w1));
        round(g, h, a, &mut b, c, d, e, &mut f, add2(k(K[2]), w2));
        round(f, g, h, &mut a, b, c, d, &mut e, add2(k(K[3]), w3));
        round(e, f, g, &mut h, a, b, c, &mut d, add2(k(K[4]), w4));
        round(d, e, f, &mut g, h, a, b, &mut c, add2(k(K[5]), w5));
        round(c, d, e, &mut f, g, h, a, &mut b, add2(k(K[6]), w6));
        round(b, c, d, &mut e, f, g, h, &mut a, add2(k(K[7]), w7));
        round(a, b, c, &mut d, e, f, g, &mut h, add2(k(K[8]), w8));
        round(h, a, b, &mut c, d, e, f, &mut g, add2(k(K[9]), w9));
        round(g, h, a, &mut b, c, d, e, &mut f, add2(k(K[10]), w10));
        round(f, g, h, &mut a, b, c, d, &mut e, add2(k(K[11]), w11));
        round(e, f, g, &mut h, a, b, c, &mut d, add2(k(K[12]), w12));
        round(d, e, f, &mut g, h, a, b, &mut c, add2(k(K[13]), w13));
        round(c, d, e, &mut f, g, h, a, &mut b, add2(k(K[14]), w14));
        round(b, c, d, &mut e, f, g, h, &mut a, add2(k(K[15]), w15));

        // Rounds 16-63 with message schedule
        w0 = add4(small_sigma1(w14), w9, small_sigma0(w1), w0);
        round(a, b, c, &mut d, e, f, g, &mut h, add2(k(K[16]), w0));
        w1 = add4(small_sigma1(w15), w10, small_sigma0(w2), w1);
        round(h, a, b, &mut c, d, e, f, &mut g, add2(k(K[17]), w1));
        w2 = add4(small_sigma1(w0), w11, small_sigma0(w3), w2);
        round(g, h, a, &mut b, c, d, e, &mut f, add2(k(K[18]), w2));
        w3 = add4(small_sigma1(w1), w12, small_sigma0(w4), w3);
        round(f, g, h, &mut a, b, c, d, &mut e, add2(k(K[19]), w3));
        w4 = add4(small_sigma1(w2), w13, small_sigma0(w5), w4);
        round(e, f, g, &mut h, a, b, c, &mut d, add2(k(K[20]), w4));
        w5 = add4(small_sigma1(w3), w14, small_sigma0(w6), w5);
        round(d, e, f, &mut g, h, a, b, &mut c, add2(k(K[21]), w5));
        w6 = add4(small_sigma1(w4), w15, small_sigma0(w7), w6);
        round(c, d, e, &mut f, g, h, a, &mut b, add2(k(K[22]), w6));
        w7 = add4(small_sigma1(w5), w0, small_sigma0(w8), w7);
        round(b, c, d, &mut e, f, g, h, &mut a, add2(k(K[23]), w7));
        w8 = add4(small_sigma1(w6), w1, small_sigma0(w9), w8);
        round(a, b, c, &mut d, e, f, g, &mut h, add2(k(K[24]), w8));
        w9 = add4(small_sigma1(w7), w2, small_sigma0(w10), w9);
        round(h, a, b, &mut c, d, e, f, &mut g, add2(k(K[25]), w9));
        w10 = add4(small_sigma1(w8), w3, small_sigma0(w11), w10);
        round(g, h, a, &mut b, c, d, e, &mut f, add2(k(K[26]), w10));
        w11 = add4(small_sigma1(w9), w4, small_sigma0(w12), w11);
        round(f, g, h, &mut a, b, c, d, &mut e, add2(k(K[27]), w11));
        w12 = add4(small_sigma1(w10), w5, small_sigma0(w13), w12);
        round(e, f, g, &mut h, a, b, c, &mut d, add2(k(K[28]), w12));
        w13 = add4(small_sigma1(w11), w6, small_sigma0(w14), w13);
        round(d, e, f, &mut g, h, a, b, &mut c, add2(k(K[29]), w13));
        w14 = add4(small_sigma1(w12), w7, small_sigma0(w15), w14);
        round(c, d, e, &mut f, g, h, a, &mut b, add2(k(K[30]), w14));
        w15 = add4(small_sigma1(w13), w8, small_sigma0(w0), w15);
        round(b, c, d, &mut e, f, g, h, &mut a, add2(k(K[31]), w15));
        w0 = add4(small_sigma1(w14), w9, small_sigma0(w1), w0);
        round(a, b, c, &mut d, e, f, g, &mut h, add2(k(K[32]), w0));
        w1 = add4(small_sigma1(w15), w10, small_sigma0(w2), w1);
        round(h, a, b, &mut c, d, e, f, &mut g, add2(k(K[33]), w1));
        w2 = add4(small_sigma1(w0), w11, small_sigma0(w3), w2);
        round(g, h, a, &mut b, c, d, e, &mut f, add2(k(K[34]), w2));
        w3 = add4(small_sigma1(w1), w12, small_sigma0(w4), w3);
        round(f, g, h, &mut a, b, c, d, &mut e, add2(k(K[35]), w3));
        w4 = add4(small_sigma1(w2), w13, small_sigma0(w5), w4);
        round(e, f, g, &mut h, a, b, c, &mut d, add2(k(K[36]), w4));
        w5 = add4(small_sigma1(w3), w14, small_sigma0(w6), w5);
        round(d, e, f, &mut g, h, a, b, &mut c, add2(k(K[37]), w5));
        w6 = add4(small_sigma1(w4), w15, small_sigma0(w7), w6);
        round(c, d, e, &mut f, g, h, a, &mut b, add2(k(K[38]), w6));
        w7 = add4(small_sigma1(w5), w0, small_sigma0(w8), w7);
        round(b, c, d, &mut e, f, g, h, &mut a, add2(k(K[39]), w7));
        w8 = add4(small_sigma1(w6), w1, small_sigma0(w9), w8);
        round(a, b, c, &mut d, e, f, g, &mut h, add2(k(K[40]), w8));
        w9 = add4(small_sigma1(w7), w2, small_sigma0(w10), w9);
        round(h, a, b, &mut c, d, e, f, &mut g, add2(k(K[41]), w9));
        w10 = add4(small_sigma1(w8), w3, small_sigma0(w11), w10);
        round(g, h, a, &mut b, c, d, e, &mut f, add2(k(K[42]), w10));
        w11 = add4(small_sigma1(w9), w4, small_sigma0(w12), w11);
        round(f, g, h, &mut a, b, c, d, &mut e, add2(k(K[43]), w11));
        w12 = add4(small_sigma1(w10), w5, small_sigma0(w13), w12);
        round(e, f, g, &mut h, a, b, c, &mut d, add2(k(K[44]), w12));
        w13 = add4(small_sigma1(w11), w6, small_sigma0(w14), w13);
        round(d, e, f, &mut g, h, a, b, &mut c, add2(k(K[45]), w13));
        w14 = add4(small_sigma1(w12), w7, small_sigma0(w15), w14);
        round(c, d, e, &mut f, g, h, a, &mut b, add2(k(K[46]), w14));
        w15 = add4(small_sigma1(w13), w8, small_sigma0(w0), w15);
        round(b, c, d, &mut e, f, g, h, &mut a, add2(k(K[47]), w15));
        w0 = add4(small_sigma1(w14), w9, small_sigma0(w1), w0);
        round(a, b, c, &mut d, e, f, g, &mut h, add2(k(K[48]), w0));
        w1 = add4(small_sigma1(w15), w10, small_sigma0(w2), w1);
        round(h, a, b, &mut c, d, e, f, &mut g, add2(k(K[49]), w1));
        w2 = add4(small_sigma1(w0), w11, small_sigma0(w3), w2);
        round(g, h, a, &mut b, c, d, e, &mut f, add2(k(K[50]), w2));
        w3 = add4(small_sigma1(w1), w12, small_sigma0(w4), w3);
        round(f, g, h, &mut a, b, c, d, &mut e, add2(k(K[51]), w3));
        w4 = add4(small_sigma1(w2), w13, small_sigma0(w5), w4);
        round(e, f, g, &mut h, a, b, c, &mut d, add2(k(K[52]), w4));
        w5 = add4(small_sigma1(w3), w14, small_sigma0(w6), w5);
        round(d, e, f, &mut g, h, a, b, &mut c, add2(k(K[53]), w5));
        w6 = add4(small_sigma1(w4), w15, small_sigma0(w7), w6);
        round(c, d, e, &mut f, g, h, a, &mut b, add2(k(K[54]), w6));
        w7 = add4(small_sigma1(w5), w0, small_sigma0(w8), w7);
        round(b, c, d, &mut e, f, g, h, &mut a, add2(k(K[55]), w7));
        w8 = add4(small_sigma1(w6), w1, small_sigma0(w9), w8);
        round(a, b, c, &mut d, e, f, g, &mut h, add2(k(K[56]), w8));
        w9 = add4(small_sigma1(w7), w2, small_sigma0(w10), w9);
        round(h, a, b, &mut c, d, e, f, &mut g, add2(k(K[57]), w9));
        w10 = add4(small_sigma1(w8), w3, small_sigma0(w11), w10);
        round(g, h, a, &mut b, c, d, e, &mut f, add2(k(K[58]), w10));
        w11 = add4(small_sigma1(w9), w4, small_sigma0(w12), w11);
        round(f, g, h, &mut a, b, c, d, &mut e, add2(k(K[59]), w11));
        w12 = add4(small_sigma1(w10), w5, small_sigma0(w13), w12);
        round(e, f, g, &mut h, a, b, c, &mut d, add2(k(K[60]), w12));
        w13 = add4(small_sigma1(w11), w6, small_sigma0(w14), w13);
        round(d, e, f, &mut g, h, a, b, &mut c, add2(k(K[61]), w13));
        w14 = add4(small_sigma1(w12), w7, small_sigma0(w15), w14);
        round(c, d, e, &mut f, g, h, a, &mut b, add2(k(K[62]), w14));
        w15 = add4(small_sigma1(w13), w8, small_sigma0(w0), w15);
        round(b, c, d, &mut e, f, g, h, &mut a, add2(k(K[63]), w15));

        // Add IV back
        a = add2(a, iv0);
        b = add2(b, iv1);
        c = add2(c, iv2);
        d = add2(d, iv3);
        e = add2(e, iv4);
        f = add2(f, iv5);
        g = add2(g, iv6);
        h = add2(h, iv7);

        // Save Transform 1 output
        let t0 = a;
        let t1 = b;
        let t2 = c;
        let t3 = d;
        let t4 = e;
        let t5 = f;
        let t6 = g;
        let t7 = h;

        // ---- Transform 2: SHA-256 of midstate padding ----
        // State carries over from Transform 1 (NOT reset to IV).
        // MIDS = precomputed K + W for the padding block after a 64-byte input.

        // All 64 rounds with MIDS constants (precomputed K + padding)
        round(a, b, c, &mut d, e, f, g, &mut h, k(MIDS[0]));
        round(h, a, b, &mut c, d, e, f, &mut g, k(MIDS[1]));
        round(g, h, a, &mut b, c, d, e, &mut f, k(MIDS[2]));
        round(f, g, h, &mut a, b, c, d, &mut e, k(MIDS[3]));
        round(e, f, g, &mut h, a, b, c, &mut d, k(MIDS[4]));
        round(d, e, f, &mut g, h, a, b, &mut c, k(MIDS[5]));
        round(c, d, e, &mut f, g, h, a, &mut b, k(MIDS[6]));
        round(b, c, d, &mut e, f, g, h, &mut a, k(MIDS[7]));
        round(a, b, c, &mut d, e, f, g, &mut h, k(MIDS[8]));
        round(h, a, b, &mut c, d, e, f, &mut g, k(MIDS[9]));
        round(g, h, a, &mut b, c, d, e, &mut f, k(MIDS[10]));
        round(f, g, h, &mut a, b, c, d, &mut e, k(MIDS[11]));
        round(e, f, g, &mut h, a, b, c, &mut d, k(MIDS[12]));
        round(d, e, f, &mut g, h, a, b, &mut c, k(MIDS[13]));
        round(c, d, e, &mut f, g, h, a, &mut b, k(MIDS[14]));
        round(b, c, d, &mut e, f, g, h, &mut a, k(MIDS[15]));
        round(a, b, c, &mut d, e, f, g, &mut h, k(MIDS[16]));
        round(h, a, b, &mut c, d, e, f, &mut g, k(MIDS[17]));
        round(g, h, a, &mut b, c, d, e, &mut f, k(MIDS[18]));
        round(f, g, h, &mut a, b, c, d, &mut e, k(MIDS[19]));
        round(e, f, g, &mut h, a, b, c, &mut d, k(MIDS[20]));
        round(d, e, f, &mut g, h, a, b, &mut c, k(MIDS[21]));
        round(c, d, e, &mut f, g, h, a, &mut b, k(MIDS[22]));
        round(b, c, d, &mut e, f, g, h, &mut a, k(MIDS[23]));
        round(a, b, c, &mut d, e, f, g, &mut h, k(MIDS[24]));
        round(h, a, b, &mut c, d, e, f, &mut g, k(MIDS[25]));
        round(g, h, a, &mut b, c, d, e, &mut f, k(MIDS[26]));
        round(f, g, h, &mut a, b, c, d, &mut e, k(MIDS[27]));
        round(e, f, g, &mut h, a, b, c, &mut d, k(MIDS[28]));
        round(d, e, f, &mut g, h, a, b, &mut c, k(MIDS[29]));
        round(c, d, e, &mut f, g, h, a, &mut b, k(MIDS[30]));
        round(b, c, d, &mut e, f, g, h, &mut a, k(MIDS[31]));
        round(a, b, c, &mut d, e, f, g, &mut h, k(MIDS[32]));
        round(h, a, b, &mut c, d, e, f, &mut g, k(MIDS[33]));
        round(g, h, a, &mut b, c, d, e, &mut f, k(MIDS[34]));
        round(f, g, h, &mut a, b, c, d, &mut e, k(MIDS[35]));
        round(e, f, g, &mut h, a, b, c, &mut d, k(MIDS[36]));
        round(d, e, f, &mut g, h, a, b, &mut c, k(MIDS[37]));
        round(c, d, e, &mut f, g, h, a, &mut b, k(MIDS[38]));
        round(b, c, d, &mut e, f, g, h, &mut a, k(MIDS[39]));
        round(a, b, c, &mut d, e, f, g, &mut h, k(MIDS[40]));
        round(h, a, b, &mut c, d, e, f, &mut g, k(MIDS[41]));
        round(g, h, a, &mut b, c, d, e, &mut f, k(MIDS[42]));
        round(f, g, h, &mut a, b, c, d, &mut e, k(MIDS[43]));
        round(e, f, g, &mut h, a, b, c, &mut d, k(MIDS[44]));
        round(d, e, f, &mut g, h, a, b, &mut c, k(MIDS[45]));
        round(c, d, e, &mut f, g, h, a, &mut b, k(MIDS[46]));
        round(b, c, d, &mut e, f, g, h, &mut a, k(MIDS[47]));
        round(a, b, c, &mut d, e, f, g, &mut h, k(MIDS[48]));
        round(h, a, b, &mut c, d, e, f, &mut g, k(MIDS[49]));
        round(g, h, a, &mut b, c, d, e, &mut f, k(MIDS[50]));
        round(f, g, h, &mut a, b, c, d, &mut e, k(MIDS[51]));
        round(e, f, g, &mut h, a, b, c, &mut d, k(MIDS[52]));
        round(d, e, f, &mut g, h, a, b, &mut c, k(MIDS[53]));
        round(c, d, e, &mut f, g, h, a, &mut b, k(MIDS[54]));
        round(b, c, d, &mut e, f, g, h, &mut a, k(MIDS[55]));
        round(a, b, c, &mut d, e, f, g, &mut h, k(MIDS[56]));
        round(h, a, b, &mut c, d, e, f, &mut g, k(MIDS[57]));
        round(g, h, a, &mut b, c, d, e, &mut f, k(MIDS[58]));
        round(f, g, h, &mut a, b, c, d, &mut e, k(MIDS[59]));
        round(e, f, g, &mut h, a, b, c, &mut d, k(MIDS[60]));
        round(d, e, f, &mut g, h, a, b, &mut c, k(MIDS[61]));
        round(c, d, e, &mut f, g, h, a, &mut b, k(MIDS[62]));
        round(b, c, d, &mut e, f, g, h, &mut a, k(MIDS[63]));

        // Add Transform 1 output
        a = add2(a, t0);
        b = add2(b, t1);
        c = add2(c, t2);
        d = add2(d, t3);
        e = add2(e, t4);
        f = add2(f, t5);
        g = add2(g, t6);
        h = add2(h, t7);

        // Save w0-w7 for Transform 3
        w0 = a;
        w1 = b;
        w2 = c;
        w3 = d;
        w4 = e;
        w5 = f;
        w6 = g;
        w7 = h;

        // ---- Transform 3: Second SHA-256 hash (hash of the hash) ----
        a = iv0;
        b = iv1;
        c = iv2;
        d = iv3;
        e = iv4;
        f = iv5;
        g = iv6;
        h = iv7;

        // Rounds 0-7: hash of Transform 2 output
        round(a, b, c, &mut d, e, f, g, &mut h, add2(k(K[0]), w0));
        round(h, a, b, &mut c, d, e, f, &mut g, add2(k(K[1]), w1));
        round(g, h, a, &mut b, c, d, e, &mut f, add2(k(K[2]), w2));
        round(f, g, h, &mut a, b, c, d, &mut e, add2(k(K[3]), w3));
        round(e, f, g, &mut h, a, b, c, &mut d, add2(k(K[4]), w4));
        round(d, e, f, &mut g, h, a, b, &mut c, add2(k(K[5]), w5));
        round(c, d, e, &mut f, g, h, a, &mut b, add2(k(K[6]), w6));
        round(b, c, d, &mut e, f, g, h, &mut a, add2(k(K[7]), w7));

        // Rounds 8-15: precomputed K + padding constants
        round(a, b, c, &mut d, e, f, g, &mut h, k(0x5807aa98));
        round(h, a, b, &mut c, d, e, f, &mut g, k(0x12835b01));
        round(g, h, a, &mut b, c, d, e, &mut f, k(0x243185be));
        round(f, g, h, &mut a, b, c, d, &mut e, k(0x550c7dc3));
        round(e, f, g, &mut h, a, b, c, &mut d, k(0x72be5d74));
        round(d, e, f, &mut g, h, a, b, &mut c, k(0x80deb1fe));
        round(c, d, e, &mut f, g, h, a, &mut b, k(0x9bdc06a7));
        round(b, c, d, &mut e, f, g, h, &mut a, k(0xc19bf274));

        // Rounds 16-31: message schedule with padding constants mixed in.
        // w0-w7 are modified in place (w0→w16, w1→w17, etc.) so that
        // subsequent sigma1/sigma0 calls reference updated values.
        w0 = add2(w0, small_sigma0(w1));
        round(a, b, c, &mut d, e, f, g, &mut h, add2(k(K[16]), w0));
        w1 = add3(w1, k(0x00a00000), small_sigma0(w2));
        round(h, a, b, &mut c, d, e, f, &mut g, add2(k(K[17]), w1));
        w2 = add3(w2, small_sigma1(w0), small_sigma0(w3));
        round(g, h, a, &mut b, c, d, e, &mut f, add2(k(K[18]), w2));
        w3 = add3(w3, small_sigma1(w1), small_sigma0(w4));
        round(f, g, h, &mut a, b, c, d, &mut e, add2(k(K[19]), w3));
        w4 = add3(w4, small_sigma1(w2), small_sigma0(w5));
        round(e, f, g, &mut h, a, b, c, &mut d, add2(k(K[20]), w4));
        w5 = add3(w5, small_sigma1(w3), small_sigma0(w6));
        round(d, e, f, &mut g, h, a, b, &mut c, add2(k(K[21]), w5));
        w6 = add4(w6, small_sigma1(w4), k(0x00000100), small_sigma0(w7));
        round(c, d, e, &mut f, g, h, a, &mut b, add2(k(K[22]), w6));
        w7 = add4(w7, small_sigma1(w5), w0, k(0x11002000));
        round(b, c, d, &mut e, f, g, h, &mut a, add2(k(K[23]), w7));
        w8 = add3(k(0x80000000), small_sigma1(w6), w1);
        round(a, b, c, &mut d, e, f, g, &mut h, add2(k(K[24]), w8));
        w9 = add2(small_sigma1(w7), w2);
        round(h, a, b, &mut c, d, e, f, &mut g, add2(k(K[25]), w9));
        w10 = add2(small_sigma1(w8), w3);
        round(g, h, a, &mut b, c, d, e, &mut f, add2(k(K[26]), w10));
        w11 = add2(small_sigma1(w9), w4);
        round(f, g, h, &mut a, b, c, d, &mut e, add2(k(K[27]), w11));
        w12 = add2(small_sigma1(w10), w5);
        round(e, f, g, &mut h, a, b, c, &mut d, add2(k(K[28]), w12));
        w13 = add2(small_sigma1(w11), w6);
        round(d, e, f, &mut g, h, a, b, &mut c, add2(k(K[29]), w13));
        w14 = add3(small_sigma1(w12), w7, k(0x00400022));
        round(c, d, e, &mut f, g, h, a, &mut b, add2(k(K[30]), w14));
        w15 = add4(k(0x00000100), small_sigma1(w13), w8, small_sigma0(w0));
        round(b, c, d, &mut e, f, g, h, &mut a, add2(k(K[31]), w15));

        // Rounds 32-63: full message schedule
        w0 = add4(small_sigma1(w14), w9, small_sigma0(w1), w0);
        round(a, b, c, &mut d, e, f, g, &mut h, add2(k(K[32]), w0));
        w1 = add4(small_sigma1(w15), w10, small_sigma0(w2), w1);
        round(h, a, b, &mut c, d, e, f, &mut g, add2(k(K[33]), w1));
        w2 = add4(small_sigma1(w0), w11, small_sigma0(w3), w2);
        round(g, h, a, &mut b, c, d, e, &mut f, add2(k(K[34]), w2));
        w3 = add4(small_sigma1(w1), w12, small_sigma0(w4), w3);
        round(f, g, h, &mut a, b, c, d, &mut e, add2(k(K[35]), w3));
        w4 = add4(small_sigma1(w2), w13, small_sigma0(w5), w4);
        round(e, f, g, &mut h, a, b, c, &mut d, add2(k(K[36]), w4));
        w5 = add4(small_sigma1(w3), w14, small_sigma0(w6), w5);
        round(d, e, f, &mut g, h, a, b, &mut c, add2(k(K[37]), w5));
        w6 = add4(small_sigma1(w4), w15, small_sigma0(w7), w6);
        round(c, d, e, &mut f, g, h, a, &mut b, add2(k(K[38]), w6));
        w7 = add4(small_sigma1(w5), w0, small_sigma0(w8), w7);
        round(b, c, d, &mut e, f, g, h, &mut a, add2(k(K[39]), w7));
        w8 = add4(small_sigma1(w6), w1, small_sigma0(w9), w8);
        round(a, b, c, &mut d, e, f, g, &mut h, add2(k(K[40]), w8));
        w9 = add4(small_sigma1(w7), w2, small_sigma0(w10), w9);
        round(h, a, b, &mut c, d, e, f, &mut g, add2(k(K[41]), w9));
        w10 = add4(small_sigma1(w8), w3, small_sigma0(w11), w10);
        round(g, h, a, &mut b, c, d, e, &mut f, add2(k(K[42]), w10));
        w11 = add4(small_sigma1(w9), w4, small_sigma0(w12), w11);
        round(f, g, h, &mut a, b, c, d, &mut e, add2(k(K[43]), w11));
        w12 = add4(small_sigma1(w10), w5, small_sigma0(w13), w12);
        round(e, f, g, &mut h, a, b, c, &mut d, add2(k(K[44]), w12));
        w13 = add4(small_sigma1(w11), w6, small_sigma0(w14), w13);
        round(d, e, f, &mut g, h, a, b, &mut c, add2(k(K[45]), w13));
        w14 = add4(small_sigma1(w12), w7, small_sigma0(w15), w14);
        round(c, d, e, &mut f, g, h, a, &mut b, add2(k(K[46]), w14));
        w15 = add4(small_sigma1(w13), w8, small_sigma0(w0), w15);
        round(b, c, d, &mut e, f, g, h, &mut a, add2(k(K[47]), w15));
        w0 = add4(small_sigma1(w14), w9, small_sigma0(w1), w0);
        round(a, b, c, &mut d, e, f, g, &mut h, add2(k(K[48]), w0));
        w1 = add4(small_sigma1(w15), w10, small_sigma0(w2), w1);
        round(h, a, b, &mut c, d, e, f, &mut g, add2(k(K[49]), w1));
        w2 = add4(small_sigma1(w0), w11, small_sigma0(w3), w2);
        round(g, h, a, &mut b, c, d, e, &mut f, add2(k(K[50]), w2));
        w3 = add4(small_sigma1(w1), w12, small_sigma0(w4), w3);
        round(f, g, h, &mut a, b, c, d, &mut e, add2(k(K[51]), w3));
        w4 = add4(small_sigma1(w2), w13, small_sigma0(w5), w4);
        round(e, f, g, &mut h, a, b, c, &mut d, add2(k(K[52]), w4));
        w5 = add4(small_sigma1(w3), w14, small_sigma0(w6), w5);
        round(d, e, f, &mut g, h, a, b, &mut c, add2(k(K[53]), w5));
        w6 = add4(small_sigma1(w4), w15, small_sigma0(w7), w6);
        round(c, d, e, &mut f, g, h, a, &mut b, add2(k(K[54]), w6));
        w7 = add4(small_sigma1(w5), w0, small_sigma0(w8), w7);
        round(b, c, d, &mut e, f, g, h, &mut a, add2(k(K[55]), w7));
        w8 = add4(small_sigma1(w6), w1, small_sigma0(w9), w8);
        round(a, b, c, &mut d, e, f, g, &mut h, add2(k(K[56]), w8));
        w9 = add4(small_sigma1(w7), w2, small_sigma0(w10), w9);
        round(h, a, b, &mut c, d, e, f, &mut g, add2(k(K[57]), w9));
        w10 = add4(small_sigma1(w8), w3, small_sigma0(w11), w10);
        round(g, h, a, &mut b, c, d, e, &mut f, add2(k(K[58]), w10));
        w11 = add4(small_sigma1(w9), w4, small_sigma0(w12), w11);
        round(f, g, h, &mut a, b, c, d, &mut e, add2(k(K[59]), w11));
        w12 = add4(small_sigma1(w10), w5, small_sigma0(w13), w12);
        round(e, f, g, &mut h, a, b, c, &mut d, add2(k(K[60]), w12));
        w13 = add4(small_sigma1(w11), w6, small_sigma0(w14), w13);
        round(d, e, f, &mut g, h, a, b, &mut c, add2(k(K[61]), w13));
        w14 = add4(small_sigma1(w12), w7, small_sigma0(w15), w14);
        round(c, d, e, &mut f, g, h, a, &mut b, add2(k(K[62]), w14));
        w15 = add4(small_sigma1(w13), w8, small_sigma0(w0), w15);
        round(b, c, d, &mut e, f, g, h, &mut a, add2(k(K[63]), w15));

        // Add IV back
        a = add2(a, iv0);
        b = add2(b, iv1);
        c = add2(c, iv2);
        d = add2(d, iv3);
        e = add2(e, iv4);
        f = add2(f, iv5);
        g = add2(g, iv6);
        h = add2(h, iv7);

        // Write output
        let out = output.as_mut_ptr() as *mut u8;
        write4(out, 0, a, shuf_mask);
        write4(out, 4, b, shuf_mask);
        write4(out, 8, c, shuf_mask);
        write4(out, 12, d, shuf_mask);
        write4(out, 16, e, shuf_mask);
        write4(out, 20, f, shuf_mask);
        write4(out, 24, g, shuf_mask);
        write4(out, 28, h, shuf_mask);
    }

    /// Computes SHA256d on eight 64-byte inputs in parallel using AVX2 intrinsics.
    ///
    /// Based on Bitcoin Core's `sha256d64_avx2::Transform_8way`.
    // https://github.com/bitcoin/bitcoin/blob/master/src/crypto/sha256_avx2.cpp
    #[cfg(all(target_arch = "x86_64", any(feature = "std", feature = "cpufeatures")))]
    #[target_feature(enable = "avx2")]
    unsafe fn sha256d_64_avx2_8way(output: &mut [[u8; 32]; 8], input: &[[u8; 64]; 8]) {
        use core::arch::x86_64::*;

        #[inline(always)]
        unsafe fn k(x: u32) -> __m256i { _mm256_set1_epi32(x as i32) }
        #[inline(always)]
        unsafe fn add2(a: __m256i, b: __m256i) -> __m256i { _mm256_add_epi32(a, b) }
        #[inline(always)]
        unsafe fn add3(a: __m256i, b: __m256i, c: __m256i) -> __m256i { add2(add2(a, b), c) }
        #[inline(always)]
        unsafe fn add4(a: __m256i, b: __m256i, c: __m256i, d: __m256i) -> __m256i {
            add2(add2(a, b), add2(c, d))
        }
        #[inline(always)]
        unsafe fn add5(
            a: __m256i,
            b: __m256i,
            c: __m256i,
            d: __m256i,
            e: __m256i,
        ) -> __m256i {
            add2(add3(a, b, c), add2(d, e))
        }
        #[inline(always)]
        unsafe fn xor2(a: __m256i, b: __m256i) -> __m256i { _mm256_xor_si256(a, b) }
        #[inline(always)]
        unsafe fn xor3(a: __m256i, b: __m256i, c: __m256i) -> __m256i { xor2(xor2(a, b), c) }
        #[inline(always)]
        unsafe fn or(a: __m256i, b: __m256i) -> __m256i { _mm256_or_si256(a, b) }
        #[inline(always)]
        unsafe fn and(a: __m256i, b: __m256i) -> __m256i { _mm256_and_si256(a, b) }

        #[inline(always)]
        unsafe fn ch(x: __m256i, y: __m256i, z: __m256i) -> __m256i {
            xor2(z, and(x, xor2(y, z)))
        }
        #[inline(always)]
        unsafe fn maj(x: __m256i, y: __m256i, z: __m256i) -> __m256i {
            or(and(x, y), and(z, or(x, y)))
        }
        #[inline(always)]
        unsafe fn big_sigma0(x: __m256i) -> __m256i {
            xor3(
                or(_mm256_srli_epi32(x, 2), _mm256_slli_epi32(x, 30)),
                or(_mm256_srli_epi32(x, 13), _mm256_slli_epi32(x, 19)),
                or(_mm256_srli_epi32(x, 22), _mm256_slli_epi32(x, 10)),
            )
        }
        #[inline(always)]
        unsafe fn big_sigma1(x: __m256i) -> __m256i {
            xor3(
                or(_mm256_srli_epi32(x, 6), _mm256_slli_epi32(x, 26)),
                or(_mm256_srli_epi32(x, 11), _mm256_slli_epi32(x, 21)),
                or(_mm256_srli_epi32(x, 25), _mm256_slli_epi32(x, 7)),
            )
        }
        #[inline(always)]
        unsafe fn small_sigma0(x: __m256i) -> __m256i {
            xor3(
                or(_mm256_srli_epi32(x, 7), _mm256_slli_epi32(x, 25)),
                or(_mm256_srli_epi32(x, 18), _mm256_slli_epi32(x, 14)),
                _mm256_srli_epi32(x, 3),
            )
        }
        #[inline(always)]
        unsafe fn small_sigma1(x: __m256i) -> __m256i {
            xor3(
                or(_mm256_srli_epi32(x, 17), _mm256_slli_epi32(x, 15)),
                or(_mm256_srli_epi32(x, 19), _mm256_slli_epi32(x, 13)),
                _mm256_srli_epi32(x, 10),
            )
        }

        #[inline(always)]
        unsafe fn round(
            a: __m256i,
            b: __m256i,
            c: __m256i,
            d: &mut __m256i,
            e: __m256i,
            f: __m256i,
            g: __m256i,
            h: &mut __m256i,
            ki: __m256i,
        ) {
            let t1 = add4(*h, big_sigma1(e), ch(e, f, g), ki);
            let t2 = add2(big_sigma0(a), maj(a, b, c));
            *d = add2(*d, t1);
            *h = add2(t1, t2);
        }

        #[inline(always)]
        unsafe fn read8(base: *const u8, offset: usize, shuf: __m256i) -> __m256i {
            let w0 = (base.add(offset) as *const u32).read_unaligned();
            let w1 = (base.add(64 + offset) as *const u32).read_unaligned();
            let w2 = (base.add(128 + offset) as *const u32).read_unaligned();
            let w3 = (base.add(192 + offset) as *const u32).read_unaligned();
            let w4 = (base.add(256 + offset) as *const u32).read_unaligned();
            let w5 = (base.add(320 + offset) as *const u32).read_unaligned();
            let w6 = (base.add(384 + offset) as *const u32).read_unaligned();
            let w7 = (base.add(448 + offset) as *const u32).read_unaligned();
            let ret = _mm256_set_epi32(
                w0 as i32, w1 as i32, w2 as i32, w3 as i32,
                w4 as i32, w5 as i32, w6 as i32, w7 as i32,
            );
            _mm256_shuffle_epi8(ret, shuf)
        }

        #[inline(always)]
        unsafe fn write8(base: *mut u8, offset: usize, v: __m256i, shuf: __m256i) {
            let v = _mm256_shuffle_epi8(v, shuf);
            (base.add(offset) as *mut u32)
                .write_unaligned(_mm256_extract_epi32::<7>(v) as u32);
            (base.add(32 + offset) as *mut u32)
                .write_unaligned(_mm256_extract_epi32::<6>(v) as u32);
            (base.add(64 + offset) as *mut u32)
                .write_unaligned(_mm256_extract_epi32::<5>(v) as u32);
            (base.add(96 + offset) as *mut u32)
                .write_unaligned(_mm256_extract_epi32::<4>(v) as u32);
            (base.add(128 + offset) as *mut u32)
                .write_unaligned(_mm256_extract_epi32::<3>(v) as u32);
            (base.add(160 + offset) as *mut u32)
                .write_unaligned(_mm256_extract_epi32::<2>(v) as u32);
            (base.add(192 + offset) as *mut u32)
                .write_unaligned(_mm256_extract_epi32::<1>(v) as u32);
            (base.add(224 + offset) as *mut u32)
                .write_unaligned(_mm256_extract_epi32::<0>(v) as u32);
        }

        let shuf_mask = _mm256_set_epi32(
            0x0C0D0E0Fu32 as i32, 0x08090A0Bu32 as i32,
            0x04050607u32 as i32, 0x00010203u32 as i32,
            0x0C0D0E0Fu32 as i32, 0x08090A0Bu32 as i32,
            0x04050607u32 as i32, 0x00010203u32 as i32,
        );

        #[rustfmt::skip]
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
        ];

        #[rustfmt::skip]
        const MIDS: [u32; 64] = [
            0xc28a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf374,
            0x649b69c1, 0xf0fe4786, 0x0fe1edc6, 0x240cf254, 0x4fe9346f, 0x6cc984be, 0x61b9411e, 0x16f988fa,
            0xf2c65152, 0xa88e5a6d, 0xb019fc65, 0xb9d99ec7, 0x9a1231c3, 0xe70eeaa0, 0xfdb1232b, 0xc7353eb0,
            0x3069bad5, 0xcb976d5f, 0x5a0f118f, 0xdc1eeefd, 0x0a35b689, 0xde0b7a04, 0x58f4ca9d, 0xe15d5b16,
            0x007f3e86, 0x37088980, 0xa507ea32, 0x6fab9537, 0x17406110, 0x0d8cd6f1, 0xcdaa3b6d, 0xc0bbbe37,
            0x83613bda, 0xdb48a363, 0x0b02e931, 0x6fd15ca7, 0x521afaca, 0x31338431, 0x6ed41a95, 0x6d437890,
            0xc39c91f2, 0x9eccabbd, 0xb5c9a0e6, 0x532fb63c, 0xd2c741c6, 0x07237ea3, 0xa4954b68, 0x4c191d76,
        ];

        let inp = input.as_ptr() as *const u8;

        // SHA-256 initial hash value
        let iv0 = k(0x6a09e667);
        let iv1 = k(0xbb67ae85);
        let iv2 = k(0x3c6ef372);
        let iv3 = k(0xa54ff53a);
        let iv4 = k(0x510e527f);
        let iv5 = k(0x9b05688c);
        let iv6 = k(0x1f83d9ab);
        let iv7 = k(0x5be0cd19);

        // ---- Transform 1: SHA-256 of the 64-byte input blocks ----
        let mut a = iv0;
        let mut b = iv1;
        let mut c = iv2;
        let mut d = iv3;
        let mut e = iv4;
        let mut f = iv5;
        let mut g = iv6;
        let mut h = iv7;

        // Load message words
        let mut w0 = read8(inp, 4 * 0, shuf_mask);
        let mut w1 = read8(inp, 4 * 1, shuf_mask);
        let mut w2 = read8(inp, 4 * 2, shuf_mask);
        let mut w3 = read8(inp, 4 * 3, shuf_mask);
        let mut w4 = read8(inp, 4 * 4, shuf_mask);
        let mut w5 = read8(inp, 4 * 5, shuf_mask);
        let mut w6 = read8(inp, 4 * 6, shuf_mask);
        let mut w7 = read8(inp, 4 * 7, shuf_mask);
        let mut w8 = read8(inp, 4 * 8, shuf_mask);
        let mut w9 = read8(inp, 4 * 9, shuf_mask);
        let mut w10 = read8(inp, 4 * 10, shuf_mask);
        let mut w11 = read8(inp, 4 * 11, shuf_mask);
        let mut w12 = read8(inp, 4 * 12, shuf_mask);
        let mut w13 = read8(inp, 4 * 13, shuf_mask);
        let mut w14 = read8(inp, 4 * 14, shuf_mask);
        let mut w15 = read8(inp, 4 * 15, shuf_mask);

        // Rounds 0-15
        round(a, b, c, &mut d, e, f, g, &mut h, add2(k(K[0]), w0));
        round(h, a, b, &mut c, d, e, f, &mut g, add2(k(K[1]), w1));
        round(g, h, a, &mut b, c, d, e, &mut f, add2(k(K[2]), w2));
        round(f, g, h, &mut a, b, c, d, &mut e, add2(k(K[3]), w3));
        round(e, f, g, &mut h, a, b, c, &mut d, add2(k(K[4]), w4));
        round(d, e, f, &mut g, h, a, b, &mut c, add2(k(K[5]), w5));
        round(c, d, e, &mut f, g, h, a, &mut b, add2(k(K[6]), w6));
        round(b, c, d, &mut e, f, g, h, &mut a, add2(k(K[7]), w7));
        round(a, b, c, &mut d, e, f, g, &mut h, add2(k(K[8]), w8));
        round(h, a, b, &mut c, d, e, f, &mut g, add2(k(K[9]), w9));
        round(g, h, a, &mut b, c, d, e, &mut f, add2(k(K[10]), w10));
        round(f, g, h, &mut a, b, c, d, &mut e, add2(k(K[11]), w11));
        round(e, f, g, &mut h, a, b, c, &mut d, add2(k(K[12]), w12));
        round(d, e, f, &mut g, h, a, b, &mut c, add2(k(K[13]), w13));
        round(c, d, e, &mut f, g, h, a, &mut b, add2(k(K[14]), w14));
        round(b, c, d, &mut e, f, g, h, &mut a, add2(k(K[15]), w15));

        // Rounds 16-63 with message schedule
        w0 = add4(small_sigma1(w14), w9, small_sigma0(w1), w0);
        round(a, b, c, &mut d, e, f, g, &mut h, add2(k(K[16]), w0));
        w1 = add4(small_sigma1(w15), w10, small_sigma0(w2), w1);
        round(h, a, b, &mut c, d, e, f, &mut g, add2(k(K[17]), w1));
        w2 = add4(small_sigma1(w0), w11, small_sigma0(w3), w2);
        round(g, h, a, &mut b, c, d, e, &mut f, add2(k(K[18]), w2));
        w3 = add4(small_sigma1(w1), w12, small_sigma0(w4), w3);
        round(f, g, h, &mut a, b, c, d, &mut e, add2(k(K[19]), w3));
        w4 = add4(small_sigma1(w2), w13, small_sigma0(w5), w4);
        round(e, f, g, &mut h, a, b, c, &mut d, add2(k(K[20]), w4));
        w5 = add4(small_sigma1(w3), w14, small_sigma0(w6), w5);
        round(d, e, f, &mut g, h, a, b, &mut c, add2(k(K[21]), w5));
        w6 = add4(small_sigma1(w4), w15, small_sigma0(w7), w6);
        round(c, d, e, &mut f, g, h, a, &mut b, add2(k(K[22]), w6));
        w7 = add4(small_sigma1(w5), w0, small_sigma0(w8), w7);
        round(b, c, d, &mut e, f, g, h, &mut a, add2(k(K[23]), w7));
        w8 = add4(small_sigma1(w6), w1, small_sigma0(w9), w8);
        round(a, b, c, &mut d, e, f, g, &mut h, add2(k(K[24]), w8));
        w9 = add4(small_sigma1(w7), w2, small_sigma0(w10), w9);
        round(h, a, b, &mut c, d, e, f, &mut g, add2(k(K[25]), w9));
        w10 = add4(small_sigma1(w8), w3, small_sigma0(w11), w10);
        round(g, h, a, &mut b, c, d, e, &mut f, add2(k(K[26]), w10));
        w11 = add4(small_sigma1(w9), w4, small_sigma0(w12), w11);
        round(f, g, h, &mut a, b, c, d, &mut e, add2(k(K[27]), w11));
        w12 = add4(small_sigma1(w10), w5, small_sigma0(w13), w12);
        round(e, f, g, &mut h, a, b, c, &mut d, add2(k(K[28]), w12));
        w13 = add4(small_sigma1(w11), w6, small_sigma0(w14), w13);
        round(d, e, f, &mut g, h, a, b, &mut c, add2(k(K[29]), w13));
        w14 = add4(small_sigma1(w12), w7, small_sigma0(w15), w14);
        round(c, d, e, &mut f, g, h, a, &mut b, add2(k(K[30]), w14));
        w15 = add4(small_sigma1(w13), w8, small_sigma0(w0), w15);
        round(b, c, d, &mut e, f, g, h, &mut a, add2(k(K[31]), w15));
        w0 = add4(small_sigma1(w14), w9, small_sigma0(w1), w0);
        round(a, b, c, &mut d, e, f, g, &mut h, add2(k(K[32]), w0));
        w1 = add4(small_sigma1(w15), w10, small_sigma0(w2), w1);
        round(h, a, b, &mut c, d, e, f, &mut g, add2(k(K[33]), w1));
        w2 = add4(small_sigma1(w0), w11, small_sigma0(w3), w2);
        round(g, h, a, &mut b, c, d, e, &mut f, add2(k(K[34]), w2));
        w3 = add4(small_sigma1(w1), w12, small_sigma0(w4), w3);
        round(f, g, h, &mut a, b, c, d, &mut e, add2(k(K[35]), w3));
        w4 = add4(small_sigma1(w2), w13, small_sigma0(w5), w4);
        round(e, f, g, &mut h, a, b, c, &mut d, add2(k(K[36]), w4));
        w5 = add4(small_sigma1(w3), w14, small_sigma0(w6), w5);
        round(d, e, f, &mut g, h, a, b, &mut c, add2(k(K[37]), w5));
        w6 = add4(small_sigma1(w4), w15, small_sigma0(w7), w6);
        round(c, d, e, &mut f, g, h, a, &mut b, add2(k(K[38]), w6));
        w7 = add4(small_sigma1(w5), w0, small_sigma0(w8), w7);
        round(b, c, d, &mut e, f, g, h, &mut a, add2(k(K[39]), w7));
        w8 = add4(small_sigma1(w6), w1, small_sigma0(w9), w8);
        round(a, b, c, &mut d, e, f, g, &mut h, add2(k(K[40]), w8));
        w9 = add4(small_sigma1(w7), w2, small_sigma0(w10), w9);
        round(h, a, b, &mut c, d, e, f, &mut g, add2(k(K[41]), w9));
        w10 = add4(small_sigma1(w8), w3, small_sigma0(w11), w10);
        round(g, h, a, &mut b, c, d, e, &mut f, add2(k(K[42]), w10));
        w11 = add4(small_sigma1(w9), w4, small_sigma0(w12), w11);
        round(f, g, h, &mut a, b, c, d, &mut e, add2(k(K[43]), w11));
        w12 = add4(small_sigma1(w10), w5, small_sigma0(w13), w12);
        round(e, f, g, &mut h, a, b, c, &mut d, add2(k(K[44]), w12));
        w13 = add4(small_sigma1(w11), w6, small_sigma0(w14), w13);
        round(d, e, f, &mut g, h, a, b, &mut c, add2(k(K[45]), w13));
        w14 = add4(small_sigma1(w12), w7, small_sigma0(w15), w14);
        round(c, d, e, &mut f, g, h, a, &mut b, add2(k(K[46]), w14));
        w15 = add4(small_sigma1(w13), w8, small_sigma0(w0), w15);
        round(b, c, d, &mut e, f, g, h, &mut a, add2(k(K[47]), w15));
        w0 = add4(small_sigma1(w14), w9, small_sigma0(w1), w0);
        round(a, b, c, &mut d, e, f, g, &mut h, add2(k(K[48]), w0));
        w1 = add4(small_sigma1(w15), w10, small_sigma0(w2), w1);
        round(h, a, b, &mut c, d, e, f, &mut g, add2(k(K[49]), w1));
        w2 = add4(small_sigma1(w0), w11, small_sigma0(w3), w2);
        round(g, h, a, &mut b, c, d, e, &mut f, add2(k(K[50]), w2));
        w3 = add4(small_sigma1(w1), w12, small_sigma0(w4), w3);
        round(f, g, h, &mut a, b, c, d, &mut e, add2(k(K[51]), w3));
        w4 = add4(small_sigma1(w2), w13, small_sigma0(w5), w4);
        round(e, f, g, &mut h, a, b, c, &mut d, add2(k(K[52]), w4));
        w5 = add4(small_sigma1(w3), w14, small_sigma0(w6), w5);
        round(d, e, f, &mut g, h, a, b, &mut c, add2(k(K[53]), w5));
        w6 = add4(small_sigma1(w4), w15, small_sigma0(w7), w6);
        round(c, d, e, &mut f, g, h, a, &mut b, add2(k(K[54]), w6));
        w7 = add4(small_sigma1(w5), w0, small_sigma0(w8), w7);
        round(b, c, d, &mut e, f, g, h, &mut a, add2(k(K[55]), w7));
        w8 = add4(small_sigma1(w6), w1, small_sigma0(w9), w8);
        round(a, b, c, &mut d, e, f, g, &mut h, add2(k(K[56]), w8));
        w9 = add4(small_sigma1(w7), w2, small_sigma0(w10), w9);
        round(h, a, b, &mut c, d, e, f, &mut g, add2(k(K[57]), w9));
        w10 = add4(small_sigma1(w8), w3, small_sigma0(w11), w10);
        round(g, h, a, &mut b, c, d, e, &mut f, add2(k(K[58]), w10));
        w11 = add4(small_sigma1(w9), w4, small_sigma0(w12), w11);
        round(f, g, h, &mut a, b, c, d, &mut e, add2(k(K[59]), w11));
        w12 = add4(small_sigma1(w10), w5, small_sigma0(w13), w12);
        round(e, f, g, &mut h, a, b, c, &mut d, add2(k(K[60]), w12));
        w13 = add4(small_sigma1(w11), w6, small_sigma0(w14), w13);
        round(d, e, f, &mut g, h, a, b, &mut c, add2(k(K[61]), w13));
        w14 = add4(small_sigma1(w12), w7, small_sigma0(w15), w14);
        round(c, d, e, &mut f, g, h, a, &mut b, add2(k(K[62]), w14));
        w15 = add4(small_sigma1(w13), w8, small_sigma0(w0), w15);
        round(b, c, d, &mut e, f, g, h, &mut a, add2(k(K[63]), w15));

        // Add IV back
        a = add2(a, iv0);
        b = add2(b, iv1);
        c = add2(c, iv2);
        d = add2(d, iv3);
        e = add2(e, iv4);
        f = add2(f, iv5);
        g = add2(g, iv6);
        h = add2(h, iv7);

        // Save Transform 1 output
        let t0 = a;
        let t1 = b;
        let t2 = c;
        let t3 = d;
        let t4 = e;
        let t5 = f;
        let t6 = g;
        let t7 = h;

        // ---- Transform 2: SHA-256 of midstate padding ----
        // State carries over from Transform 1 (NOT reset to IV).
        // MIDS = precomputed K + W for the padding block after a 64-byte input.

        // All 64 rounds with MIDS constants (precomputed K + padding)
        round(a, b, c, &mut d, e, f, g, &mut h, k(MIDS[0]));
        round(h, a, b, &mut c, d, e, f, &mut g, k(MIDS[1]));
        round(g, h, a, &mut b, c, d, e, &mut f, k(MIDS[2]));
        round(f, g, h, &mut a, b, c, d, &mut e, k(MIDS[3]));
        round(e, f, g, &mut h, a, b, c, &mut d, k(MIDS[4]));
        round(d, e, f, &mut g, h, a, b, &mut c, k(MIDS[5]));
        round(c, d, e, &mut f, g, h, a, &mut b, k(MIDS[6]));
        round(b, c, d, &mut e, f, g, h, &mut a, k(MIDS[7]));
        round(a, b, c, &mut d, e, f, g, &mut h, k(MIDS[8]));
        round(h, a, b, &mut c, d, e, f, &mut g, k(MIDS[9]));
        round(g, h, a, &mut b, c, d, e, &mut f, k(MIDS[10]));
        round(f, g, h, &mut a, b, c, d, &mut e, k(MIDS[11]));
        round(e, f, g, &mut h, a, b, c, &mut d, k(MIDS[12]));
        round(d, e, f, &mut g, h, a, b, &mut c, k(MIDS[13]));
        round(c, d, e, &mut f, g, h, a, &mut b, k(MIDS[14]));
        round(b, c, d, &mut e, f, g, h, &mut a, k(MIDS[15]));
        round(a, b, c, &mut d, e, f, g, &mut h, k(MIDS[16]));
        round(h, a, b, &mut c, d, e, f, &mut g, k(MIDS[17]));
        round(g, h, a, &mut b, c, d, e, &mut f, k(MIDS[18]));
        round(f, g, h, &mut a, b, c, d, &mut e, k(MIDS[19]));
        round(e, f, g, &mut h, a, b, c, &mut d, k(MIDS[20]));
        round(d, e, f, &mut g, h, a, b, &mut c, k(MIDS[21]));
        round(c, d, e, &mut f, g, h, a, &mut b, k(MIDS[22]));
        round(b, c, d, &mut e, f, g, h, &mut a, k(MIDS[23]));
        round(a, b, c, &mut d, e, f, g, &mut h, k(MIDS[24]));
        round(h, a, b, &mut c, d, e, f, &mut g, k(MIDS[25]));
        round(g, h, a, &mut b, c, d, e, &mut f, k(MIDS[26]));
        round(f, g, h, &mut a, b, c, d, &mut e, k(MIDS[27]));
        round(e, f, g, &mut h, a, b, c, &mut d, k(MIDS[28]));
        round(d, e, f, &mut g, h, a, b, &mut c, k(MIDS[29]));
        round(c, d, e, &mut f, g, h, a, &mut b, k(MIDS[30]));
        round(b, c, d, &mut e, f, g, h, &mut a, k(MIDS[31]));
        round(a, b, c, &mut d, e, f, g, &mut h, k(MIDS[32]));
        round(h, a, b, &mut c, d, e, f, &mut g, k(MIDS[33]));
        round(g, h, a, &mut b, c, d, e, &mut f, k(MIDS[34]));
        round(f, g, h, &mut a, b, c, d, &mut e, k(MIDS[35]));
        round(e, f, g, &mut h, a, b, c, &mut d, k(MIDS[36]));
        round(d, e, f, &mut g, h, a, b, &mut c, k(MIDS[37]));
        round(c, d, e, &mut f, g, h, a, &mut b, k(MIDS[38]));
        round(b, c, d, &mut e, f, g, h, &mut a, k(MIDS[39]));
        round(a, b, c, &mut d, e, f, g, &mut h, k(MIDS[40]));
        round(h, a, b, &mut c, d, e, f, &mut g, k(MIDS[41]));
        round(g, h, a, &mut b, c, d, e, &mut f, k(MIDS[42]));
        round(f, g, h, &mut a, b, c, d, &mut e, k(MIDS[43]));
        round(e, f, g, &mut h, a, b, c, &mut d, k(MIDS[44]));
        round(d, e, f, &mut g, h, a, b, &mut c, k(MIDS[45]));
        round(c, d, e, &mut f, g, h, a, &mut b, k(MIDS[46]));
        round(b, c, d, &mut e, f, g, h, &mut a, k(MIDS[47]));
        round(a, b, c, &mut d, e, f, g, &mut h, k(MIDS[48]));
        round(h, a, b, &mut c, d, e, f, &mut g, k(MIDS[49]));
        round(g, h, a, &mut b, c, d, e, &mut f, k(MIDS[50]));
        round(f, g, h, &mut a, b, c, d, &mut e, k(MIDS[51]));
        round(e, f, g, &mut h, a, b, c, &mut d, k(MIDS[52]));
        round(d, e, f, &mut g, h, a, b, &mut c, k(MIDS[53]));
        round(c, d, e, &mut f, g, h, a, &mut b, k(MIDS[54]));
        round(b, c, d, &mut e, f, g, h, &mut a, k(MIDS[55]));
        round(a, b, c, &mut d, e, f, g, &mut h, k(MIDS[56]));
        round(h, a, b, &mut c, d, e, f, &mut g, k(MIDS[57]));
        round(g, h, a, &mut b, c, d, e, &mut f, k(MIDS[58]));
        round(f, g, h, &mut a, b, c, d, &mut e, k(MIDS[59]));
        round(e, f, g, &mut h, a, b, c, &mut d, k(MIDS[60]));
        round(d, e, f, &mut g, h, a, b, &mut c, k(MIDS[61]));
        round(c, d, e, &mut f, g, h, a, &mut b, k(MIDS[62]));
        round(b, c, d, &mut e, f, g, h, &mut a, k(MIDS[63]));

        // Add Transform 1 output
        a = add2(a, t0);
        b = add2(b, t1);
        c = add2(c, t2);
        d = add2(d, t3);
        e = add2(e, t4);
        f = add2(f, t5);
        g = add2(g, t6);
        h = add2(h, t7);

        // Save w0-w7 for Transform 3
        w0 = a;
        w1 = b;
        w2 = c;
        w3 = d;
        w4 = e;
        w5 = f;
        w6 = g;
        w7 = h;

        // ---- Transform 3: Second SHA-256 hash (hash of the hash) ----
        a = iv0;
        b = iv1;
        c = iv2;
        d = iv3;
        e = iv4;
        f = iv5;
        g = iv6;
        h = iv7;

        // Rounds 0-7: hash of Transform 2 output
        round(a, b, c, &mut d, e, f, g, &mut h, add2(k(K[0]), w0));
        round(h, a, b, &mut c, d, e, f, &mut g, add2(k(K[1]), w1));
        round(g, h, a, &mut b, c, d, e, &mut f, add2(k(K[2]), w2));
        round(f, g, h, &mut a, b, c, d, &mut e, add2(k(K[3]), w3));
        round(e, f, g, &mut h, a, b, c, &mut d, add2(k(K[4]), w4));
        round(d, e, f, &mut g, h, a, b, &mut c, add2(k(K[5]), w5));
        round(c, d, e, &mut f, g, h, a, &mut b, add2(k(K[6]), w6));
        round(b, c, d, &mut e, f, g, h, &mut a, add2(k(K[7]), w7));

        // Rounds 8-15: precomputed K + padding constants
        round(a, b, c, &mut d, e, f, g, &mut h, k(0x5807aa98));
        round(h, a, b, &mut c, d, e, f, &mut g, k(0x12835b01));
        round(g, h, a, &mut b, c, d, e, &mut f, k(0x243185be));
        round(f, g, h, &mut a, b, c, d, &mut e, k(0x550c7dc3));
        round(e, f, g, &mut h, a, b, c, &mut d, k(0x72be5d74));
        round(d, e, f, &mut g, h, a, b, &mut c, k(0x80deb1fe));
        round(c, d, e, &mut f, g, h, a, &mut b, k(0x9bdc06a7));
        round(b, c, d, &mut e, f, g, h, &mut a, k(0xc19bf274));

        // Rounds 16-31: message schedule with padding constants mixed in.
        // w0-w7 are modified in place (w0→w16, w1→w17, etc.) so that
        // subsequent sigma1/sigma0 calls reference updated values.
        w0 = add2(w0, small_sigma0(w1));
        round(a, b, c, &mut d, e, f, g, &mut h, add2(k(K[16]), w0));
        w1 = add3(w1, k(0x00a00000), small_sigma0(w2));
        round(h, a, b, &mut c, d, e, f, &mut g, add2(k(K[17]), w1));
        w2 = add3(w2, small_sigma1(w0), small_sigma0(w3));
        round(g, h, a, &mut b, c, d, e, &mut f, add2(k(K[18]), w2));
        w3 = add3(w3, small_sigma1(w1), small_sigma0(w4));
        round(f, g, h, &mut a, b, c, d, &mut e, add2(k(K[19]), w3));
        w4 = add3(w4, small_sigma1(w2), small_sigma0(w5));
        round(e, f, g, &mut h, a, b, c, &mut d, add2(k(K[20]), w4));
        w5 = add3(w5, small_sigma1(w3), small_sigma0(w6));
        round(d, e, f, &mut g, h, a, b, &mut c, add2(k(K[21]), w5));
        w6 = add4(w6, small_sigma1(w4), k(0x00000100), small_sigma0(w7));
        round(c, d, e, &mut f, g, h, a, &mut b, add2(k(K[22]), w6));
        w7 = add4(w7, small_sigma1(w5), w0, k(0x11002000));
        round(b, c, d, &mut e, f, g, h, &mut a, add2(k(K[23]), w7));
        w8 = add3(k(0x80000000), small_sigma1(w6), w1);
        round(a, b, c, &mut d, e, f, g, &mut h, add2(k(K[24]), w8));
        w9 = add2(small_sigma1(w7), w2);
        round(h, a, b, &mut c, d, e, f, &mut g, add2(k(K[25]), w9));
        w10 = add2(small_sigma1(w8), w3);
        round(g, h, a, &mut b, c, d, e, &mut f, add2(k(K[26]), w10));
        w11 = add2(small_sigma1(w9), w4);
        round(f, g, h, &mut a, b, c, d, &mut e, add2(k(K[27]), w11));
        w12 = add2(small_sigma1(w10), w5);
        round(e, f, g, &mut h, a, b, c, &mut d, add2(k(K[28]), w12));
        w13 = add2(small_sigma1(w11), w6);
        round(d, e, f, &mut g, h, a, b, &mut c, add2(k(K[29]), w13));
        w14 = add3(small_sigma1(w12), w7, k(0x00400022));
        round(c, d, e, &mut f, g, h, a, &mut b, add2(k(K[30]), w14));
        w15 = add4(k(0x00000100), small_sigma1(w13), w8, small_sigma0(w0));
        round(b, c, d, &mut e, f, g, h, &mut a, add2(k(K[31]), w15));

        // Rounds 32-63: full message schedule
        w0 = add4(small_sigma1(w14), w9, small_sigma0(w1), w0);
        round(a, b, c, &mut d, e, f, g, &mut h, add2(k(K[32]), w0));
        w1 = add4(small_sigma1(w15), w10, small_sigma0(w2), w1);
        round(h, a, b, &mut c, d, e, f, &mut g, add2(k(K[33]), w1));
        w2 = add4(small_sigma1(w0), w11, small_sigma0(w3), w2);
        round(g, h, a, &mut b, c, d, e, &mut f, add2(k(K[34]), w2));
        w3 = add4(small_sigma1(w1), w12, small_sigma0(w4), w3);
        round(f, g, h, &mut a, b, c, d, &mut e, add2(k(K[35]), w3));
        w4 = add4(small_sigma1(w2), w13, small_sigma0(w5), w4);
        round(e, f, g, &mut h, a, b, c, &mut d, add2(k(K[36]), w4));
        w5 = add4(small_sigma1(w3), w14, small_sigma0(w6), w5);
        round(d, e, f, &mut g, h, a, b, &mut c, add2(k(K[37]), w5));
        w6 = add4(small_sigma1(w4), w15, small_sigma0(w7), w6);
        round(c, d, e, &mut f, g, h, a, &mut b, add2(k(K[38]), w6));
        w7 = add4(small_sigma1(w5), w0, small_sigma0(w8), w7);
        round(b, c, d, &mut e, f, g, h, &mut a, add2(k(K[39]), w7));
        w8 = add4(small_sigma1(w6), w1, small_sigma0(w9), w8);
        round(a, b, c, &mut d, e, f, g, &mut h, add2(k(K[40]), w8));
        w9 = add4(small_sigma1(w7), w2, small_sigma0(w10), w9);
        round(h, a, b, &mut c, d, e, f, &mut g, add2(k(K[41]), w9));
        w10 = add4(small_sigma1(w8), w3, small_sigma0(w11), w10);
        round(g, h, a, &mut b, c, d, e, &mut f, add2(k(K[42]), w10));
        w11 = add4(small_sigma1(w9), w4, small_sigma0(w12), w11);
        round(f, g, h, &mut a, b, c, d, &mut e, add2(k(K[43]), w11));
        w12 = add4(small_sigma1(w10), w5, small_sigma0(w13), w12);
        round(e, f, g, &mut h, a, b, c, &mut d, add2(k(K[44]), w12));
        w13 = add4(small_sigma1(w11), w6, small_sigma0(w14), w13);
        round(d, e, f, &mut g, h, a, b, &mut c, add2(k(K[45]), w13));
        w14 = add4(small_sigma1(w12), w7, small_sigma0(w15), w14);
        round(c, d, e, &mut f, g, h, a, &mut b, add2(k(K[46]), w14));
        w15 = add4(small_sigma1(w13), w8, small_sigma0(w0), w15);
        round(b, c, d, &mut e, f, g, h, &mut a, add2(k(K[47]), w15));
        w0 = add4(small_sigma1(w14), w9, small_sigma0(w1), w0);
        round(a, b, c, &mut d, e, f, g, &mut h, add2(k(K[48]), w0));
        w1 = add4(small_sigma1(w15), w10, small_sigma0(w2), w1);
        round(h, a, b, &mut c, d, e, f, &mut g, add2(k(K[49]), w1));
        w2 = add4(small_sigma1(w0), w11, small_sigma0(w3), w2);
        round(g, h, a, &mut b, c, d, e, &mut f, add2(k(K[50]), w2));
        w3 = add4(small_sigma1(w1), w12, small_sigma0(w4), w3);
        round(f, g, h, &mut a, b, c, d, &mut e, add2(k(K[51]), w3));
        w4 = add4(small_sigma1(w2), w13, small_sigma0(w5), w4);
        round(e, f, g, &mut h, a, b, c, &mut d, add2(k(K[52]), w4));
        w5 = add4(small_sigma1(w3), w14, small_sigma0(w6), w5);
        round(d, e, f, &mut g, h, a, b, &mut c, add2(k(K[53]), w5));
        w6 = add4(small_sigma1(w4), w15, small_sigma0(w7), w6);
        round(c, d, e, &mut f, g, h, a, &mut b, add2(k(K[54]), w6));
        w7 = add4(small_sigma1(w5), w0, small_sigma0(w8), w7);
        round(b, c, d, &mut e, f, g, h, &mut a, add2(k(K[55]), w7));
        w8 = add4(small_sigma1(w6), w1, small_sigma0(w9), w8);
        round(a, b, c, &mut d, e, f, g, &mut h, add2(k(K[56]), w8));
        w9 = add4(small_sigma1(w7), w2, small_sigma0(w10), w9);
        round(h, a, b, &mut c, d, e, f, &mut g, add2(k(K[57]), w9));
        w10 = add4(small_sigma1(w8), w3, small_sigma0(w11), w10);
        round(g, h, a, &mut b, c, d, e, &mut f, add2(k(K[58]), w10));
        w11 = add4(small_sigma1(w9), w4, small_sigma0(w12), w11);
        round(f, g, h, &mut a, b, c, d, &mut e, add2(k(K[59]), w11));
        w12 = add4(small_sigma1(w10), w5, small_sigma0(w13), w12);
        round(e, f, g, &mut h, a, b, c, &mut d, add2(k(K[60]), w12));
        w13 = add4(small_sigma1(w11), w6, small_sigma0(w14), w13);
        round(d, e, f, &mut g, h, a, b, &mut c, add2(k(K[61]), w13));
        w14 = add4(small_sigma1(w12), w7, small_sigma0(w15), w14);
        round(c, d, e, &mut f, g, h, a, &mut b, add2(k(K[62]), w14));
        w15 = add4(small_sigma1(w13), w8, small_sigma0(w0), w15);
        round(b, c, d, &mut e, f, g, h, &mut a, add2(k(K[63]), w15));

        // Add IV back
        a = add2(a, iv0);
        b = add2(b, iv1);
        c = add2(c, iv2);
        d = add2(d, iv3);
        e = add2(e, iv4);
        f = add2(f, iv5);
        g = add2(g, iv6);
        h = add2(h, iv7);

        // Write output
        let out = output.as_mut_ptr() as *mut u8;
        write8(out, 0, a, shuf_mask);
        write8(out, 4, b, shuf_mask);
        write8(out, 8, c, shuf_mask);
        write8(out, 12, d, shuf_mask);
        write8(out, 16, e, shuf_mask);
        write8(out, 20, f, shuf_mask);
        write8(out, 24, g, shuf_mask);
        write8(out, 28, h, shuf_mask);
    }

    #[cfg(all(target_arch = "aarch64", any(feature = "std", feature = "cpufeatures")))]
    #[target_feature(enable = "sha2")]
    unsafe fn sha256d_64_arm_2way(output: &mut [[u8; 32]; 2], input: &[[u8; 64]; 2]) {
        // Based on Bitcoin Core's sha256d64_arm_shani::Transform_2way
        // https://github.com/bitcoin/bitcoin/blob/master/src/crypto/sha256_arm_shani.cpp#L200-L895
        use core::arch::aarch64::vst1q_u8;

        // initial state
        #[rustfmt::skip]
        const INIT: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
        ];

        // SHA256 round constants
        #[rustfmt::skip]
        const K: [u32; 64] = [
            0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5,
            0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
            0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,
            0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
            0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC,
            0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
            0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7,
            0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
            0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13,
            0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
            0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3,
            0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
            0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5,
            0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
            0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,
            0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
        ];

        // Precomputed W[i] + K[i] for the 2nd transform (padding block).
        #[rustfmt::skip]
        const MIDS: [u32; 64] = [
            0xc28a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
            0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
            0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf374,
            0x649b69c1, 0xf0fe4786, 0x0fe1edc6, 0x240cf254,
            0x4fe9346f, 0x6cc984be, 0x61b9411e, 0x16f988fa,
            0xf2c65152, 0xa88e5a6d, 0xb019fc65, 0xb9d99ec7,
            0x9a1231c3, 0xe70eeaa0, 0xfdb1232b, 0xc7353eb0,
            0x3069bad5, 0xcb976d5f, 0x5a0f118f, 0xdc1eeefd,
            0x0a35b689, 0xde0b7a04, 0x58f4ca9d, 0xe15d5b16,
            0x007f3e86, 0x37088980, 0xa507ea32, 0x6fab9537,
            0x17406110, 0x0d8cd6f1, 0xcdaa3b6d, 0xc0bbbe37,
            0x83613bda, 0xdb48a363, 0x0b02e931, 0x6fd15ca7,
            0x521afaca, 0x31338431, 0x6ed41a95, 0x6d437890,
            0xc39c91f2, 0x9eccabbd, 0xb5c9a0e6, 0x532fb63c,
            0xd2c741c6, 0x07237ea3, 0xa4954b68, 0x4c191d76
        ];

        // Precomputed values for Transform 3 rounds 9-16.
        // FINS[0..3]: msg2 + K[8..11]
        // FINS[4..7]: vsha256su0q_u32(msg2, msg3)
        // FINS[8..11]: msg2 + K[12..15]
        #[rustfmt::skip]
        const FINS: [u32; 12] = [
            0x5807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
            0x80000000, 0x00000000, 0x00000000, 0x00000000,
            0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf274,
        ];

        // Padding processed in the 3rd transform (byteswapped).
        const FINAL: [u32; 8] = [0x80000000, 0, 0, 0, 0, 0, 0, 0x100];

        let (mut state0_a, mut state0_b, mut state1_a, mut state1_b);
        let (abcd_save_a, abcd_save_b, efgh_save_a, efgh_save_b);

        #[rustfmt::skip]
        let (mut msg0_a, mut msg0_b, mut msg1_a, mut msg1_b, mut msg2_a, mut msg2_b, mut msg3_a, mut msg3_b);
        let (mut tmp0_a, mut tmp0_b, mut tmp2_a, mut tmp2_b, mut tmp);

        // Load state
        state0_a = vld1q_u32(INIT.as_ptr().add(0));
        state0_b = state0_a;
        state1_a = vld1q_u32(INIT.as_ptr().add(4));
        state1_b = state1_a;

        // Load message
        msg0_a = vld1q_u32(input[0].as_ptr().add(0).cast::<u32>());
        msg1_a = vld1q_u32(input[0].as_ptr().add(16).cast::<u32>());
        msg2_a = vld1q_u32(input[0].as_ptr().add(32).cast::<u32>());
        msg3_a = vld1q_u32(input[0].as_ptr().add(48).cast::<u32>());
        msg0_b = vld1q_u32(input[1].as_ptr().add(0).cast::<u32>());
        msg1_b = vld1q_u32(input[1].as_ptr().add(16).cast::<u32>());
        msg2_b = vld1q_u32(input[1].as_ptr().add(32).cast::<u32>());
        msg3_b = vld1q_u32(input[1].as_ptr().add(48).cast::<u32>());

        // Reverse for little endian
        msg0_a = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(msg0_a)));
        msg1_a = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(msg1_a)));
        msg2_a = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(msg2_a)));
        msg3_a = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(msg3_a)));
        msg0_b = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(msg0_b)));
        msg1_b = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(msg1_b)));
        msg2_b = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(msg2_b)));
        msg3_b = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(msg3_b)));

        // Transform 1: Rounds 1-4
        tmp = vld1q_u32(K.as_ptr().add(0));
        tmp0_a = vaddq_u32(msg0_a, tmp);
        tmp0_b = vaddq_u32(msg0_b, tmp);
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        msg0_a = vsha256su0q_u32(msg0_a, msg1_a);
        msg0_b = vsha256su0q_u32(msg0_b, msg1_b);
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp0_a);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp0_b);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp0_a);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp0_b);
        msg0_a = vsha256su1q_u32(msg0_a, msg2_a, msg3_a);
        msg0_b = vsha256su1q_u32(msg0_b, msg2_b, msg3_b);

        // Transform 1: Rounds 5-8
        tmp = vld1q_u32(K.as_ptr().add(4));
        tmp0_a = vaddq_u32(msg1_a, tmp);
        tmp0_b = vaddq_u32(msg1_b, tmp);
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        msg1_a = vsha256su0q_u32(msg1_a, msg2_a);
        msg1_b = vsha256su0q_u32(msg1_b, msg2_b);
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp0_a);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp0_b);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp0_a);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp0_b);
        msg1_a = vsha256su1q_u32(msg1_a, msg3_a, msg0_a);
        msg1_b = vsha256su1q_u32(msg1_b, msg3_b, msg0_b);

        // Transform 1: Rounds 9-12
        tmp = vld1q_u32(K.as_ptr().add(8));
        tmp0_a = vaddq_u32(msg2_a, tmp);
        tmp0_b = vaddq_u32(msg2_b, tmp);
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        msg2_a = vsha256su0q_u32(msg2_a, msg3_a);
        msg2_b = vsha256su0q_u32(msg2_b, msg3_b);
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp0_a);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp0_b);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp0_a);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp0_b);
        msg2_a = vsha256su1q_u32(msg2_a, msg0_a, msg1_a);
        msg2_b = vsha256su1q_u32(msg2_b, msg0_b, msg1_b);

        // Transform 1: Rounds 13-16
        tmp = vld1q_u32(K.as_ptr().add(12));
        tmp0_a = vaddq_u32(msg3_a, tmp);
        tmp0_b = vaddq_u32(msg3_b, tmp);
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        msg3_a = vsha256su0q_u32(msg3_a, msg0_a);
        msg3_b = vsha256su0q_u32(msg3_b, msg0_b);
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp0_a);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp0_b);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp0_a);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp0_b);
        msg3_a = vsha256su1q_u32(msg3_a, msg1_a, msg2_a);
        msg3_b = vsha256su1q_u32(msg3_b, msg1_b, msg2_b);

        // Transform 1: Rounds 17-20
        tmp = vld1q_u32(K.as_ptr().add(16));
        tmp0_a = vaddq_u32(msg0_a, tmp);
        tmp0_b = vaddq_u32(msg0_b, tmp);
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        msg0_a = vsha256su0q_u32(msg0_a, msg1_a);
        msg0_b = vsha256su0q_u32(msg0_b, msg1_b);
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp0_a);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp0_b);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp0_a);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp0_b);
        msg0_a = vsha256su1q_u32(msg0_a, msg2_a, msg3_a);
        msg0_b = vsha256su1q_u32(msg0_b, msg2_b, msg3_b);

        // Transform 1: Rounds 21-24
        tmp = vld1q_u32(K.as_ptr().add(20));
        tmp0_a = vaddq_u32(msg1_a, tmp);
        tmp0_b = vaddq_u32(msg1_b, tmp);
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        msg1_a = vsha256su0q_u32(msg1_a, msg2_a);
        msg1_b = vsha256su0q_u32(msg1_b, msg2_b);
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp0_a);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp0_b);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp0_a);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp0_b);
        msg1_a = vsha256su1q_u32(msg1_a, msg3_a, msg0_a);
        msg1_b = vsha256su1q_u32(msg1_b, msg3_b, msg0_b);

        // Transform 1: Rounds 25-28
        tmp = vld1q_u32(K.as_ptr().add(24));
        tmp0_a = vaddq_u32(msg2_a, tmp);
        tmp0_b = vaddq_u32(msg2_b, tmp);
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        msg2_a = vsha256su0q_u32(msg2_a, msg3_a);
        msg2_b = vsha256su0q_u32(msg2_b, msg3_b);
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp0_a);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp0_b);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp0_a);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp0_b);
        msg2_a = vsha256su1q_u32(msg2_a, msg0_a, msg1_a);
        msg2_b = vsha256su1q_u32(msg2_b, msg0_b, msg1_b);

        // Transform 1: Rounds 29-32
        tmp = vld1q_u32(K.as_ptr().add(28));
        tmp0_a = vaddq_u32(msg3_a, tmp);
        tmp0_b = vaddq_u32(msg3_b, tmp);
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        msg3_a = vsha256su0q_u32(msg3_a, msg0_a);
        msg3_b = vsha256su0q_u32(msg3_b, msg0_b);
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp0_a);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp0_b);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp0_a);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp0_b);
        msg3_a = vsha256su1q_u32(msg3_a, msg1_a, msg2_a);
        msg3_b = vsha256su1q_u32(msg3_b, msg1_b, msg2_b);

        // Transform 1: Rounds 33-36
        tmp = vld1q_u32(K.as_ptr().add(32));
        tmp0_a = vaddq_u32(msg0_a, tmp);
        tmp0_b = vaddq_u32(msg0_b, tmp);
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        msg0_a = vsha256su0q_u32(msg0_a, msg1_a);
        msg0_b = vsha256su0q_u32(msg0_b, msg1_b);
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp0_a);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp0_b);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp0_a);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp0_b);
        msg0_a = vsha256su1q_u32(msg0_a, msg2_a, msg3_a);
        msg0_b = vsha256su1q_u32(msg0_b, msg2_b, msg3_b);

        // Transform 1: Rounds 37-40
        tmp = vld1q_u32(K.as_ptr().add(36));
        tmp0_a = vaddq_u32(msg1_a, tmp);
        tmp0_b = vaddq_u32(msg1_b, tmp);
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        msg1_a = vsha256su0q_u32(msg1_a, msg2_a);
        msg1_b = vsha256su0q_u32(msg1_b, msg2_b);
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp0_a);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp0_b);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp0_a);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp0_b);
        msg1_a = vsha256su1q_u32(msg1_a, msg3_a, msg0_a);
        msg1_b = vsha256su1q_u32(msg1_b, msg3_b, msg0_b);

        // Transform 1: Rounds 41-44
        tmp = vld1q_u32(K.as_ptr().add(40));
        tmp0_a = vaddq_u32(msg2_a, tmp);
        tmp0_b = vaddq_u32(msg2_b, tmp);
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        msg2_a = vsha256su0q_u32(msg2_a, msg3_a);
        msg2_b = vsha256su0q_u32(msg2_b, msg3_b);
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp0_a);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp0_b);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp0_a);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp0_b);
        msg2_a = vsha256su1q_u32(msg2_a, msg0_a, msg1_a);
        msg2_b = vsha256su1q_u32(msg2_b, msg0_b, msg1_b);

        // Transform 1: Rounds 45-48
        tmp = vld1q_u32(K.as_ptr().add(44));
        tmp0_a = vaddq_u32(msg3_a, tmp);
        tmp0_b = vaddq_u32(msg3_b, tmp);
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        msg3_a = vsha256su0q_u32(msg3_a, msg0_a);
        msg3_b = vsha256su0q_u32(msg3_b, msg0_b);
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp0_a);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp0_b);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp0_a);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp0_b);
        msg3_a = vsha256su1q_u32(msg3_a, msg1_a, msg2_a);
        msg3_b = vsha256su1q_u32(msg3_b, msg1_b, msg2_b);

        // Transform 1: Rounds 49-52
        tmp = vld1q_u32(K.as_ptr().add(48));
        tmp0_a = vaddq_u32(msg0_a, tmp);
        tmp0_b = vaddq_u32(msg0_b, tmp);
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp0_a);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp0_b);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp0_a);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp0_b);

        // Transform 1: Rounds 53-56
        tmp = vld1q_u32(K.as_ptr().add(52));
        tmp0_a = vaddq_u32(msg1_a, tmp);
        tmp0_b = vaddq_u32(msg1_b, tmp);
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp0_a);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp0_b);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp0_a);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp0_b);

        // Transform 1: Rounds 57-60
        tmp = vld1q_u32(K.as_ptr().add(56));
        tmp0_a = vaddq_u32(msg2_a, tmp);
        tmp0_b = vaddq_u32(msg2_b, tmp);
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp0_a);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp0_b);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp0_a);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp0_b);

        // Transform 1: Rounds 61-64
        tmp = vld1q_u32(K.as_ptr().add(60));
        tmp0_a = vaddq_u32(msg3_a, tmp);
        tmp0_b = vaddq_u32(msg3_b, tmp);
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp0_a);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp0_b);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp0_a);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp0_b);

        // Transform 1: Update state
        tmp = vld1q_u32(&INIT[0]);
        state0_a = vaddq_u32(state0_a, tmp);
        state0_b = vaddq_u32(state0_b, tmp);
        tmp = vld1q_u32(&INIT[4]);
        state1_a = vaddq_u32(state1_a, tmp);
        state1_b = vaddq_u32(state1_b, tmp);

        // ------------------ Transform 2 -------------------

        // Transform 2: Save state
        abcd_save_a = state0_a;
        abcd_save_b = state0_b;
        efgh_save_a = state1_a;
        efgh_save_b = state1_b;

        // Transform 2: Rounds 1-4
        tmp = vld1q_u32(MIDS.as_ptr().add(0));
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp);

        // Transform 2: Rounds 5-8
        tmp = vld1q_u32(MIDS.as_ptr().add(4));
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp);

        // Transform 2: Rounds 9-12
        tmp = vld1q_u32(MIDS.as_ptr().add(8));
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp);

        // Transform 2: Rounds 13-16
        tmp = vld1q_u32(MIDS.as_ptr().add(12));
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp);

        // Transform 2: Rounds 17-20
        tmp = vld1q_u32(MIDS.as_ptr().add(16));
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp);

        // Transform 2: Rounds 21-24
        tmp = vld1q_u32(MIDS.as_ptr().add(20));
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp);

        // Transform 2: Rounds 25-28
        tmp = vld1q_u32(MIDS.as_ptr().add(24));
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp);

        // Transform 2: Rounds 29-32
        tmp = vld1q_u32(MIDS.as_ptr().add(28));
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp);

        // Transform 2: Rounds 33-36
        tmp = vld1q_u32(MIDS.as_ptr().add(32));
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp);

        // Transform 2: Rounds 37-40
        tmp = vld1q_u32(MIDS.as_ptr().add(36));
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp);

        // Transform 2: Rounds 41-44
        tmp = vld1q_u32(MIDS.as_ptr().add(40));
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp);

        // Transform 2: Rounds 45-48
        tmp = vld1q_u32(MIDS.as_ptr().add(44));
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp);

        // Transform 2: Rounds 49-52
        tmp = vld1q_u32(MIDS.as_ptr().add(48));
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp);

        // Transform 2: Rounds 53-56
        tmp = vld1q_u32(MIDS.as_ptr().add(52));
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp);

        // Transform 2: Rounds 57-60
        tmp = vld1q_u32(MIDS.as_ptr().add(56));
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp);

        // Transform 2: Rounds 61-64
        tmp = vld1q_u32(MIDS.as_ptr().add(60));
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp);

        // Transform 2: Update state
        state0_a = vaddq_u32(state0_a, abcd_save_a);
        state0_b = vaddq_u32(state0_b, abcd_save_b);
        state1_a = vaddq_u32(state1_a, efgh_save_a);
        state1_b = vaddq_u32(state1_b, efgh_save_b);

        // ------------------ Transform 3 -------------------

        msg0_a = state0_a;
        msg0_b = state0_b;
        msg1_a = state1_a;
        msg1_b = state1_b;
        msg2_a = vld1q_u32(FINAL.as_ptr().add(0));
        msg2_b = msg2_a;
        msg3_a = vld1q_u32(FINAL.as_ptr().add(4));
        msg3_b = msg3_a;

        // Transform 3: Load state
        state0_a = vld1q_u32(INIT.as_ptr());
        state0_b = state0_a;
        state1_a = vld1q_u32(INIT.as_ptr().add(4));
        state1_b = state1_a;

        // Transform 3: Rounds 1-4
        tmp = vld1q_u32(K.as_ptr().add(0));
        tmp0_a = vaddq_u32(msg0_a, tmp);
        tmp0_b = vaddq_u32(msg0_b, tmp);
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        msg0_a = vsha256su0q_u32(msg0_a, msg1_a);
        msg0_b = vsha256su0q_u32(msg0_b, msg1_b);
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp0_a);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp0_b);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp0_a);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp0_b);
        msg0_a = vsha256su1q_u32(msg0_a, msg2_a, msg3_a);
        msg0_b = vsha256su1q_u32(msg0_b, msg2_b, msg3_b);

        // Transform 3: Rounds 5-8
        tmp = vld1q_u32(K.as_ptr().add(4));
        tmp0_a = vaddq_u32(msg1_a, tmp);
        tmp0_b = vaddq_u32(msg1_b, tmp);
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        msg1_a = vsha256su0q_u32(msg1_a, msg2_a);
        msg1_b = vsha256su0q_u32(msg1_b, msg2_b);
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp0_a);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp0_b);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp0_a);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp0_b);
        msg1_a = vsha256su1q_u32(msg1_a, msg3_a, msg0_a);
        msg1_b = vsha256su1q_u32(msg1_b, msg3_b, msg0_b);

        // Transform 3: Rounds 9-12
        tmp = vld1q_u32(FINS.as_ptr().add(0));
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        msg2_a = vld1q_u32(FINS.as_ptr().add(4));
        msg2_b = msg2_a;
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp);
        msg2_a = vsha256su1q_u32(msg2_a, msg0_a, msg1_a);
        msg2_b = vsha256su1q_u32(msg2_b, msg0_b, msg1_b);

        // Transform 3: Rounds 13-16
        tmp = vld1q_u32(FINS.as_ptr().add(8));
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        msg3_a = vsha256su0q_u32(msg3_a, msg0_a);
        msg3_b = vsha256su0q_u32(msg3_b, msg0_b);
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp);
        msg3_a = vsha256su1q_u32(msg3_a, msg1_a, msg2_a);
        msg3_b = vsha256su1q_u32(msg3_b, msg1_b, msg2_b);

        // Transform 3: Rounds 17-20
        tmp = vld1q_u32(K.as_ptr().add(16));
        tmp0_a = vaddq_u32(msg0_a, tmp);
        tmp0_b = vaddq_u32(msg0_b, tmp);
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        msg0_a = vsha256su0q_u32(msg0_a, msg1_a);
        msg0_b = vsha256su0q_u32(msg0_b, msg1_b);
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp0_a);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp0_b);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp0_a);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp0_b);
        msg0_a = vsha256su1q_u32(msg0_a, msg2_a, msg3_a);
        msg0_b = vsha256su1q_u32(msg0_b, msg2_b, msg3_b);

        // Transform 3: Rounds 21-24
        tmp = vld1q_u32(K.as_ptr().add(20));
        tmp0_a = vaddq_u32(msg1_a, tmp);
        tmp0_b = vaddq_u32(msg1_b, tmp);
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        msg1_a = vsha256su0q_u32(msg1_a, msg2_a);
        msg1_b = vsha256su0q_u32(msg1_b, msg2_b);
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp0_a);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp0_b);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp0_a);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp0_b);
        msg1_a = vsha256su1q_u32(msg1_a, msg3_a, msg0_a);
        msg1_b = vsha256su1q_u32(msg1_b, msg3_b, msg0_b);

        // Transform 3: Rounds 25-28
        tmp = vld1q_u32(K.as_ptr().add(24));
        tmp0_a = vaddq_u32(msg2_a, tmp);
        tmp0_b = vaddq_u32(msg2_b, tmp);
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        msg2_a = vsha256su0q_u32(msg2_a, msg3_a);
        msg2_b = vsha256su0q_u32(msg2_b, msg3_b);
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp0_a);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp0_b);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp0_a);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp0_b);
        msg2_a = vsha256su1q_u32(msg2_a, msg0_a, msg1_a);
        msg2_b = vsha256su1q_u32(msg2_b, msg0_b, msg1_b);

        // Transform 3: Rounds 29-32
        tmp = vld1q_u32(K.as_ptr().add(28));
        tmp0_a = vaddq_u32(msg3_a, tmp);
        tmp0_b = vaddq_u32(msg3_b, tmp);
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        msg3_a = vsha256su0q_u32(msg3_a, msg0_a);
        msg3_b = vsha256su0q_u32(msg3_b, msg0_b);
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp0_a);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp0_b);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp0_a);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp0_b);
        msg3_a = vsha256su1q_u32(msg3_a, msg1_a, msg2_a);
        msg3_b = vsha256su1q_u32(msg3_b, msg1_b, msg2_b);

        // Transform 3: Rounds 33-36
        tmp = vld1q_u32(K.as_ptr().add(32));
        tmp0_a = vaddq_u32(msg0_a, tmp);
        tmp0_b = vaddq_u32(msg0_b, tmp);
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        msg0_a = vsha256su0q_u32(msg0_a, msg1_a);
        msg0_b = vsha256su0q_u32(msg0_b, msg1_b);
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp0_a);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp0_b);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp0_a);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp0_b);
        msg0_a = vsha256su1q_u32(msg0_a, msg2_a, msg3_a);
        msg0_b = vsha256su1q_u32(msg0_b, msg2_b, msg3_b);

        // Transform 3: Rounds 37-40
        tmp = vld1q_u32(K.as_ptr().add(36));
        tmp0_a = vaddq_u32(msg1_a, tmp);
        tmp0_b = vaddq_u32(msg1_b, tmp);
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        msg1_a = vsha256su0q_u32(msg1_a, msg2_a);
        msg1_b = vsha256su0q_u32(msg1_b, msg2_b);
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp0_a);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp0_b);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp0_a);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp0_b);
        msg1_a = vsha256su1q_u32(msg1_a, msg3_a, msg0_a);
        msg1_b = vsha256su1q_u32(msg1_b, msg3_b, msg0_b);

        // Transform 3: Rounds 41-44
        tmp = vld1q_u32(K.as_ptr().add(40));
        tmp0_a = vaddq_u32(msg2_a, tmp);
        tmp0_b = vaddq_u32(msg2_b, tmp);
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        msg2_a = vsha256su0q_u32(msg2_a, msg3_a);
        msg2_b = vsha256su0q_u32(msg2_b, msg3_b);
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp0_a);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp0_b);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp0_a);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp0_b);
        msg2_a = vsha256su1q_u32(msg2_a, msg0_a, msg1_a);
        msg2_b = vsha256su1q_u32(msg2_b, msg0_b, msg1_b);

        // Transform 3: Rounds 45-48
        tmp = vld1q_u32(K.as_ptr().add(44));
        tmp0_a = vaddq_u32(msg3_a, tmp);
        tmp0_b = vaddq_u32(msg3_b, tmp);
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        msg3_a = vsha256su0q_u32(msg3_a, msg0_a);
        msg3_b = vsha256su0q_u32(msg3_b, msg0_b);
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp0_a);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp0_b);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp0_a);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp0_b);
        msg3_a = vsha256su1q_u32(msg3_a, msg1_a, msg2_a);
        msg3_b = vsha256su1q_u32(msg3_b, msg1_b, msg2_b);

        // Transform 3: Rounds 49-52
        tmp = vld1q_u32(K.as_ptr().add(48));
        tmp0_a = vaddq_u32(msg0_a, tmp);
        tmp0_b = vaddq_u32(msg0_b, tmp);
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp0_a);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp0_b);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp0_a);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp0_b);

        // Transform 3: Rounds 53-56
        tmp = vld1q_u32(K.as_ptr().add(52));
        tmp0_a = vaddq_u32(msg1_a, tmp);
        tmp0_b = vaddq_u32(msg1_b, tmp);
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp0_a);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp0_b);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp0_a);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp0_b);

        // Transform 3: Rounds 57-60
        tmp = vld1q_u32(K.as_ptr().add(56));
        tmp0_a = vaddq_u32(msg2_a, tmp);
        tmp0_b = vaddq_u32(msg2_b, tmp);
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp0_a);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp0_b);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp0_a);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp0_b);

        // Transform 3: Rounds 61-64
        tmp = vld1q_u32(K.as_ptr().add(60));
        tmp0_a = vaddq_u32(msg3_a, tmp);
        tmp0_b = vaddq_u32(msg3_b, tmp);
        tmp2_a = state0_a;
        tmp2_b = state0_b;
        state0_a = vsha256hq_u32(state0_a, state1_a, tmp0_a);
        state0_b = vsha256hq_u32(state0_b, state1_b, tmp0_b);
        state1_a = vsha256h2q_u32(state1_a, tmp2_a, tmp0_a);
        state1_b = vsha256h2q_u32(state1_b, tmp2_b, tmp0_b);

        // Transform 3: Update state
        tmp = vld1q_u32(INIT.as_ptr().add(0));
        state0_a = vaddq_u32(state0_a, tmp);
        state0_b = vaddq_u32(state0_b, tmp);
        tmp = vld1q_u32(INIT.as_ptr().add(4));
        state1_a = vaddq_u32(state1_a, tmp);
        state1_b = vaddq_u32(state1_b, tmp);

        // Store result
        vst1q_u8(output[0].as_mut_ptr().add(0), vrev32q_u8(vreinterpretq_u8_u32(state0_a)));
        vst1q_u8(output[0].as_mut_ptr().add(16), vrev32q_u8(vreinterpretq_u8_u32(state1_a)));
        vst1q_u8(output[1].as_mut_ptr().add(0), vrev32q_u8(vreinterpretq_u8_u32(state0_b)));
        vst1q_u8(output[1].as_mut_ptr().add(16), vrev32q_u8(vreinterpretq_u8_u32(state1_b)));
    }

    // Algorithm copied from libsecp256k1
    fn software_process_block(state: &mut [u32; 8], blocks: &[u8]) {
        debug_assert!(!blocks.is_empty() && blocks.len() % BLOCK_SIZE == 0);

        for block in blocks.chunks_exact(BLOCK_SIZE) {
            let mut w = [0u32; 16];
            for (w_val, buff_bytes) in w.iter_mut().zip(block.bitcoin_as_chunks().0) {
                *w_val = u32::from_be_bytes(*buff_bytes);
            }

            let mut a = state[0];
            let mut b = state[1];
            let mut c = state[2];
            let mut d = state[3];
            let mut e = state[4];
            let mut f = state[5];
            let mut g = state[6];
            let mut h = state[7];

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
            let _ = w[15]; // silence "unnecessary assignment" lint in macro

            state[0] = state[0].wrapping_add(a);
            state[1] = state[1].wrapping_add(b);
            state[2] = state[2].wrapping_add(c);
            state[3] = state[3].wrapping_add(d);
            state[4] = state[4].wrapping_add(e);
            state[5] = state[5].wrapping_add(f);
            state[6] = state[6].wrapping_add(g);
            state[7] = state[7].wrapping_add(h);
        }
    }
}
