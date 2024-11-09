#![feature(stdarch_x86_avx512)]
#![feature(avx512_target_feature)]

use std::arch::x86_64::*;

use rand::RngCore;

#[target_feature(enable = "sse2")]
unsafe fn mul2(x: __m128i) -> __m128i {
    _mm_xor_si128(
        _mm_add_epi8(x, x),
        _mm_and_si128(
            _mm_set1_epi8(0x1b),
            _mm_cmplt_epi8(x, _mm_setzero_si128())
        )
    )
}

#[target_feature(enable = "sse2,sse4.1")]
unsafe fn mul2_sse4(x: __m128i) -> __m128i {
    let dbl = _mm_add_epi8(x, x);
    let xor = _mm_xor_si128(dbl, _mm_set1_epi8(0x1b));
    _mm_blendv_epi8(dbl, xor, x)
}

#[target_feature(enable = "gfni")]
unsafe fn mul2_gfni(x: __m128i) -> __m128i {
    _mm_gf2p8mul_epi8(x, _mm_set1_epi8(2))
}

fn main() { unsafe {
    let mut x = [0; 16];
    rand::thread_rng().fill_bytes(&mut x);
    let r0 = _mm_loadu_si128(x.as_ptr() as *const __m128i);
    let r1 = mul2(r0);
    let r2 = mul2_gfni(r0);
    let mut y = [0u8; 16];
    let mut z = [0u8; 16];
    _mm_storeu_si128(y.as_mut_ptr().cast(), r1);
    _mm_storeu_si128(z.as_mut_ptr().cast(), r2);
    assert_eq!(y, z)
}}