use std::arch::x86_64::*;

use digest::{array::Array, block_buffer::{BlockBuffer, Eager}, consts::{U64, U128}, core_api::{BlockSizeUser, BufferKindUser, CoreWrapper, FixedOutputCore, UpdateCore}, HashMarker, OutputSizeUser, Reset};

type Mat1024 = (__m512i, __m512i);

#[inline(always)]
unsafe fn mix_bytes(x: Mat1024) -> Mat1024 {
    const ROW2: i64 = 0x0202020202020202;
    const ROW3: i64 = 0x0303030303030303;
    const ROW4: i64 = 0x0404040404040404;
    const ROW5: i64 = 0x0505050505050505;
    const ROW7: i64 = 0x0707070707070707;

    let wide0 = _mm512_shuffle_i32x4(x.0, x.0, 0b00000000);
    let wide1 = _mm512_shuffle_i32x4(x.0, x.0, 0b01010101);
    let wide2 = _mm512_shuffle_i32x4(x.0, x.0, 0b10101010);
    let wide3 = _mm512_shuffle_i32x4(x.0, x.0, 0b11111111);
    let wide4 = _mm512_shuffle_i32x4(x.1, x.1, 0b00000000);
    let wide5 = _mm512_shuffle_i32x4(x.1, x.1, 0b01010101);
    let wide6 = _mm512_shuffle_i32x4(x.1, x.1, 0b10101010);
    let wide7 = _mm512_shuffle_i32x4(x.1, x.1, 0b11111111);


    (
        _mm512_xor_si512(
            _mm512_xor_si512(
                _mm512_xor_si512(
                    _mm512_gf2p8mul_epi8(wide0, _mm512_setr_epi64(ROW2, ROW2, ROW7, ROW7, ROW5, ROW5, ROW3, ROW3)),
                    _mm512_gf2p8mul_epi8(wide1, _mm512_setr_epi64(ROW2, ROW2, ROW2, ROW2, ROW7, ROW7, ROW5, ROW5)),
                ),
                _mm512_xor_si512(
                    _mm512_gf2p8mul_epi8(wide2, _mm512_setr_epi64(ROW3, ROW3, ROW2, ROW2, ROW2, ROW2, ROW7, ROW7)),
                    _mm512_gf2p8mul_epi8(wide3, _mm512_setr_epi64(ROW4, ROW4, ROW3, ROW3, ROW2, ROW2, ROW2, ROW2))
                )
            ),
            _mm512_xor_si512(
                _mm512_xor_si512(
                    _mm512_gf2p8mul_epi8(wide4, _mm512_setr_epi64(ROW5, ROW5, ROW4, ROW4, ROW3, ROW3, ROW2, ROW2)),
                    _mm512_gf2p8mul_epi8(wide5, _mm512_setr_epi64(ROW3, ROW3, ROW5, ROW5, ROW4, ROW4, ROW3, ROW3))
                ),
                _mm512_xor_si512(
                    _mm512_gf2p8mul_epi8(wide6, _mm512_setr_epi64(ROW5, ROW5, ROW3, ROW3, ROW5, ROW5, ROW4, ROW4)),
                    _mm512_gf2p8mul_epi8(wide7, _mm512_setr_epi64(ROW7, ROW7, ROW5, ROW5, ROW3, ROW3, ROW5, ROW5))
                )
            )
        ),

        _mm512_xor_si512(
            _mm512_xor_si512(
                _mm512_xor_si512(
                    _mm512_gf2p8mul_epi8(wide0, _mm512_setr_epi64(ROW5, ROW5, ROW4, ROW4, ROW3, ROW3, ROW2, ROW2)),
                    _mm512_gf2p8mul_epi8(wide1, _mm512_setr_epi64(ROW3, ROW3, ROW5, ROW5, ROW4, ROW4, ROW3, ROW3)),
                ),
                _mm512_xor_si512(
                    _mm512_gf2p8mul_epi8(wide2, _mm512_setr_epi64(ROW5, ROW5, ROW3, ROW3, ROW5, ROW5, ROW4, ROW4)),
                    _mm512_gf2p8mul_epi8(wide3, _mm512_setr_epi64(ROW7, ROW7, ROW5, ROW5, ROW3, ROW3, ROW5, ROW5))
                )
            ),
            _mm512_xor_si512(
                _mm512_xor_si512(
                    _mm512_gf2p8mul_epi8(wide4, _mm512_setr_epi64(ROW2, ROW2, ROW7, ROW7, ROW5, ROW5, ROW3, ROW3)),
                    _mm512_gf2p8mul_epi8(wide5, _mm512_setr_epi64(ROW2, ROW2, ROW2, ROW2, ROW7, ROW7, ROW5, ROW5))
                ),
                _mm512_xor_si512(
                    _mm512_gf2p8mul_epi8(wide6, _mm512_setr_epi64(ROW3, ROW3, ROW2, ROW2, ROW2, ROW2, ROW7, ROW7)),
                    _mm512_gf2p8mul_epi8(wide7, _mm512_setr_epi64(ROW4, ROW4, ROW3, ROW3, ROW2, ROW2, ROW2, ROW2))
                )
            )
        )
    )
}

#[inline(always)]
unsafe fn round512<const R: u8, const P: bool>(x: Mat1024) -> Mat1024 {
    // AddRoundConstant
    let rc0 = 0x7060504030201000u64 | 0x0101010101010101u64*(R as u64);
    let rc1 = 0xf0e0d0c0b0a09080u64 | 0x0101010101010101u64*(R as u64);

    let x = if P {
        (
            _mm512_xor_si512(x.0, _mm512_setr_epi64(rc0 as i64, rc1 as i64, 0, 0, 0, 0, 0, 0)),
            x.1
        )
    } else {
        (
            _mm512_xor_si512(x.0, _mm512_set1_epi8(-1)),
            _mm512_xor_si512(x.1, _mm512_set_epi64(!rc1 as i64, !rc0 as i64, -1, -1, -1, -1, -1, -1))
        )
    };

    // SubBytes (Rijndael S-box)
    let submat = _mm512_set1_epi64(0xF1E3C78F1F3E7CF8u64 as i64);
    let x = (
        _mm512_gf2p8affineinv_epi64_epi8(x.0, submat, 0b01100011),
        _mm512_gf2p8affineinv_epi64_epi8(x.1, submat, 0b01100011)
    );

    // ShiftBytes
    let shiftmat: Mat1024 = if P {
        (
            _mm512_setr_epi64(
                0x0706050403020100, 0x0f0e0d0c0b0a0908,
                0x1817161514131211, 0x101f1e1d1c1b1a19,
                0x2928272625242322, 0x21202f2e2d2c2b2a,
                0x3a39383736353433, 0x3231303f3e3d3c3b
            ),
            _mm512_setr_epi64(
                0x0b0a090807060504, 0x030201000f0e0d0c,
                0x1c1b1a1918171615, 0x14131211101f1e1d,
                0x2d2c2b2a29282726, 0x2524232221202f2e,
                0x3231303f3e3d3c3b, 0x3a39383736353433
            )
        )
    } else {
        (
            _mm512_setr_epi64(
                0x0807060504030201, 0x000f0e0d0c0b0a09,
                0x1a19181716151413, 0x1211101f1e1d1c1b,
                0x2c2b2a2928272625, 0x24232221202f2e2d,
                0x3231303f3e3d3c3b, 0x3a39383736353433
            ),
            _mm512_setr_epi64(
                0x0706050403020100, 0x0f0e0d0c0b0a0908,
                0x1918171615141312, 0x11101f1e1d1c1b1a,
                0x2b2a292827262524, 0x232221202f2e2d2c,
                0x3d3c3b3a39383736, 0x3534333231303f3e
            )
        )
    };
    let x = (
        _mm512_permutexvar_epi8(shiftmat.0, x.0),
        _mm512_permutexvar_epi8(shiftmat.1, x.1)
    );

    // MixBytes
    let x = mix_bytes(x);

    x
}

#[inline(always)]
unsafe fn transpose((top, bot): Mat1024) -> Mat1024 {
    const TRANSPOSE: i64 = 0x7060504030201000;
    const OFFSET: i64 = 0x0101010101010101;
    (
        _mm512_permutex2var_epi8(
            top,
            _mm512_setr_epi64(
                TRANSPOSE, TRANSPOSE+OFFSET, 
                TRANSPOSE+OFFSET*2, TRANSPOSE+OFFSET*3, 
                TRANSPOSE+OFFSET*4, TRANSPOSE+OFFSET*5, 
                TRANSPOSE+OFFSET*6, TRANSPOSE+OFFSET*7
            ),
            bot
        ),
        _mm512_permutex2var_epi8(
            top,
            _mm512_setr_epi64(
                TRANSPOSE+OFFSET*8, TRANSPOSE+OFFSET*9, 
                TRANSPOSE+OFFSET*10, TRANSPOSE+OFFSET*11, 
                TRANSPOSE+OFFSET*12, TRANSPOSE+OFFSET*13, 
                TRANSPOSE+OFFSET*14, TRANSPOSE+OFFSET*15
            ),
            bot
        )
    )
}

#[inline(always)]
unsafe fn untranspose((top, bot): Mat1024) -> Mat1024 {
    const UNTRANSPOSE1: i64 = 0x3830282018100800;
    const UNTRANSPOSE2: i64 = 0x7870686058504840;
    const OFFSET: i64 = 0x0101010101010101;
    (
        _mm512_permutex2var_epi8(
            top,
            _mm512_setr_epi64(
                UNTRANSPOSE1, UNTRANSPOSE2, 
                UNTRANSPOSE1+OFFSET, UNTRANSPOSE2+OFFSET,
                UNTRANSPOSE1+OFFSET*2, UNTRANSPOSE2+OFFSET*2, 
                UNTRANSPOSE1+OFFSET*3, UNTRANSPOSE2+OFFSET*3
            ),
            bot
        ),
        _mm512_permutex2var_epi8(
            top,
            _mm512_setr_epi64(
                UNTRANSPOSE1+OFFSET*4, UNTRANSPOSE2+OFFSET*4, 
                UNTRANSPOSE1+OFFSET*5, UNTRANSPOSE2+OFFSET*5,
                UNTRANSPOSE1+OFFSET*6, UNTRANSPOSE2+OFFSET*6, 
                UNTRANSPOSE1+OFFSET*7, UNTRANSPOSE2+OFFSET*7
            ),
            bot
        )
    )
}

#[inline(always)]
unsafe fn permute256<const P: bool>(x: Mat1024) -> Mat1024 {
    let x = round512::<0, P>(x);
    let x = round512::<1, P>(x);
    let x = round512::<2, P>(x);
    let x = round512::<3, P>(x);
    let x = round512::<4, P>(x);
    let x = round512::<5, P>(x);
    let x = round512::<6, P>(x);
    let x = round512::<7, P>(x);
    let x = round512::<8, P>(x);
    let x = round512::<9, P>(x);
    let x = round512::<10, P>(x);
    let x = round512::<11, P>(x);
    let x = round512::<12, P>(x);
    let x = round512::<13, P>(x);

    x
}

#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn initial512() -> Mat1024 {
    //transpose((
    //    _mm512_setzero_epi32(),
    //    _mm512_set_epi64(u64::to_be(512) as i64, 0, 0, 0, 0, 0, 0, 0)
    //))
    (
        _mm512_setzero_epi32(),
        _mm512_setr_epi64(0, 0, 0, 0, 0, 0, 0x0200000000000000, 0)
    )
}

#[target_feature(enable = "sse4.1,avx512f,avx512bw,gfni,avx512vbmi")]
unsafe fn compress512(h: Mat1024, m: &[u8; 128]) -> Mat1024 {
    let m = transpose((
        _mm512_loadu_si512(m.as_ptr() as *const i32),
        _mm512_loadu_si512(m[64..].as_ptr() as *const i32)
    ));

    let p = permute256::<true>((
        _mm512_xor_si512(h.0, m.0),
        _mm512_xor_si512(h.1, m.1)
    ));
    let q = permute256::<false>(m);

    (
        _mm512_xor_si512(
            _mm512_xor_si512(
                p.0,
                q.0
            ),
            h.0
        ),
        _mm512_xor_si512(
            _mm512_xor_si512(
                p.1,
                q.1
            ),
            h.1
        )
    )
}

#[target_feature(enable = "sse4.1,avx,avx512f,avx512bw,gfni,avx512vbmi")]
unsafe fn finalize512(h: Mat1024, out: &mut [u8; 64]) {
    let p = permute256::<true>(h);
    let h = untranspose((
        _mm512_xor_si512(h.0, p.0),
        _mm512_xor_si512(h.1, p.1)
    ));
    _mm512_storeu_si512(out.as_mut_ptr() as *mut i32, h.1);
}


#[derive(Clone)]
pub struct Groestl512Core {
    chaining: Mat1024,
    nblocks: u64
}

impl Default for Groestl512Core {
    #[inline]
    fn default() -> Self {
        Self { chaining: unsafe { initial512() }, nblocks: 0 }
    }
}

impl Reset for Groestl512Core {
    #[inline]
    fn reset(&mut self) {
        *self = Default::default();
    }
}

impl HashMarker for Groestl512Core {}

impl BlockSizeUser for Groestl512Core {
    type BlockSize = U128;
}

impl OutputSizeUser for Groestl512Core {
    type OutputSize = U64;
}

impl BufferKindUser for Groestl512Core {
    type BufferKind = Eager;
}

impl UpdateCore for Groestl512Core {
    #[inline]
    fn update_blocks(&mut self, blocks: &[Array<u8, U128>]) {
        self.nblocks += blocks.len() as u64;
        for block in blocks {
            self.chaining = unsafe { compress512(self.chaining, &block.0) };
        }
    }
}

impl FixedOutputCore for Groestl512Core {
    #[inline]
    fn finalize_fixed_core(&mut self, buffer: &mut BlockBuffer<U128, Eager>, out: &mut Array<u8, U64>) {
        self.nblocks += if buffer.remaining() <= 8 {
            2
        } else {
            1
        };
        buffer.len64_padding_be(self.nblocks, |block|
            self.chaining = unsafe { compress512(self.chaining, &block.0) }
        );
        unsafe { finalize512(self.chaining, &mut out.0) };
    }
}

pub type Groestl512 = CoreWrapper<Groestl512Core>;
