#![feature(stdarch_x86_avx512)]
#![feature(avx512_target_feature)]

//pub mod g512;

use std::arch::x86_64::*;

use digest::{array::Array, block_buffer::{BlockBuffer, Eager}, consts::{U16, U20, U28, U32, U64}, core_api::{BlockSizeUser, BufferKindUser, CoreWrapper, CtVariableCoreWrapper, TruncSide, UpdateCore, VariableOutputCore}, HashMarker, InvalidOutputSize, OutputSizeUser};

#[inline(always)]
unsafe fn mix_bytes(x: __m512i) -> __m512i {
    const ROW2: i64 = 0x0202020202020202;
    const ROW3: i64 = 0x0303030303030303;
    const ROW4: i64 = 0x0404040404040404;
    const ROW5: i64 = 0x0505050505050505;
    const ROW7: i64 = 0x0707070707070707;

    let half0 = _mm512_extracti64x4_epi64(x, 0);
    let half1 = _mm512_extracti64x4_epi64(x, 1);
    
    let wide0 = _mm512_set1_epi64(_mm256_extract_epi64(half0, 0));
    let wide1 = _mm512_set1_epi64(_mm256_extract_epi64(half0, 1));
    let wide2 = _mm512_set1_epi64(_mm256_extract_epi64(half0, 2));
    let wide3 = _mm512_set1_epi64(_mm256_extract_epi64(half0, 3));
    let wide4 = _mm512_set1_epi64(_mm256_extract_epi64(half1, 0));
    let wide5 = _mm512_set1_epi64(_mm256_extract_epi64(half1, 1));
    let wide6 = _mm512_set1_epi64(_mm256_extract_epi64(half1, 2));
    let wide7 = _mm512_set1_epi64(_mm256_extract_epi64(half1, 3));

    _mm512_xor_si512(
        _mm512_xor_si512(
            _mm512_xor_si512(
                _mm512_gf2p8mul_epi8(wide0, _mm512_setr_epi64(ROW2, ROW7, ROW5, ROW3, ROW5, ROW4, ROW3, ROW2)),
                _mm512_gf2p8mul_epi8(wide1, _mm512_setr_epi64(ROW2, ROW2, ROW7, ROW5, ROW3, ROW5, ROW4, ROW3)),
            ),
            _mm512_xor_si512(
                _mm512_gf2p8mul_epi8(wide2, _mm512_setr_epi64(ROW3, ROW2, ROW2, ROW7, ROW5, ROW3, ROW5, ROW4)),
                _mm512_gf2p8mul_epi8(wide3, _mm512_setr_epi64(ROW4, ROW3, ROW2, ROW2, ROW7, ROW5, ROW3, ROW5))
            )
        ),
        _mm512_xor_si512(
            _mm512_xor_si512(
                _mm512_gf2p8mul_epi8(wide4, _mm512_setr_epi64(ROW5, ROW4, ROW3, ROW2, ROW2, ROW7, ROW5, ROW3)),
                _mm512_gf2p8mul_epi8(wide5, _mm512_setr_epi64(ROW3, ROW5, ROW4, ROW3, ROW2, ROW2, ROW7, ROW5))
            ),
            _mm512_xor_si512(
                _mm512_gf2p8mul_epi8(wide6, _mm512_setr_epi64(ROW5, ROW3, ROW5, ROW4, ROW3, ROW2, ROW2, ROW7)),
                _mm512_gf2p8mul_epi8(wide7, _mm512_setr_epi64(ROW7, ROW5, ROW3, ROW5, ROW4, ROW3, ROW2, ROW2))
            )
        )
    )
}

#[inline(always)]
unsafe fn round256<const R: u8, const P: bool>(x: __m512i) -> __m512i {
    // AddRoundConstant
    let rc = 0x7060504030201000u64 | 0x0101010101010101u64*(R as u64);
    let x = _mm512_xor_si512(x, if P {
        _mm512_setr_epi64(rc as i64, 0, 0, 0, 0, 0, 0, 0)
    } else {
        _mm512_set_epi64(!rc as i64, -1, -1, -1, -1, -1, -1, -1)
    });

    // SubBytes (Rijndael S-box)
    let x = _mm512_gf2p8affineinv_epi64_epi8(x, _mm512_set1_epi64(0xF1E3C78F1F3E7CF8u64 as i64), 0b01100011);

    // ShiftBytes
    let x = _mm512_rorv_epi64(x, if P {
        _mm512_setr_epi64(0, 8, 16, 24, 32, 40, 48, 56)
    } else {
        _mm512_setr_epi64(8, 24, 40, 56, 0, 16, 32, 48)
    });

    // MixBytes
    let x = mix_bytes(x);

    x
}

#[inline(always)]
unsafe fn transpose(x: __m512i) -> __m512i {
    const TRANSPOSE: i64 = 0x3830282018100800;
    const OFFSET: i64 = 0x0101010101010101;
    _mm512_permutexvar_epi8(
        _mm512_setr_epi64(
            TRANSPOSE,
            TRANSPOSE+OFFSET,
            TRANSPOSE+OFFSET*2,
            TRANSPOSE+OFFSET*3,
            TRANSPOSE+OFFSET*4,
            TRANSPOSE+OFFSET*5,
            TRANSPOSE+OFFSET*6,
            TRANSPOSE+OFFSET*7
        ),
        x
    )
}

#[inline(always)]
unsafe fn permute256<const P: bool>(x: __m512i) -> __m512i {
    let x = round256::<0, P>(x);
    let x = round256::<1, P>(x);
    let x = round256::<2, P>(x);
    let x = round256::<3, P>(x);
    let x = round256::<4, P>(x);
    let x = round256::<5, P>(x);
    let x = round256::<6, P>(x);
    let x = round256::<7, P>(x);
    let x = round256::<8, P>(x);
    let x = round256::<9, P>(x);

    x
}

#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn initial256(output_bits: u64) -> __m512i {
    transpose(_mm512_set_epi64(output_bits.to_be() as i64, 0, 0, 0, 0, 0, 0, 0))
}

#[target_feature(enable = "avx2,avx512f,avx512bw,gfni,avx512vbmi")]
unsafe fn compress256(h: __m512i, m: &[u8; 64]) -> __m512i {
    let m = transpose(_mm512_loadu_si512(m.as_ptr() as *const i32));
    _mm512_xor_si512(
        _mm512_xor_si512(
            permute256::<true>(_mm512_xor_si512(h, m)),
            permute256::<false>(m)
        ),
        h
    )
}

#[target_feature(enable = "avx,avx2,avx512f,avx512bw,gfni,avx512vbmi")]
unsafe fn finalize256(h: __m512i, out: &mut [u8; 32]) {
    let h = transpose(_mm512_xor_si512(h, permute256::<true>(h)));
    _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, _mm512_extracti64x4_epi64(h, 1));
}

#[derive(Clone)]
pub struct GroestlShortCore {
    chaining: __m512i,
    nblocks: u64
}

impl HashMarker for GroestlShortCore {}

impl BlockSizeUser for GroestlShortCore {
    type BlockSize = U64;
}

impl OutputSizeUser for GroestlShortCore {
    type OutputSize = U32;
}

impl BufferKindUser for GroestlShortCore {
    type BufferKind = Eager;
}

impl UpdateCore for GroestlShortCore {
    #[inline]
    fn update_blocks(&mut self, blocks: &[Array<u8, U64>]) {
        self.nblocks += blocks.len() as u64;
        for block in blocks {
            self.chaining = unsafe { compress256(self.chaining, &block.0) };
        }
    }
}

impl VariableOutputCore for GroestlShortCore {
    const TRUNC_SIDE: TruncSide = TruncSide::Right;

    #[inline]
    fn new(output_size: usize) -> Result<Self, InvalidOutputSize> {
        match output_size {
            1..=32 => Ok(Self { chaining: unsafe { initial256((output_size * 8).try_into().unwrap()) }, nblocks: 0 }),
            _ => Err(InvalidOutputSize)
        }
    }

    #[inline]
    fn finalize_variable_core(&mut self, buffer: &mut BlockBuffer<U64, Eager>, out: &mut Array<u8, U32>) {
        self.nblocks += if buffer.remaining() <= 8 {
            2
        } else {
            1
        };
        buffer.len64_padding_be(self.nblocks, |block|
            self.chaining = unsafe { compress256(self.chaining, &block.0) }
        );
        unsafe { finalize256(self.chaining, &mut out.0) };
    }
}

pub use digest;
pub type Groestl128 = CoreWrapper<CtVariableCoreWrapper<GroestlShortCore, U16>>;
pub type Groestl160 = CoreWrapper<CtVariableCoreWrapper<GroestlShortCore, U20>>;
pub type Groestl224 = CoreWrapper<CtVariableCoreWrapper<GroestlShortCore, U28>>;
pub type Groestl256 = CoreWrapper<CtVariableCoreWrapper<GroestlShortCore, U32>>;
pub use digest::Digest;