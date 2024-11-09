# AVX512/GFNI implementation of Grøstl

Grøstl is a SHA-3 finalist that was meant to run well on hardware with AES instructions. Turns out, it runs even better with GFNI instructions thanks to the lower overhead.

This also runs faster than the RustCrypto implementation, which uses a lookup table instead of specialized instructions.

Unfortunately, nobody uses it because Keccak was selected by NIST, and BLAKE ended up being popular as well thanks to its simplicity. Additionally, because this requires nightly Rust and a fairly rare breed of CPU (seeing as how Intel wants people to forget about AVX512), I don't really see the utility here at the moment.

Regardless, if the implementation is correct, this should be a reasonably secure 256-bit hash, on par with BLAKE2s.
