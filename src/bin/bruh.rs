use std::{sync::{atomic::{compiler_fence, Ordering}, OnceLock}, arch::x86_64::*};

use groestl_gfni::digest::{Digest, core_api::CoreWrapper};
use rand::{thread_rng, RngCore};

#[inline(always)]
fn square<T: std::ops::Mul<T> + Copy>(x: T) -> <T as std::ops::Mul<T>>::Output {
    x*x
}

fn profile(gen: impl Digest + Clone) {
    static DATA: OnceLock<Vec<u8>> = OnceLock::new();
    let mut hsh = None;

    let data = DATA.get_or_init(|| vec![11; 1024*1024*128]);
    let mut cpb = Vec::new();

    for _ in 0..40 {
        compiler_fence(Ordering::SeqCst);
        unsafe { __cpuid(0) };
        compiler_fence(Ordering::SeqCst);
        let t0 = unsafe { _rdtsc() };
        compiler_fence(Ordering::SeqCst);
        let mut h = gen.clone();
        h.update(data);
        let hash = h.finalize();
        compiler_fence(Ordering::SeqCst);
        let t1 = unsafe { __rdtscp(&mut 0) };
        compiler_fence(Ordering::SeqCst);
        unsafe { __cpuid(0) };
        compiler_fence(Ordering::SeqCst);
        match hsh {
            None => hsh = Some(hash),
            Some(ref b) => assert_eq!(*b, hash)
        }
        cpb.push((t1-t0) as f32 / data.len() as f32);
    }

    let mean = cpb.iter().copied().sum::<f32>() / cpb.len() as f32;
    let stdev = (cpb.iter().map(|&z| square(z-mean)).sum::<f32>() / cpb.len() as f32).sqrt();

    println!("mean = {mean}, stdev = {stdev}");
}

fn main() {
    let mut msg = vec![0; i32::MAX as usize]; // world's most famous mersenne prime

    thread_rng().fill_bytes(&mut msg);

    use std::time::Instant;

    let t = Instant::now();
    let reff = {
        let mut h = groestl::Groestl256::new();
        h.update(&msg);
        h.finalize()
    };
    println!("{}", (Instant::now()-t).as_secs_f32());


    let t = Instant::now();
    let test = {
        let mut h = groestl_gfni::Groestl256::new();
        h.update(&msg);
        h.finalize()
    };
    println!("{}", (Instant::now()-t).as_secs_f32());

    assert_eq!(reff[..], test[..]);

    // let orig = groestl::Groestl256::new();
    // let test = groestl_gfni::Groestl256::new();

    // profile(orig);
    // profile(test);
}
