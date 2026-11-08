#![feature(test)]

extern crate slice_ring_buffer;
extern crate test;

use std::collections::VecDeque;

const MAX_NO_ITERS: usize = 1_000_000_000;

#[bench]
fn push_front_std_vecdeque(b: &mut test::Bencher) {
    let mut deq = VecDeque::<u8>::with_capacity(MAX_NO_ITERS);
    b.iter(|| {
        deq.push_front(3);
        test::black_box(&mut deq);
    });
}

#[bench]
fn push_front_slice_ring_buffer(b: &mut test::Bencher) {
    let mut deq =
        slice_ring_buffer::SliceRingBuffer::<u8>::with_capacity(MAX_NO_ITERS);
    b.iter(|| {
        deq.push_front(3);
        test::black_box(&mut deq);
    });
}
