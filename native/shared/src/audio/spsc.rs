//! Fixed-capacity lock-free single-producer/single-consumer ring buffer.
//!
//! Purpose-built for the audio command queue: the main (FFI) thread is the
//! only producer, the platform audio thread the only consumer, and the
//! consumer must never block, allocate, or take a lock — an audio callback
//! that waits on a mutex held across a frame hitch produces an audible
//! glitch, and a poisoned lock would kill sound for the rest of the
//! session (both observed failure modes of the old shared-mutable mixer).
//!
//! Standard two-counter ring: `tail` is written only by the producer,
//! `head` only by the consumer; each side reads the other's counter with
//! Acquire and publishes its own with Release. One slot is sacrificed to
//! distinguish full from empty.

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

struct Ring<T> {
    buf: Box<[UnsafeCell<MaybeUninit<T>>]>,
    /// next slot to read; owned by the consumer
    head: AtomicUsize,
    /// next slot to write; owned by the producer
    tail: AtomicUsize,
}

// The ring itself is shared between exactly two threads with disjoint
// roles; the unsafe impls are sound because Producer and Consumer are
// each !Clone and only touch their own counter + the slots their
// protocol entitles them to.
unsafe impl<T: Send> Send for Ring<T> {}
unsafe impl<T: Send> Sync for Ring<T> {}

impl<T> Drop for Ring<T> {
    fn drop(&mut self) {
        // By the time the last Arc drops we have exclusive access; drain
        // any unconsumed items so their destructors (Arc<SoundData>!) run.
        let mut head = *self.head.get_mut();
        let tail = *self.tail.get_mut();
        while head != tail {
            unsafe { (*self.buf[head].get()).assume_init_drop() };
            head = (head + 1) % self.buf.len();
        }
    }
}

pub struct Producer<T> {
    ring: Arc<Ring<T>>,
}

pub struct Consumer<T> {
    ring: Arc<Ring<T>>,
}

pub fn channel<T: Send>(capacity: usize) -> (Producer<T>, Consumer<T>) {
    assert!(capacity >= 2);
    let buf: Box<[UnsafeCell<MaybeUninit<T>>]> = (0..capacity)
        .map(|_| UnsafeCell::new(MaybeUninit::uninit()))
        .collect();
    let ring = Arc::new(Ring {
        buf,
        head: AtomicUsize::new(0),
        tail: AtomicUsize::new(0),
    });
    (Producer { ring: ring.clone() }, Consumer { ring })
}

impl<T> Producer<T> {
    /// Push an item; returns it back if the ring is full (the caller
    /// decides whether dropping the command is acceptable).
    pub fn push(&mut self, item: T) -> Result<(), T> {
        let ring = &*self.ring;
        let tail = ring.tail.load(Ordering::Relaxed);
        let next = (tail + 1) % ring.buf.len();
        if next == ring.head.load(Ordering::Acquire) {
            return Err(item); // full
        }
        unsafe { (*ring.buf[tail].get()).write(item) };
        ring.tail.store(next, Ordering::Release);
        Ok(())
    }
}

impl<T> Consumer<T> {
    /// Pop the oldest item, if any. Wait-free; safe to call from a
    /// real-time audio callback.
    pub fn pop(&mut self) -> Option<T> {
        let ring = &*self.ring;
        let head = ring.head.load(Ordering::Relaxed);
        if head == ring.tail.load(Ordering::Acquire) {
            return None; // empty
        }
        let item = unsafe { (*ring.buf[head].get()).assume_init_read() };
        ring.head.store((head + 1) % ring.buf.len(), Ordering::Release);
        Some(item)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_pop_order() {
        let (mut tx, mut rx) = channel::<u32>(4);
        assert!(tx.push(1).is_ok());
        assert!(tx.push(2).is_ok());
        assert!(tx.push(3).is_ok()); // capacity 4 → 3 usable slots
        assert!(tx.push(4).is_err()); // full
        assert_eq!(rx.pop(), Some(1));
        assert_eq!(rx.pop(), Some(2));
        assert!(tx.push(4).is_ok());
        assert_eq!(rx.pop(), Some(3));
        assert_eq!(rx.pop(), Some(4));
        assert_eq!(rx.pop(), None);
    }

    #[test]
    fn cross_thread_stream() {
        let (mut tx, mut rx) = channel::<u64>(64);
        let producer = std::thread::spawn(move || {
            for i in 0..100_000u64 {
                loop {
                    match tx.push(i) {
                        Ok(()) => break,
                        Err(_) => std::hint::spin_loop(),
                    }
                }
            }
        });
        let mut expected = 0u64;
        while expected < 100_000 {
            if let Some(v) = rx.pop() {
                assert_eq!(v, expected, "out-of-order or corrupted item");
                expected += 1;
            } else {
                std::hint::spin_loop();
            }
        }
        producer.join().unwrap();
    }

    #[test]
    fn drop_drains_unconsumed_items() {
        let flag = Arc::new(());
        let (mut tx, rx) = channel::<Arc<()>>(8);
        for _ in 0..5 {
            tx.push(flag.clone()).unwrap();
        }
        assert_eq!(Arc::strong_count(&flag), 6);
        drop(tx);
        drop(rx);
        assert_eq!(Arc::strong_count(&flag), 1, "ring leaked queued items");
    }
}
