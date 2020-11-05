
/// Tabulation datum type (internal use only).
#[repr(C)]
#[derive(Copy, Clone, Default, Debug)]
pub struct Datum<T: super::Float> {
    pub alpha: T,              // (x[i+1] - x[i]) / wedge_switch[i]
    pub beta: T,               // x[i] - x0
    pub wedge_switch: T::UInt, // (yinf / ysup) * tail_switch
}

/// Backing storage for fixed-size arrays (internal use only).
pub trait Storage<T>: Clone + AsRef<[T]> + AsMut<[T]> {
    fn init() -> Self;
}

macro_rules! impl_storage {
    ($sz:expr) => {
        impl<T: Default + Copy> Storage<T> for [T; $sz] {
            fn init() -> Self {
                [T::default(); $sz]
            }
        }
        impl<T: Default + Copy> Storage<T> for [T; $sz + 1] {
            fn init() -> Self {
                [T::default(); $sz + 1]
            }
        }
    }
}
impl_storage!(16);
impl_storage!(32);
impl_storage!(64);
impl_storage!(128);
impl_storage!(256);
impl_storage!(512);
impl_storage!(1024);
impl_storage!(2048);
impl_storage!(4096);
