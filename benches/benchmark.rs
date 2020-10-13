use criterion::{criterion_group, criterion_main, Criterion};
use etf::distributions::CentralNormal;
use etf::primitives::Distribution as _;
use rand::distributions::Distribution;
use rand_distr;
//use rand_pcg;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;


pub fn etf_benchmark(c: &mut Criterion) {
    let dist = CentralNormal::new(1.0_f64);
    //let mut rng = rand_pcg::Lcg128Xsl64::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7ac28fa16a64abf96);
    let mut rng = Xoshiro256StarStar::seed_from_u64(0);
    //let mut rng = rand::thread_rng();
    c.bench_function("ETF", |b| b.iter(|| dist.sample(&mut rng)));
}
pub fn rand_benchmark(c: &mut Criterion) {
    let dist = rand_distr::StandardNormal;
    //let mut rng = rand_pcg::Lcg128Xsl64::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7ac28fa16a64abf96);
    let mut rng = Xoshiro256StarStar::seed_from_u64(0);
    //let mut rng = rand::thread_rng();
    c.bench_function("rand", |b| b.iter(|| Distribution::<f64>::sample(&dist, &mut rng)));
}

criterion_group!(benches, rand_benchmark, etf_benchmark);
criterion_main!(benches);
