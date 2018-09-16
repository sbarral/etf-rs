extern crate etf;
extern crate rand;

use rand::Rng;

fn main() {
    let table = etf::table::Table256 {
        x: [0.0f64; 257],
        yinf: [0.0f64; 256],
        ysup: [0.0f64; 256],
    };

    //let d = etf::DistBuilder::new_symmetric(&*table).standalone();
    //let d = etf::DistBuilder::new_symmetric(&*table).extended(|r| r.gen(), 0.5);
    let _d1 = etf::DistSymmetric::new(1.0, |x| x, &table); //.sup_extended(|r| Some(r.gen()), 0.5);
    let _d2 = etf::DistAny::new_tailed(|x| x, &table, |rng| Some(rng.gen()), 0.5);
}
