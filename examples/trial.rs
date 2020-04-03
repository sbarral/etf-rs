extern crate etf;
extern crate rand;

use etf::*;

fn normal_pdf(x: f32) -> f32 {
    (-x*x).exp()
}
fn normal_dpdf(x: f32) -> f32 {
    -2f32*x*(-x*x).exp()
}

fn main() {
    let partition = util::midpoint_prepartition(&normal_pdf, 2.0, -2.0, 400);
    let table: Box<table::Table32<f32>> = util::newton_tabulation(
        &normal_pdf,
        &normal_dpdf,
        *partition,
        &[0.0],
        1e-2,
        1.0,
        100
    ).unwrap();

    for i in 0..table.ysup.len() {
        println!("{:<20} {:<20} {:<20}", table.x[i], table.ysup[i], table.yinf[i]);
        println!("{:<20} {:<20} {:<20}", table.x[i+1], table.ysup[i], table.yinf[i]);
    }

}
