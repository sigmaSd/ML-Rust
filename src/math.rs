pub trait Math {
    fn dot(&self, other: &[f64]) -> f64;
}

impl Math for Vec<f64> {
    fn dot(&self, other: &[f64]) -> f64 {
        self.iter().zip(other).fold(0.0, |acc, (x, y)| acc + x * y)
    }
}

pub enum Activfn {}
impl Activfn {
    pub fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
}

pub fn mse(i: usize, n: usize, a: f64, input: &[&[f64]]) -> f64 {
    let f = |x: &[f64]| (x[0] - x[1]).powf(2.0);
    a * sum(i, n, f, input)
}

fn sum(i: usize, n: usize, f: impl Fn(&[f64]) -> f64, inputs: &[&[f64]]) -> f64 {
    let mut s = 0.0;
    for input in inputs.iter().take(n).skip(i) {
        s += f(input);
    }
    s
}
