pub trait Math {
    fn dot(&self, other: &[f64]) -> f64;
    fn mul_x(&self, other: f64) -> Vec<f64>;
    fn mul_vec(&self, other: &[f64]) -> Vec<f64>;
    fn minus_vec(&self, other: &[f64]) -> Vec<f64>;
}

impl Math for Vec<f64> {
    fn dot(&self, other: &[f64]) -> f64 {
        self.iter().zip(other).fold(0.0, |acc, (x, y)| acc + x * y)
    }
    fn mul_x(&self, other: f64) -> Vec<f64> {
        self.iter().map(|v| v * other).collect()
    }
    fn mul_vec(&self, other: &[f64]) -> Vec<f64> {
        self.iter().zip(other).map(|(a, b)| a * b).collect()
    }
    fn minus_vec(&self, other: &[f64]) -> Vec<f64> {
        self.iter().zip(other).map(|(a, b)| a - b).collect()
    }
}
impl Math for &[f64] {
    fn dot(&self, other: &[f64]) -> f64 {
        self.iter().zip(other).fold(0.0, |acc, (x, y)| acc + x * y)
    }
    fn mul_x(&self, other: f64) -> Vec<f64> {
        self.iter().map(|v| v * other).collect()
    }
    fn mul_vec(&self, other: &[f64]) -> Vec<f64> {
        self.iter().zip(other).map(|(a, b)| a * b).collect()
    }
    fn minus_vec(&self, other: &[f64]) -> Vec<f64> {
        self.iter().zip(other).map(|(a, b)| a - b).collect()
    }
}

pub enum Activfn {}
impl Activfn {
    pub fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
    pub fn deriv_sigmoid(x: f64) -> f64 {
        let fx = Activfn::sigmoid(x);
        fx * (1. - fx)
    }
}

pub fn mse(i: usize, n: usize, a: f64, input: Vec<Vec<f64>>) -> f64 {
    let f = |x: &[f64]| (x[0] - x[1]).powf(2.0);
    a * sum(i, n, f, input)
}

fn sum(i: usize, n: usize, f: impl Fn(&[f64]) -> f64, inputs: Vec<Vec<f64>>) -> f64 {
    let mut s = 0.0;
    for input in inputs.iter().take(n).skip(i) {
        s += f(input);
    }
    s
}
