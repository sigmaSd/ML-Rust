trait VecTools<T> {
    fn map_in_place(&mut self, f: impl FnMut(&T) -> T);
}

impl<T> VecTools<T> for &mut [T] {
    fn map_in_place(&mut self, mut f: impl FnMut(&T) -> T) {
        for item in self.iter_mut() {
            *item = f(&item);
        }
    }
}

pub trait Math {
    fn dot(&self, other: &[f64]) -> f64;
    fn mul_x(&mut self, other: f64) -> &mut [f64];
    fn mul_vec(&mut self, other: &[f64]) -> &mut [f64];
    fn minus_vec(&mut self, other: &[f64]) -> &mut [f64];
}

impl Math for &mut [f64] {
    fn dot(&self, other: &[f64]) -> f64 {
        self.iter().zip(other).fold(0.0, |acc, (x, y)| acc + x * y)
    }
    fn mul_x(&mut self, other: f64) -> &mut [f64] {
        self.map_in_place(|v| *v * other);

        self
    }
    fn mul_vec(&mut self, other: &[f64]) -> &mut [f64] {
        let mut other = other.iter();
        self.map_in_place(|v| v * other.next().unwrap());

        self
    }
    fn minus_vec(&mut self, other: &[f64]) -> &mut [f64] {
        let mut other = other.iter();
        self.map_in_place(|v| v - other.next().unwrap());

        self
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
