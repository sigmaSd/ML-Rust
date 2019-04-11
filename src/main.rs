mod math;
use math::{mse, Activfn, Math};

struct NeuralNet {
    n1: Neuron,
    n2: Neuron,
    n3: Neuron,
}
impl NeuralNet {
    fn new() -> Self {
        Self {
            n1: Neuron::new(vec![0.0, 1.0], 0.0),
            n2: Neuron::new(vec![0.0, 1.0], 0.0),
            n3: Neuron::new(vec![0.0, 1.0], 0.0),
        }
    }
    fn feedforward(&self, inputs: &[f64]) -> f64 {
        let h1 = self.n1.feedforward(&inputs);
        let h2 = self.n2.feedforward(&inputs);

        self.n3.feedforward(&[h1, h2])

        //dbg!(out);
        //out
    }
}
struct Neuron {
    weights: Vec<f64>,
    bias: f64,
}

impl Neuron {
    fn new(weights: Vec<f64>, bias: f64) -> Self {
        Self { weights, bias }
    }
    fn feedforward(&self, inputs: &[f64]) -> f64 {
        let result = self.weights.dot(inputs) + self.bias;
        //dbg!(result);
        Activfn::sigmoid(result)
    }
}

fn main() {
    let v = [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0]];
    let r = mse(
        0,
        4,
        0.25,
        &v.iter().map(|vv| &vv[..]).collect::<Vec<&[f64]>>(),
    );
    dbg!(r);
    test_neural_net();
}

fn test_neural_net() {
    let neural_net = NeuralNet::new();
    let inputs = vec![2.0, 3.0];

    let _out = neural_net.feedforward(&inputs);
    //dbg!(out);
}
fn _test_neuron() {
    let weights = vec![0.0, 1.0];
    let bias = 0.0;

    let n1 = Neuron::new(weights.clone(), bias);
    let n2 = Neuron::new(weights.clone(), bias);
    let n3 = Neuron::new(weights.clone(), bias);

    let inputs = vec![2.0, 3.0];

    let out1 = n1.feedforward(&inputs);
    let out2 = n2.feedforward(&inputs);

    let out = n3.feedforward(&[out1, out2]);

    dbg!(out);
}
