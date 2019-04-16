mod math;
use math::{mse, Activfn, Math};

const LEARNING_RATE: f64 = 0.01;
const EPOCH: usize = 100_000;

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
    fn feedforward(&mut self, inputs: &[f64]) -> f64 {
        let h1 = self.n1.feedforward(&inputs);
        let h2 = self.n2.feedforward(&inputs);

        let h1h2 = vec![h1, h2];
        self.n3.feedforward(&h1h2)

        //dbg!(out);
        //out
    }
    fn train(&mut self, data: Vec<Vec<f64>>, all_y_true: Vec<f64>) {
        for _ in 0..EPOCH {
            let mut all_y_preds = Vec::new();
            for (x, y_true) in data.iter().zip(all_y_true.iter()) {
                let y_pred = self.feedforward(x);
                all_y_preds.push(y_pred);

                self.backpropagate(y_pred, *y_true);
            }
            let loss = Self::calculate_loss(&all_y_preds, &all_y_true);
            println!("{}", loss);
        }
    }
    fn backpropagate(&mut self, y_pred: f64, y_true: f64) {
        let d_l_d_ypred = -2. * (y_true - y_pred);

        let n3 = self.n3.backpropagate(d_l_d_ypred, None);
        let n2 = self.n2.backpropagate(d_l_d_ypred, Some(self.n3.weights[0]));
        let n1 = self.n1.backpropagate(d_l_d_ypred, Some(self.n3.weights[1]));

        self.n3 = n3.unwrap();
        self.n2 = n2.unwrap();
        self.n1 = n1.unwrap();
    }
    fn calculate_loss(all_y_preds: &[f64], all_y_true: &[f64]) -> f64 {
        mse(
            0,
            all_y_preds.len(),
            0.25,
            all_y_true
                .iter()
                .zip(all_y_preds)
                .map(|(a, b)| [*a, *b].to_vec())
                .collect::<Vec<Vec<f64>>>(),
        )
    }
}

#[derive(Default)]
struct Neuron {
    weights: Vec<f64>,
    bias: f64,
    input: Option<Vec<f64>>,
    output: Option<f64>,
    sum: Option<f64>,
}

impl Neuron {
    fn new(weights: Vec<f64>, bias: f64) -> Self {
        Self {
            weights,
            bias,
            input: None,
            output: None,
            sum: None,
        }
    }
    fn feedforward(&mut self, inputs: &[f64]) -> f64 {
        self.input = Some(inputs.to_vec());

        let result = self.weights.as_slice().dot(inputs) + self.bias;

        self.sum = Some(result);
        //dbg!(result);
        let out = Activfn::sigmoid(result);
        self.output = Some(out);

        Activfn::sigmoid(result)
    }
    fn backpropagate(&mut self, d_l_d_ypred: f64, out_weight: Option<f64>) -> Option<Neuron> {
        let d_ypred_d_b = Activfn::deriv_sigmoid(self.sum?);

        let d_ypred_d_w: Vec<f64> = self.input.as_ref()?.as_slice().mul_x(d_ypred_d_b);

        let d_ypred_d_h = if let Some(out_weight) = out_weight {
            out_weight * d_ypred_d_b
        } else {
            1.
        };

        let weights = self.weights.as_slice().minus_vec(
            &d_ypred_d_w
                .as_slice()
                .mul_x(d_l_d_ypred)
                .as_slice()
                .mul_x(LEARNING_RATE)
                .as_slice()
                .mul_x(d_ypred_d_h),
        );

        let bias = self.bias - d_ypred_d_b * d_l_d_ypred * LEARNING_RATE * d_ypred_d_h;

        Some(Neuron {
            weights,
            bias,
            ..Default::default()
        })
    }
}

fn _test_neural_net() {
    let mut neural_net = NeuralNet::new();
    let inputs = vec![2.0, 3.0];

    let _out = neural_net.feedforward(&inputs);
    //dbg!(out);
}
fn _test_neuron() {
    let weights = vec![0.0, 1.0];
    let bias = 0.0;

    let mut n1 = Neuron::new(weights.clone(), bias);
    let mut n2 = Neuron::new(weights.clone(), bias);
    let mut n3 = Neuron::new(weights.clone(), bias);

    let inputs = vec![2.0, 3.0];

    let out1 = n1.feedforward(&inputs);
    let out2 = n2.feedforward(&inputs);

    let out = n3.feedforward(&[out1, out2]);

    dbg!(out);
}

fn main() {
    let data = vec![
        vec![-2., -1.],
        vec![25., 6.],
        vec![17., 4.],
        vec![-15., -6.],
    ];

    let all_y_trues = vec![1., 0., 0., 1.];

    let mut network = NeuralNet::new();
    network.train(data, all_y_trues);

    // let emily = vec![-7., -3.];// # 128 pounds, 63 inches
    // let frank = vec![20., 2.];

    // dbg!(network.feedforward(&emily));
    // dbg!(network.feedforward(&frank));
}
