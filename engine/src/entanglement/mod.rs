use rand::{distributions::WeightedIndex, prelude::Distribution};

/// Simplified evidence bundle used for smoke-testing the entanglement channel.
#[derive(Debug, Clone)]
pub struct EntangledView {
    pub signal: String,
    pub weight: f32,
}

impl EntangledView {
    pub fn from_input(input: &str) -> Self {
        let weights = [0.2, 0.8];
        let dist = WeightedIndex::new(weights).expect("valid weights");
        let mut rng = rand::thread_rng();
        let choice = dist.sample(&mut rng);
        let weight = weights[choice] as f32;
        let signal = format!("{}::weighted", input.trim());
        Self { signal, weight }
    }
}
