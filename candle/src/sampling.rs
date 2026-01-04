use candle_core::{DType, Result, Tensor};
use rand::{distributions::Distribution, thread_rng};

pub struct LogitsProcessor {
    rng: rand::rngs::ThreadRng,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<usize>,
    min_p: Option<f64>,
}

impl LogitsProcessor {
    pub fn new(
        _seed: u64,
        temperature: Option<f64>,
        top_p: Option<f64>,
        top_k: Option<usize>,
    ) -> Self {
        Self::new_with_min_p(_seed, temperature, top_p, top_k, None)
    }

    pub fn new_with_min_p(
        _seed: u64,
        temperature: Option<f64>,
        top_p: Option<f64>,
        top_k: Option<usize>,
        min_p: Option<f64>,
    ) -> Self {
        // In a real implementation we might use a seeded RNG, but thread_rng is fine for now
        // if the user provided a seed, we could use StdRng::seed_from_u64(seed)
        Self {
            rng: thread_rng(),
            temperature,
            top_p,
            top_k,
            min_p,
        }
    }

    pub fn apply_repetition_penalty(&self, logits: &mut Vec<f32>, tokens: &[u32], penalty: f32) {
        if penalty == 1.0 {
            return;
        }
        let mut seen = std::collections::HashSet::new();
        for &t in tokens {
            if seen.insert(t) {
                let logit = logits[t as usize];
                if logit < 0.0 {
                    logits[t as usize] = logit * penalty;
                } else {
                    logits[t as usize] = logit / penalty;
                }
            }
        }
    }

    pub fn sample(
        &mut self,
        logits: &Tensor,
        prev_tokens: &[u32],
        repetition_penalty: Option<f32>,
    ) -> Result<u32> {
        let logits = logits.to_dtype(DType::F32)?;
        let device = logits.device();
        let mut logits_v: Vec<f32> = logits.to_vec1()?;

        // 0. Repetition penalty
        if let Some(penalty) = repetition_penalty {
            self.apply_repetition_penalty(&mut logits_v, prev_tokens, penalty);
        }

        // 1. Temperature scaling
        if let Some(temp) = self.temperature {
            if temp > 0.0 && temp != 1.0 {
                for l in logits_v.iter_mut() {
                    *l /= temp as f32;
                }
            }
        }

        // 2. Top-K filtering
        if let Some(k) = self.top_k {
            if k > 0 && k < logits_v.len() {
                let mut indexed_logits: Vec<(usize, f32)> =
                    logits_v.iter().enumerate().map(|(i, &l)| (i, l)).collect();
                indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                let min_kept_logit = indexed_logits[k - 1].1;
                for l in logits_v.iter_mut() {
                    if *l < min_kept_logit {
                        *l = f32::NEG_INFINITY;
                    }
                }
            }
        }

        // Softmax
        let logits_t = Tensor::from_vec(logits_v, (logits.dim(0)?,), device)?;
        let probs = candle_nn::ops::softmax(&logits_t, 0)?;
        let mut probs_v: Vec<f32> = probs.to_vec1()?;

        // 3. Min-P filtering: zero out probabilities below max_prob * min_p threshold
        if let Some(min_p_threshold) = self.min_p {
            if min_p_threshold > 0.0 {
                let max_prob = probs_v.iter().cloned().fold(0.0f32, f32::max);
                let threshold = max_prob * min_p_threshold as f32;
                for p in probs_v.iter_mut() {
                    if *p < threshold {
                        *p = 0.0;
                    }
                }
                // Re-normalize
                let sum: f32 = probs_v.iter().sum();
                if sum > 0.0 {
                    for p in probs_v.iter_mut() {
                        *p /= sum;
                    }
                }
            }
        }

        // 3. Top-P (Nucleus) filtering (applied to probabilities)
        let sampled_idx = if let Some(p) = self.top_p {
            if p > 0.0 && p < 1.0 {
                let mut indexed_probs: Vec<(usize, f32)> =
                    probs_v.iter().enumerate().map(|(i, &pr)| (i, pr)).collect();
                indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                let mut cumulative_prob = 0.0;
                let mut cutoff_idx = indexed_probs.len();
                for (i, (_, pr)) in indexed_probs.iter().enumerate() {
                    cumulative_prob += pr;
                    if cumulative_prob > p as f32 {
                        cutoff_idx = i + 1;
                        break;
                    }
                }

                // Re-normalize top-p
                let top_indexed = &indexed_probs[..cutoff_idx];
                let sum: f32 = top_indexed.iter().map(|(_, pr)| pr).sum();

                let dist = rand::distributions::WeightedIndex::new(
                    top_indexed.iter().map(|(_, pr)| pr / sum),
                )
                .unwrap();
                top_indexed[dist.sample(&mut self.rng)].0
            } else {
                self.simple_sample(&probs_v)
            }
        } else {
            self.simple_sample(&probs_v)
        };

        Ok(sampled_idx as u32)
    }

    fn simple_sample(&mut self, probs: &[f32]) -> usize {
        let dist = rand::distributions::WeightedIndex::new(probs).unwrap();
        dist.sample(&mut self.rng)
    }
}
