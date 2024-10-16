use rand::Rng;

const CHECK_CYCLES: u32 = 3; // Check if the queue is full, it's not
const INSERT_EASY_CYCLES: u32 = 4; // Queue not full
const INSERT_HARD_CYCLES: u32 = 5; // Need to remove an item from the queue

struct TopK {
    busy_cycles: u32,
    queue: Vec<(f64, u32)>,
}

impl TopK {
    fn new() -> Self {
        TopK {
            busy_cycles: 0,
            queue: Vec::new(),
        }
    }

    // Return the number of cycles that we need to stall
    fn check(&mut self, score: f64, id: u32) -> u32 {
        let out = self.busy_cycles;
        if self.queue.len() < 32 {
            self.busy_cycles = INSERT_EASY_CYCLES;
            self.queue.push((score, id));
            self.queue.sort_by(|a, b| a.partial_cmp(b).unwrap());
        } else if score > self.queue[0].0 {
            self.busy_cycles = INSERT_HARD_CYCLES;
            self.queue[0] = (score, id);
            self.queue.sort_by(|a, b| a.partial_cmp(b).unwrap());
        } else {
            self.busy_cycles = CHECK_CYCLES;
        }
        out
    }
}

struct PE {
    topk: TopK,
    updating_cycles: u32,
    d: u32,
    mac_units: u32,
}

impl PE {
    fn new(d: u32, mac_units: u32) -> Self {
        PE {
            topk: TopK::new(),
            updating_cycles: 0,
            d,
            mac_units,
        }
    }

    // Return number of stall cycles, number of useful cycles
    fn run_one_batch(&mut self, offset: u32, corpus_batch_size: Option<u32>) -> (u32, u32) {
        let corpus_batch_size = corpus_batch_size.unwrap_or(self.mac_units);
        assert!(corpus_batch_size <= self.mac_units);
        let base_cycles = self.d;
        let mut top_k_cycles = 0;
        let prev_stall = self.updating_cycles;
        let mut rng = rand::thread_rng();

        for m in 0..corpus_batch_size {
            let id = m + offset * self.mac_units;
            let score = rng.gen::<f64>();
            top_k_cycles += self.topk.check(score, id);
        }

        self.updating_cycles = top_k_cycles;
        if prev_stall > base_cycles {
            (prev_stall - base_cycles, base_cycles)
        } else {
            (0, base_cycles)
        }
    }
}

struct NMA {
    pe_array: Vec<PE>,
    d: u32,
    mac_units: u32,
    num_vectors: u32,
}

impl NMA {
    fn new(d: u32, mac_units: u32, num_pe: u32) -> Self {
        let pe_array = (0..num_pe).map(|_| PE::new(d, mac_units)).collect();
        NMA {
            pe_array,
            d,
            mac_units,
            num_vectors: 0,
        }
    }

    fn run_one_batch(
        &mut self,
        offset: u32,
        query_batch_size: usize,
        corpus_batch_size: Option<u32>,
    ) -> (u32, u32) {
        let mut cycles = 0;
        let mut stall_cycles = 0;
        for pe in self.pe_array.iter_mut().take(query_batch_size) {
            let (stall, useful) = pe.run_one_batch(offset, corpus_batch_size);
            cycles = useful; // Should be the same for all
            if stall > stall_cycles {
                stall_cycles = stall;
            }
        }
        (stall_cycles, cycles)
    }

    fn run_batches(&mut self, total_vectors: u32, query_batch_size: usize) -> (u32, u32) {
        let num_full_batches = total_vectors / self.mac_units;
        let remainder = total_vectors % self.mac_units;
        let mut cycles = 0;
        let mut stall_cycles = 0;

        for i in 0..num_full_batches {
            let (stall, useful) = self.run_one_batch(i, query_batch_size, None);
            cycles += useful;
            stall_cycles += stall;
        }

        if remainder > 0 {
            let (stall, useful) = self.run_one_batch(num_full_batches, query_batch_size, Some(remainder));
            cycles += useful;
            stall_cycles += stall;
        }

        (stall_cycles, cycles)
    }

    fn store_vectors(&mut self, num_vectors: u32) {
        self.num_vectors = num_vectors;
    }

    fn run_search(&mut self, query_batch_size: usize) -> (u32, u32) {
        self.run_batches(self.num_vectors, query_batch_size)
    }
}

struct IKS {
    nma_array: Vec<NMA>,
    d: u32,
    mac_units: u32,
}

impl IKS {
    fn new(d: u32, mac_units: u32, num_pe: u32, num_nma: u32) -> Self {
        let nma_array = (0..num_nma).map(|_| NMA::new(d, mac_units, num_pe)).collect();
        IKS {
            nma_array,
            d,
            mac_units,
        }
    }

    fn store_vectors(&mut self, num_vectors: u32) {
        let num_nma = self.nma_array.len() as u32;
        let mut vectors_per_nma = vec![num_vectors / num_nma; num_nma as usize];
        let remainder = num_vectors % num_nma;
        for i in 0..remainder as usize {
            vectors_per_nma[i] += 1;
        }
        for (nma, &vectors) in self.nma_array.iter_mut().zip(vectors_per_nma.iter()) {
            nma.store_vectors(vectors);
        }
    }

    fn run_search(&mut self, query_batch_size: usize) -> (u32, u32) {
        let mut stall = 0;
        let mut useful = 0;
        for nma in self.nma_array.iter_mut() {
            let (s, u) = nma.run_search(query_batch_size);
            if s > stall {
                stall = s;
            }
            if u > useful {
                useful = u;
            }
        }
        (stall, useful)
    }
}

fn main() {
    let mut iks = IKS::new(768, 68, 64, 8);
    iks.store_vectors(35_678_076);
    let (stall, useful) = iks.run_search(32);
    let total = (stall + useful) as f64;
    println!("Total time: {} ms", total / 1_000_000.0);
    println!("Stalled for: {}% of all cycles", stall as f64 / total * 100.0);


}

