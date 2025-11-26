// #[pyfunction]
// fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
//     Ok((a + b).to_string())
// }
use pyo3::pymodule;
#[pymodule]
mod paralelize {
    use pyo3::{prelude::*, pyfunction};
    use rayon::{
        iter::{IntoParallelRefIterator, ParallelIterator},
        ThreadPoolBuilder,
    };
    use std::{
        cmp::max,
        collections::{HashMap, HashSet, VecDeque},
        iter::zip,
    };
    #[pyclass]
    pub struct Network {
        pub base_capacity: u32,
        pub encrypted_cap: u32,
        mst: Vec<(u32, u32)>,
        pub edges: HashMap<(u32, u32), i32>,
        nodes: Vec<u32>,
        // TODO This need interior mut
        // The rest can just be cloned
        pub shortest_paths: HashMap<(u32, u32), Vec<Vec<(u32)>>>,
    }
    #[pymethods]
    impl Network {
        #[new]
        fn new(
            base_capacity: u32,
            encrypted_capacity: u32,
            edges: Vec<(u32, u32)>,
            nodes: Vec<u32>,
        ) -> Self {
            let mut local_edges: HashMap<(u32, u32), i32> = edges
                .into_iter()
                .map(|(u, v)| ((u, v), base_capacity as i32))
                .collect();
            let doubled_edges: HashMap<(u32, u32), i32> = local_edges
                .iter()
                .map(|((u, v), cap)| ((v.clone(), u.clone()), cap.clone()))
                .collect();
            local_edges.extend(doubled_edges);
            Network {
                base_capacity,
                encrypted_cap: encrypted_capacity,
                mst: Vec::new(),
                edges: local_edges,
                nodes: nodes,
                shortest_paths: HashMap::new(),
            }
        }
        #[getter]
        pub fn edges(&self) -> HashMap<(u32, u32), i32> {
            self.edges.clone()
        }
        #[getter]
        pub fn mst(&self) -> Vec<(u32, u32)> {
            self.mst.clone()
        }
        #[getter]
        pub fn base_capacity(&self) -> u32 {
            self.base_capacity
        }
        #[getter]
        pub fn encrypted_cap(&self) -> u32 {
            self.encrypted_cap
        }

        pub fn get_edge_use(
            &self,
            chromosome: Vec<u32>,
            transceivers: Vec<(u32, u32)>,
        ) -> HashMap<(u32, u32), i32> {
            let mut local_edges = self.edges.clone();
            let deployments = chromosome_to_deployments(chromosome, &self, transceivers);
            apply_transceivers(&self, &deployments, &mut local_edges);
            local_edges
        }
        pub fn set_shortest_paths(&mut self) {
            let pairs = make_pairs(&self);
            let mut local_edges = self.edges.clone();
            pairs.iter().for_each(|(src, dst)| {
                self.shortest_paths.insert(
                    (*src, *dst),
                    vec![shortest_path(&self, &mut local_edges, *src, *dst).unwrap()],
                );
            });
        }
        fn neighbors(&self, node: u32) -> Vec<&u32> {
            self.edges
                .iter()
                .filter_map(|((u, v), cap)| match node == *u {
                    true => Some(v),
                    false => None,
                })
                .collect()
        }
        fn reset_capacities(&mut self) {
            for ((_, _), cap) in self.edges.iter_mut() {
                *cap = self.base_capacity as i32;
            }
            self.encrypt_edges()
        }
        fn set_mst(&mut self, mst: Vec<(u32, u32)>) {
            self.mst = mst;
        }
        fn encrypt_edges(&mut self) {
            for (u, v) in self.mst.iter() {
                let edge = self.edges.get_mut(&(*u, *v)).unwrap();
                *edge = self.encrypted_cap as i32;
                let edge2 = self.edges.get_mut(&(*v, *u)).unwrap();
                *edge2 = self.encrypted_cap as i32;
            }
        }
    }

    // #[pyfunction]
    fn shortest_path(
        network: &Network,
        local_edges: &mut HashMap<(u32, u32), i32>,
        src: u32,
        dst: u32,
    ) -> Option<Vec<u32>> {
        let mut queue = VecDeque::new();
        queue.push_back((src.clone(), vec![src.clone()]));
        let mut visited = HashSet::new();
        visited.insert(src);
        while !queue.is_empty() {
            let (node, path) = queue.pop_front().unwrap();
            if node == dst {
                return Some(path);
            }
            for neighbour in network.neighbors(node).iter() {
                if !visited.contains(&neighbour)
                    && *local_edges.get(&(node, **neighbour)).unwrap() > 0
                {
                    visited.insert(**neighbour);
                    let mut new_path = path.clone();
                    new_path.push(**neighbour);
                    queue.push_back((**neighbour, new_path));
                }
            }
        }
        None
    }
    fn dry_run_edge_count_change(edges: &mut HashMap<(u32, u32), i32>, path: &Vec<u32>) -> bool {
        let mut flag = true;
        let path1 = path[0..path.len() - 1].iter();
        let path2 = path[1..path.len()].iter();
        let zipped_paths = zip(path1, path2);
        for (u, v) in zipped_paths.clone() {
            *edges.get_mut(&(*u, *v)).unwrap() -= 1;
            let edge = edges.get_mut(&(*v, *u)).unwrap();
            *edge -= 1;
            if *edge < 0 {
                flag = false;
            }
        }
        if !flag {
            for (u, v) in zipped_paths {
                *edges.get_mut(&(*u, *v)).unwrap() += 1;
                *edges.get_mut(&(*v, *u)).unwrap() += 1;
            }
        }
        return flag;
    }

    fn make_pairs(network: &Network) -> Vec<(u32, u32)> {
        let node_amount = network.nodes.len();
        let mut pairs = Vec::new();
        for i in 0..node_amount {
            for j in (i + 1)..node_amount {
                pairs.push((i as u32, j as u32));
            }
        }
        pairs
    }

    fn apply_transceivers(
        network: &Network,
        deployments: &Vec<(u32, u32, u32, u32)>,
        mut local_edges: &mut HashMap<(u32, u32), i32>,
    ) -> Option<(u32, u32)> {
        for (src, dst, t_type, count) in deployments.iter() {
            let mut path_idx = 0;
            let mut reversed = false;
            let mut path_arr = network.shortest_paths.get(&(*src, *dst));
            if path_arr.is_none() {
                path_arr = network.shortest_paths.get(&(*dst, *src));
                reversed = true;
            }
            if path_arr.is_none() {
                return Some((*src, *dst));
            }
            let (mut next_path, mut path_arr_len) = {
                let path_arr_local = path_arr.unwrap();
                (path_arr_local[0].clone(), path_arr_local.len())
            };
            for _ in 0..*count {
                while !dry_run_edge_count_change(&mut local_edges, &next_path) {
                    // if path_arr_len > path_idx + 1 {
                    //     path_idx += 1;
                    //     let key = match reversed {
                    //         false => (*src, *dst),
                    //         true => (*dst, *src),
                    //     };
                    //     next_path = network.shortest_paths.get(&key).unwrap()[path_idx].clone();
                    // } else {
                    let path_opt = shortest_path(network, &mut local_edges, *src, *dst);
                    if path_opt.is_none() {
                        return Some((*src, *dst));
                    }
                    next_path = path_opt.unwrap();
                    // println!("Found next path");
                    if path_idx > 10 {
                        return Some((*src, *dst));
                    }
                    path_idx += 1;
                    // network
                    //     .shortest_paths
                    //     .get_mut(&(*src, *dst))
                    //     .unwrap()
                    //     .push(path_opt.unwrap());
                    // path_arr_len += 1;
                    // }
                }
            }
        }
        None
    }
    #[pyfunction]
    fn chromosome_to_deployments(
        chromosome: Vec<u32>,
        network: &Network,
        transceivers: Vec<(u32, u32)>,
    ) -> Vec<(u32, u32, u32, u32)> {
        let pairs = make_pairs(network);
        let mut idx = 0;
        let mut deployments = Vec::new();
        for (src, dst) in pairs {
            for (t_type) in transceivers.iter() {
                let count = chromosome[idx];
                idx += 1;
                if count > 0 {
                    deployments.push((src, dst, t_type.0, count))
                }
            }
        }
        deployments
    }
    #[pyfunction]
    pub fn fitness_for_deployments(
        network: &Network,
        chromosome: Vec<u32>,
        per_node_demand: u32,
        transceivers: Vec<(u32, u32)>,
        w_unconnected: f64,
        w_cost: f64,
        w_unmet: f64,
        w_overflow: f64,
        w_possible_penalty: f64,
        allowed_overlfow: i32,
    ) -> PyResult<(
        f64,
        i32,
        i32,
        bool,
        i32,
        Vec<(usize, i32)>,
        Option<(u32, u32)>,
    )> {
        let mut possible_penalty = 0.;
        let t_amount = chromosome.par_iter().sum::<u32>() as f64;
        let deployments =
            chromosome_to_deployments(chromosome.clone(), network, transceivers.clone());
        let mut local_edges = network.edges.clone();
        let transceiver_result = apply_transceivers(network, &deployments, &mut local_edges);
        let mut overflowing_pair = None;
        if transceiver_result.is_some() {
            possible_penalty = w_possible_penalty * t_amount;
            overflowing_pair = transceiver_result;
        }
        let transceiver_type_amount = transceivers.len();
        let unconnected_nodes: u32 = chromosome
            .chunks_exact(transceiver_type_amount)
            .map(|x| match x.par_iter().sum::<u32>() == 0 {
                true => 1,
                false => 0,
            })
            .sum();
        let cost: i32 = chromosome
            .chunks_exact(transceiver_type_amount)
            .map(|x| {
                x.iter()
                    .enumerate()
                    .map(|(idx, t_amount)| t_amount * transceivers[idx].1)
                    .sum::<u32>() as i32
            })
            .sum();
        let mut nodes_requiring_attention = Vec::new();
        let unmet: i32 = chromosome
            .chunks_exact(transceiver_type_amount)
            .enumerate()
            .map(|(idx, x)| {
                let unmet_per_node = (per_node_demand as i32)
                    - (x.iter()
                        .enumerate()
                        .map(|(idx, t_amount)| t_amount * transceivers[idx].0)
                        .sum::<u32>() as i32);
                if unmet_per_node > 0 {
                    nodes_requiring_attention.push((idx, 1));
                }
                if unmet_per_node < -allowed_overlfow {
                    nodes_requiring_attention.push((idx, -1));
                }
                max(unmet_per_node, 0)
            })
            .sum();
        let overflow: i32 = chromosome
            .chunks_exact(transceiver_type_amount)
            .map(|x| {
                max(
                    (x.iter()
                        .enumerate()
                        .map(|(idx, t_amount)| t_amount * transceivers[idx].0)
                        .sum::<u32>() as i32)
                        - (per_node_demand as i32),
                    0,
                )
            })
            .sum();
        let met_reward = match unmet == 0 {
            true => 50000. * (chromosome.len() as f64),
            false => 0.,
        };
        Ok((
            // ((unconnected_nodes as f64) * w_unconnected
            (cost as f64) * w_cost,
            // + (unmet as f64).powf(2.) * w_unmet
            // + (overflow as f64).powf(2.) * w_overflow
            // + possible_penalty
            // + met_reward),
            unconnected_nodes as i32,
            unmet as i32,
            possible_penalty == 0.,
            overflow,
            nodes_requiring_attention,
            overflowing_pair,
        ))
    }
    #[pyfunction]
    fn appraise_pop(
        network: &mut Network,
        pop: Vec<Vec<u32>>,
        per_node_demand: u32,
        transceivers: Vec<(u32, u32)>,
        w_unconnected: f64,
        w_cost: f64,
        w_unmet: f64,
        w_overflow: f64,
        w_possible_penalty: f64,
        allowed_overflow: i32,
    ) -> PyResult<
        Vec<(
            f64,
            i32,
            i32,
            bool,
            i32,
            Vec<(usize, i32)>,
            Option<(u32, u32)>,
        )>,
    > {
        // if let Ok(_) = init_thread_pool() {
        // } else {
        // }
        Ok(pop
            .par_iter()
            .map(|c| {
                fitness_for_deployments(
                    network,
                    c.clone(),
                    per_node_demand,
                    transceivers.clone(),
                    w_unconnected,
                    w_cost,
                    w_unmet,
                    w_overflow,
                    w_possible_penalty,
                    allowed_overflow,
                )
                .unwrap()
            })
            .collect())
    }
    fn init_thread_pool() -> Result<(), rayon::ThreadPoolBuildError> {
        ThreadPoolBuilder::new().num_threads(20).build_global()
    }
}
