extern crate fast_vector_similarity;

use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::Array1;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand::Rng; 
use fast_vector_similarity::{spearman_rho, kendall_tau, approximate_distance_correlation, jensen_shannon_dependency_measure, exact_hoeffdings_d_func, normalized_mutual_information};
use rand::SeedableRng; // To create a reproducible RNG
use rand::rngs::StdRng;
use procfs::Meminfo;

fn generate_random_vectors(size: usize, rng: &mut StdRng) -> (Array1<f64>, Array1<f64>) {
    let vector_1: Array1<f64> = Array1::random_using(size, Uniform::new(0., 1.), rng);
    let vector_2: Array1<f64> = Array1::random_using(size, Uniform::new(0., 1.), rng);
    (vector_1, vector_2)
}

fn generate_noisy_linear_vectors(size: usize, rng: &mut StdRng) -> (Array1<f64>, Array1<f64>) {
    let vector_1: Array1<f64> = Array1::random_using(size, Uniform::new(0., 1.), rng); // Reproducible random numbers
    let noise: Array1<f64> = Array1::random_using(size, Uniform::new(-0.1, 0.1), rng); // Reproducible noise
    let vector_2 = &vector_1 * 2.0 + 1.0 + &noise; // Linear dependency with some noise
    (vector_1, vector_2)
}

fn generate_noisy_nonlinear_vectors(size: usize, rng: &mut StdRng) -> (Array1<f64>, Array1<f64>) {
    let vector_1: Array1<f64> = Array1::random_using(size, Uniform::new(0., 1.), rng);
    let noise: Array1<f64> = Array1::random_using(size, Uniform::new(-0.1, 0.1), rng);

    // Use the same seeded RNG to make the choice reproducible
    let choice = rng.gen_range(0..5);

    let vector_2 = match choice {
        0 => {
            println!("Chosen function: Sinusoidal");
            vector_1.mapv(|x| x.sin()) + &noise // Sinusoidal dependency
        }
        1 => {
            println!("Chosen function: Exponential");
            vector_1.mapv(|x| x.exp()) + &noise // Exponential dependency
        }
        2 => {
            println!("Chosen function: Logarithmic");
            vector_1.mapv(|x| x.ln() + 1.0) + &noise // Logarithmic dependency (shifted to avoid undefined values for x <= 0)
        }
        3 => {
            println!("Chosen function: Quadratic");
            vector_1.mapv(|x| x.powi(2)) + &noise // Quadratic dependency
        }
        4 => {
            println!("Chosen function: Cubic");
            vector_1.mapv(|x| x.powi(3)) + &noise // Cubic dependency
        }
        _ => unreachable!(),
    };

    (vector_1, vector_2)
}

fn log_memory_usage() {
    if let Ok(meminfo) = Meminfo::new() {
        let rss = meminfo.active / 1024; // Convert to MB
        println!("Memory Usage: RSS: {} MB", rss);
    } else {
        println!("Failed to retrieve memory usage stats.");
    }
}

fn benchmark_similarity_computations(c: &mut Criterion) {
    let seed = 42;
    let mut rng = StdRng::seed_from_u64(seed);

    let sizes = vec![500, 1000, 2500, 5000]; // Vector sizes to benchmark

    for &size in &sizes {
        let (random_vec1, random_vec2) = generate_random_vectors(size, &mut rng);
        let (linear_vec1, linear_vec2) = generate_noisy_linear_vectors(size, &mut rng);
        let (nonlinear_vec1, nonlinear_vec2) = generate_noisy_nonlinear_vectors(size, &mut rng);

        // Benchmark Spearman Rho for random vectors
        c.bench_function(&format!("Spearman_rho_random_size_{}", size), |b| {
            b.iter(|| spearman_rho(random_vec1.as_slice().unwrap(), random_vec2.as_slice().unwrap()))
        });
        let spearman_rho_result = spearman_rho(random_vec1.as_slice().unwrap(), random_vec2.as_slice().unwrap());
        println!("Spearman_rho (random, size {}): {:.12}", size, spearman_rho_result);
        log_memory_usage();

        // Benchmark Kendall Tau for random vectors
        c.bench_function(&format!("Kendall_tau_random_size_{}", size), |b| {
            b.iter(|| kendall_tau(random_vec1.as_slice().unwrap(), random_vec2.as_slice().unwrap()))
        });
        let kendall_tau_result = kendall_tau(random_vec1.as_slice().unwrap(), random_vec2.as_slice().unwrap());
        println!("Kendall_tau (random, size {}): {:.12}", size, kendall_tau_result);
        log_memory_usage();

        // Benchmark Approximate Distance Correlation for random vectors
        c.bench_function(&format!("Approx_DistCorr_random_size_{}", size), |b| {
            b.iter(|| approximate_distance_correlation(&random_vec1, &random_vec2))
        });
        let approx_distcorr_result = approximate_distance_correlation(&random_vec1, &random_vec2);
        println!("Approx_DistCorr (random, size {}): {:.12}", size, approx_distcorr_result);
        log_memory_usage();

        // Benchmark Jensen-Shannon similarity for random vectors
        c.bench_function(&format!("Jensen_Shannon_random_size_{}", size), |b| {
            b.iter(|| jensen_shannon_dependency_measure(&random_vec1, &random_vec2))
        });
        let jensen_shannon_result = jensen_shannon_dependency_measure(&random_vec1, &random_vec2);
        println!("Jensen_Shannon (random, size {}): {:.12}", size, jensen_shannon_result);
        log_memory_usage();

        // Benchmark Hoeffding's D for random vectors
        c.bench_function(&format!("Hoeffding_D_random_size_{}", size), |b| {
            b.iter(|| exact_hoeffdings_d_func(&random_vec1, &random_vec2))
        });
        let hoeffding_d_result = exact_hoeffdings_d_func(&random_vec1, &random_vec2);
        println!("Hoeffding_D (random, size {}): {:.12}", size, hoeffding_d_result);
        log_memory_usage();

        // Benchmark NMI for random vectors
        c.bench_function(&format!("NMI_random_size_{}", size), |b| {
            b.iter(|| normalized_mutual_information(&random_vec1, &random_vec2))
        });
        let nmi_result = normalized_mutual_information(&random_vec1, &random_vec2);
        println!("NMI (random, size {}): {:.12}", size, nmi_result);
        log_memory_usage();

        // Benchmark Spearman Rho for noisy linear vectors
        c.bench_function(&format!("Spearman_rho_linear_size_{}", size), |b| {
            b.iter(|| spearman_rho(linear_vec1.as_slice().unwrap(), linear_vec2.as_slice().unwrap()))
        });
        let spearman_rho_linear_result = spearman_rho(linear_vec1.as_slice().unwrap(), linear_vec2.as_slice().unwrap());
        println!("Spearman_rho (linear, size {}): {:.12}", size, spearman_rho_linear_result);
        log_memory_usage();

        // Benchmark Kendall Tau for noisy linear vectors
        c.bench_function(&format!("Kendall_tau_linear_size_{}", size), |b| {
            b.iter(|| kendall_tau(linear_vec1.as_slice().unwrap(), linear_vec2.as_slice().unwrap()))
        });
        let kendall_tau_linear_result = kendall_tau(linear_vec1.as_slice().unwrap(), linear_vec2.as_slice().unwrap());
        println!("Kendall_tau (linear, size {}): {:.12}", size, kendall_tau_linear_result);
        log_memory_usage();

        // Benchmark Approximate Distance Correlation for noisy linear vectors
        c.bench_function(&format!("Approx_DistCorr_linear_size_{}", size), |b| {
            b.iter(|| approximate_distance_correlation(&linear_vec1, &linear_vec2))
        });
        let approx_distcorr_linear_result = approximate_distance_correlation(&linear_vec1, &linear_vec2);
        println!("Approx_DistCorr (linear, size {}): {:.12}", size, approx_distcorr_linear_result);
        log_memory_usage();

        // Benchmark Jensen-Shannon similarity for noisy linear vectors
        c.bench_function(&format!("Jensen_Shannon_linear_size_{}", size), |b| {
            b.iter(|| jensen_shannon_dependency_measure(&linear_vec1, &linear_vec2))
        });
        let jensen_shannon_linear_result = jensen_shannon_dependency_measure(&linear_vec1, &linear_vec2);
        println!("Jensen_Shannon (linear, size {}): {:.12}", size, jensen_shannon_linear_result);
        log_memory_usage();

        // Benchmark Hoeffding's D for noisy linear vectors
        c.bench_function(&format!("Hoeffding_D_linear_size_{}", size), |b| {
            b.iter(|| exact_hoeffdings_d_func(&linear_vec1, &linear_vec2))
        });
        let hoeffding_d_linear_result = exact_hoeffdings_d_func(&linear_vec1, &linear_vec2);
        println!("Hoeffding_D (linear, size {}): {:.12}", size, hoeffding_d_linear_result);
        log_memory_usage();

        // Benchmark NMI for noisy linear vectors
        c.bench_function(&format!("NMI_linear_size_{}", size), |b| {
            b.iter(|| normalized_mutual_information(&linear_vec1, &linear_vec2))
        });
        let nmi_linear_result = normalized_mutual_information(&linear_vec1, &linear_vec2);
        println!("NMI (linear, size {}): {:.12}", size, nmi_linear_result);
        log_memory_usage();

        // Benchmark Spearman Rho for noisy nonlinear vectors
        c.bench_function(&format!("Spearman_rho_nonlinear_size_{}", size), |b| {
            b.iter(|| spearman_rho(nonlinear_vec1.as_slice().unwrap(), nonlinear_vec2.as_slice().unwrap()))
        });
        let spearman_rho_nonlinear_result = spearman_rho(nonlinear_vec1.as_slice().unwrap(), nonlinear_vec2.as_slice().unwrap());
        println!("Spearman_rho (nonlinear, size {}): {:.12}", size, spearman_rho_nonlinear_result);
        log_memory_usage();

        // Benchmark Kendall Tau for noisy nonlinear vectors
        c.bench_function(&format!("Kendall_tau_nonlinear_size_{}", size), |b| {
            b.iter(|| kendall_tau(nonlinear_vec1.as_slice().unwrap(), nonlinear_vec2.as_slice().unwrap()))
        });
        let kendall_tau_nonlinear_result = kendall_tau(nonlinear_vec1.as_slice().unwrap(), nonlinear_vec2.as_slice().unwrap());
        println!("Kendall_tau (nonlinear, size {}): {:.12}", size, kendall_tau_nonlinear_result);
        log_memory_usage();

        // Benchmark Approximate Distance Correlation for noisy nonlinear vectors
        c.bench_function(&format!("Approx_DistCorr_nonlinear_size_{}", size), |b| {
            b.iter(|| approximate_distance_correlation(&nonlinear_vec1, &nonlinear_vec2))
        });
        let approx_distcorr_nonlinear_result = approximate_distance_correlation(&nonlinear_vec1, &nonlinear_vec2);
        println!("Approx_DistCorr (nonlinear, size {}): {:.12}", size, approx_distcorr_nonlinear_result);
        log_memory_usage();

        // Benchmark Jensen-Shannon similarity for noisy nonlinear vectors
        c.bench_function(&format!("Jensen_Shannon_nonlinear_size_{}", size), |b| {
            b.iter(|| jensen_shannon_dependency_measure(&nonlinear_vec1, &nonlinear_vec2))
        });
        let jensen_shannon_nonlinear_result = jensen_shannon_dependency_measure(&nonlinear_vec1, &nonlinear_vec2);
        println!("Jensen_Shannon (nonlinear, size {}): {:.12}", size, jensen_shannon_nonlinear_result);
        log_memory_usage();

        // Benchmark Hoeffding's D for noisy nonlinear vectors
        c.bench_function(&format!("Hoeffding_D_nonlinear_size_{}", size), |b| {
            b.iter(|| exact_hoeffdings_d_func(&nonlinear_vec1, &nonlinear_vec2))
        });
        let hoeffding_d_nonlinear_result = exact_hoeffdings_d_func(&nonlinear_vec1, &nonlinear_vec2);
        println!("Hoeffding_D (nonlinear, size {}): {:.12}", size, hoeffding_d_nonlinear_result);
        log_memory_usage();

        // Benchmark NMI for noisy nonlinear vectors
        c.bench_function(&format!("NMI_nonlinear_size_{}", size), |b| {
            b.iter(|| normalized_mutual_information(&nonlinear_vec1, &nonlinear_vec2))
        });
        let nmi_nonlinear_result = normalized_mutual_information(&nonlinear_vec1, &nonlinear_vec2);
        println!("NMI (nonlinear, size {}): {:.12}", size, nmi_nonlinear_result);
        log_memory_usage();
    }
}

// Criterion setup
fn criterion_benchmark(c: &mut Criterion) {
    benchmark_similarity_computations(c);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
