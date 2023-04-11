
use mat_mult::matrix::FloatMatrix;
use std::time::Instant;

fn main() {
    let mut test_matrices_simd_1 = vec![];
    let mut test_matrices_simd_2 = vec![];

    for i in 0..200usize {
        test_matrices_simd_1.push(FloatMatrix::new(vec![5.0; i.pow(2)], i, i));
        test_matrices_simd_2.push(FloatMatrix::new(vec![15.0; i.pow(2)], i, i));
    }

    let mut simd_times = vec![];
    let mut naive_times = vec![];

    let test_matrices_naive_1 = test_matrices_simd_1.clone();
    let test_matrices_naive_2 = test_matrices_simd_2.clone();
    
    
    for (m1, m2) in test_matrices_simd_1.into_iter().zip(test_matrices_simd_2.into_iter()) {
        let simd_start = Instant::now();
        let _m3 = m1 * m2;
        let simd_end = Instant::now();
        simd_times.push(simd_end-simd_start);
    }

    for (m1, m2) in test_matrices_naive_1.into_iter().zip(test_matrices_naive_2.into_iter()) {
        let naive_start = Instant::now();
        let _m3 = m1.naive_mult(m2);
        let naive_end = Instant::now();
        naive_times.push(naive_end-naive_start);
    }

    for (i, (s_t, n_t)) in simd_times.iter().zip(naive_times.iter()).enumerate() {
        println!("{}x{} \t simd {:?} naive {:?}", i, i, s_t, n_t);
    }
}
