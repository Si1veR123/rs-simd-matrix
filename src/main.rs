
use mat_mult::matrix::FloatMatrix;
use std::time::Instant;

fn main() {
    let test = FloatMatrix::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0], 4, 3);
    let tr = test.get_transpose();
    println!("{:?}", tr);

    let mut test_matrices_1 = vec![];
    let mut test_matrices_2 = vec![];

    for i in 0..1000usize {
        test_matrices_1.push(FloatMatrix::new(vec![5.0; i.pow(2)], i, i));
        test_matrices_2.push(FloatMatrix::new(vec![15.0; i.pow(2)], i, i));
    }
    
    for (i, (m1, m2)) in test_matrices_1.into_iter().zip(test_matrices_2.into_iter()).enumerate() {
        let m1_1 = m1.clone();
        let m2_1 = m2.clone();
        let simd_start = Instant::now();
        let _m3 = m1_1 * m2_1;
        let simd_end = Instant::now();
        
        let naive_start = Instant::now();
        let _m3 = m1.naive_mult(m2);
        let naive_end = Instant::now();

        println!("{}x{} \t simd {:?} naive {:?}", i, i, simd_end-simd_start, naive_end-naive_start);
    }
}
