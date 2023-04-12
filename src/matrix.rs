use std::ops::Mul;

use std::simd;
use std::simd::SimdFloat;

#[derive(Debug, Clone, PartialEq)]
pub struct FloatMatrix {
    data: Vec<f64>,
    dim: (usize, usize)
}

impl FloatMatrix {
    pub fn new(data: Vec<f64>, width: usize, height: usize) -> Self {
        assert_eq!(data.len(), width*height);
        Self { data, dim: (width, height) }
    }

    pub fn as_raw(self) -> Vec<f64> {
        self.data
    }

    pub fn get_transpose(&self) -> Self {
        let mut transposed = Vec::with_capacity(self.data.len());
        (0..self.data.len())
            .map(|n| (n / self.dim.1) + ((n%self.dim.1)*self.dim.0))
            .for_each(|n| transposed.push(self.data.get(n).unwrap().clone()));

        Self { data: transposed, dim: (self.dim.1, self.dim.0) }
    }

    pub fn naive_mult(self, rhs: Self) -> Self {
        let mut new_data = Vec::with_capacity(rhs.dim.0*self.dim.1);

        for row_i in 0..self.dim.1 {
            let row = self.get_row(row_i);
            for col_i in 0..rhs.dim.0 {
                for (i, val1) in row.iter().enumerate() {
                    let val2 = rhs.get_row(i)[col_i];
                    new_data.push(val1*val2)
                }
            }
        }

        Self { data: new_data, dim: (rhs.dim.0, self.dim.1) }
    }

    #[inline]
    fn get_row(&self, n: usize) -> &[f64] {
        &self.data[n*self.dim.0..(n+1)*self.dim.0]
    }
}


macro_rules! lane_size_mult {
    ($rem: ident, $sum: ident, $vector1: ident, $vector2: ident, $lane_size: literal) => {
        if $rem >= $lane_size {
            let current_len = $vector1.len()-$rem;
            let next_simd_vector_one: simd::Simd<f64, $lane_size> = simd::Simd::from_slice(&$vector1[current_len..current_len+$lane_size]);
            let next_simd_vector_two: simd::Simd<f64, $lane_size> = simd::Simd::from_slice(&$vector2[current_len..current_len+$lane_size]);
            let result = next_simd_vector_one * next_simd_vector_two;
            $sum += result.reduce_sum();
            $rem -= $lane_size;
        }
    };
}


fn execute_mult_sum_simd(vector1: &[f64], vector2: &[f64]) -> f64 {
    assert_eq!(vector1.len(), vector2.len());

    let mut remaining_length = vector1.len();
    let mut sum = 0.0;

    while remaining_length > 0 {
        lane_size_mult!(remaining_length, sum, vector1, vector2, 64);
        lane_size_mult!(remaining_length, sum, vector1, vector2, 32);
        lane_size_mult!(remaining_length, sum, vector1, vector2, 16);
        lane_size_mult!(remaining_length, sum, vector1, vector2, 8);
        lane_size_mult!(remaining_length, sum, vector1, vector2, 4);
        lane_size_mult!(remaining_length, sum, vector1, vector2, 2);

        // simd for one value is slower than normal multiplication
        if remaining_length == 1 {
            sum += vector1.last().unwrap()*vector2.last().unwrap();
            remaining_length = 0;
        }
    }

    sum
}

impl Mul for FloatMatrix {
    type Output = FloatMatrix;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.dim.0 == 0 || self.dim.1 == 0 || rhs.dim.0 == 0 || rhs.dim.1 == 0 {
            return FloatMatrix::new(vec![], 0, 0);
        }

        if !is_x86_feature_detected!("avx2") {
            return self.naive_mult(rhs)
        }

        assert_eq!(self.dim.0, rhs.dim.1);

        let mut new_matrix_data = Vec::with_capacity(rhs.dim.0*self.dim.1);

        let mut simd_cols = Vec::with_capacity(rhs.dim.0);
        for col in 0..rhs.dim.0 {
            let mut col_vals = Vec::with_capacity(rhs.dim.1);
            for col_val in 0..rhs.dim.1 {
                col_vals.push(rhs.get_row(col_val)[col]);
            }
            simd_cols.push(col_vals);
        }

        for row in 0..self.dim.1 {
            let row_simd = self.get_row(row);
            for col in &simd_cols {
                let result = execute_mult_sum_simd(row_simd, col.as_slice());
                new_matrix_data.push(result);
            }
        }

        Self { data: new_matrix_data, dim: (rhs.dim.0, self.dim.1) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simd_mult_sum_small() {
        let result = execute_mult_sum_simd(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
        assert_eq!(result, 700.0);
    }

    #[test]
    fn simd_mult_sum_big() {
        // 9900, 9901, 9902 etc.
        let vec1: Vec<f64> = (9900..10000).map(|x| x as f64).collect();
        // 99, 199, 299 etc.
        let vec2: Vec<f64> = (0..100).map(|x| ((x*100)+99) as f64).collect();

        let result = execute_mult_sum_simd(vec1.as_slice(), vec2.as_slice());
        assert_eq!(result, 5031835050.0);
    }

    #[test]
    fn square_matrix() {
        let m1 = FloatMatrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3);
        let m2 = m1.clone();

        let m3 = m1 * m2;

        assert_eq!(m3.as_raw(), vec![30.0, 36.0, 42.0, 66.0, 81.0, 96.0, 102.0, 126.0, 150.0])
    }

    #[test]
    fn different_sized_small_matrix() {
        let m1 = FloatMatrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 4, 2);
        let m2 = FloatMatrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 4);

        let m3 = m1 * m2;

        assert_eq!(m3.as_raw(), vec![70.0, 80.0, 90.0, 158.0, 184.0, 210.0]);
    }

    #[test]
    fn medium_square_matrix() {
        let mut m1_data = vec![];
        for i in 0..100 {
            m1_data.push(i as f64);
        }

        let m1 = FloatMatrix::new(m1_data, 10, 10);
        let m2 = m1.clone();

        let m3 = m1 * m2;

        assert_eq!(m3.as_raw(), vec![2850.0, 2895.0, 2940.0, 2985.0, 3030.0, 3075.0, 3120.0, 3165.0, 3210.0, 3255.0, 7350.0, 7495.0, 7640.0, 7785.0, 7930.0, 
                                    8075.0, 8220.0, 8365.0, 8510.0, 8655.0, 11850.0, 12095.0, 12340.0, 12585.0, 12830.0, 13075.0, 13320.0, 13565.0, 13810.0, 14055.0, 
                                    16350.0, 16695.0, 17040.0, 17385.0, 17730.0, 18075.0, 18420.0, 18765.0, 19110.0, 19455.0, 20850.0, 21295.0, 21740.0, 22185.0, 
                                    22630.0, 23075.0, 23520.0, 23965.0, 24410.0, 24855.0, 25350.0, 25895.0, 26440.0, 26985.0, 27530.0, 28075.0, 28620.0, 29165.0, 
                                    29710.0, 30255.0, 29850.0, 30495.0, 31140.0, 31785.0, 32430.0, 33075.0, 33720.0, 34365.0, 35010.0, 35655.0, 34350.0, 35095.0, 
                                    35840.0, 36585.0, 37330.0, 38075.0, 38820.0, 39565.0, 40310.0, 41055.0, 38850.0, 39695.0, 40540.0, 41385.0, 42230.0, 43075.0, 
                                    43920.0, 44765.0, 45610.0, 46455.0, 43350.0, 44295.0, 45240.0, 46185.0, 47130.0, 48075.0, 49020.0, 49965.0, 50910.0, 51855.0])
    }
}
