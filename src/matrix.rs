use std::ops::Mul;

use std::simd;

#[derive(Debug, Clone)]
pub struct FloatMatrix {
    data: Vec<f64>,
    dim: (usize, usize)
}

impl FloatMatrix {
    pub fn new(data: Vec<f64>, width: usize, height: usize) -> Self {
        Self { data, dim: (width, height) }
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
    ($rem: ident, $calculate_buffer: ident, $vector1: ident, $vector2: ident, $lane_size: literal) => {
        if $rem >= $lane_size {
            let current_len = $calculate_buffer.len();
            let next_simd_vector_one: simd::Simd<f64, $lane_size> = simd::Simd::from_slice(&$vector1[current_len..current_len+$lane_size]);
            let next_simd_vector_two: simd::Simd<f64, $lane_size> = simd::Simd::from_slice(&$vector2[current_len..current_len+$lane_size]);
            let result = next_simd_vector_one * next_simd_vector_two;
            $calculate_buffer.extend(result.as_array());
            $rem -= $lane_size;
        }
    };
}


fn execute_mult_simd(vector1: &[f64], vector2: &[f64]) -> Vec<f64> {
    assert_eq!(vector1.len(), vector2.len());

    let mut remaining_length = vector1.len();
    let mut calculated = Vec::with_capacity(vector1.len());

    while remaining_length > 0 {
        lane_size_mult!(remaining_length, calculated, vector1, vector2, 64);
        lane_size_mult!(remaining_length, calculated, vector1, vector2, 32);
        lane_size_mult!(remaining_length, calculated, vector1, vector2, 16);
        lane_size_mult!(remaining_length, calculated, vector1, vector2, 8);
        lane_size_mult!(remaining_length, calculated, vector1, vector2, 4);
        lane_size_mult!(remaining_length, calculated, vector1, vector2, 2);

        // simd for one value is slower than normal multiplication
        if remaining_length == 1 {
            calculated.push(vector1.last().unwrap()*vector2.last().unwrap());
            remaining_length = 0;
        }
    }

    calculated
}

impl Mul for FloatMatrix {
    type Output = FloatMatrix;

    fn mul(self, rhs: Self) -> Self::Output {
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
                let result = execute_mult_simd(row_simd, col.as_slice()).iter().cloned().sum();
                new_matrix_data.push(result);
            }
        }

        Self { data: new_matrix_data, dim: (rhs.dim.0, self.dim.1) }
    }
}
