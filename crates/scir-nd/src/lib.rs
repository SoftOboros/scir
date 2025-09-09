//! ndarray interoperability helpers for SciR

use ndarray::Array1;

/// Convert a Vec into an ndarray Array1.
pub fn vec_to_array1<T>(v: Vec<T>) -> Array1<T> {
    Array1::from(v)
}

/// Convert an Array1 into a Vec.
pub fn array1_to_vec<T: Clone>(a: &Array1<T>) -> Vec<T> {
    a.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use scir_core::assert_close;

    #[test]
    fn roundtrip_vec() {
        let v = vec![1.0, 2.0, 3.0];
        let arr = vec_to_array1(v.clone());
        let back = array1_to_vec(&arr);
        assert_close!(&v, &back, slice, tol = 0.0);
    }

    #[test]
    fn roundtrip_array() {
        let arr = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let back = vec_to_array1(array1_to_vec(&arr));
        assert_close!(&arr, &back, array, tol = 0.0);
    }
}
