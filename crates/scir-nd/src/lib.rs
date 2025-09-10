//! ndarray interoperability helpers for SciR.
#![deny(missing_docs)]

use ndarray::Array1;

/// Convert a `Vec<T>` into an ndarray `Array1<T>`.
///
/// # Examples
/// ```
/// use ndarray::Array1;
/// let v = vec![1,2,3];
/// let a: Array1<i32> = scir_nd::vec_to_array1(v);
/// assert_eq!(a.len(), 3);
/// ```
pub fn vec_to_array1<T>(v: Vec<T>) -> Array1<T> {
    Array1::from(v)
}

/// Convert an `Array1<T>` into a `Vec<T>`.
///
/// # Examples
/// ```
/// use ndarray::Array1;
/// let a = Array1::from_vec(vec![1,2,3]);
/// let v = scir_nd::array1_to_vec(&a);
/// assert_eq!(v, vec![1,2,3]);
/// ```
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
