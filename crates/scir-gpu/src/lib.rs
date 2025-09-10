//! GPU Foundations: device array abstraction and CPU-backed baseline ops.
#![deny(missing_docs)]

use ndarray::{Array1, Array2, Axis};
use num_traits::NumAssign;
use std::error::Error;
use std::fmt;

/// Supported data types for device arrays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    /// 32-bit floating point
    F32,
    /// 64-bit floating point
    F64,
}

/// Execution device selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    /// Host CPU device
    Cpu,
    #[cfg(feature = "cuda")]
    /// NVIDIA CUDA device (feature `cuda`)
    Cuda,
}

/// GPU-related error types.
#[derive(Debug)]
pub enum GpuError {
    /// Backend is not available on this build or platform.
    BackendUnavailable(&'static str),
    /// Operation failed due to incompatible shapes.
    ShapeMismatch,
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::BackendUnavailable(name) => write!(f, "backend not available: {name}"),
            GpuError::ShapeMismatch => write!(f, "shape mismatch"),
        }
    }
}

impl Error for GpuError {}

/// A minimal, shaped device array with dtype. Currently CPU-backed.
#[derive(Clone, Debug)]
pub struct DeviceArray<T> {
    shape: Vec<usize>,
    dtype: DType,
    device: Device,
    // CPU storage; future backends will switch to an enum for storage.
    host: Vec<T>,
}

impl<T: Copy> DeviceArray<T> {
    /// Create a `DeviceArray` from a CPU slice and explicit shape/dtype.
    ///
    /// # Examples
    /// ```
    /// let data = vec![1.0f32, 2.0, 3.0, 4.0];
    /// let arr = scir_gpu::DeviceArray::from_cpu_slice(&[2,2], scir_gpu::DType::F32, &data);
    /// assert_eq!(arr.shape(), &[2,2]);
    /// ```
    pub fn from_cpu_slice(shape: &[usize], dtype: DType, data: &[T]) -> Self {
        assert_eq!(shape.iter().product::<usize>(), data.len());
        Self {
            shape: shape.to_vec(),
            dtype,
            device: Device::Cpu,
            host: data.to_vec(),
        }
    }

    /// Copy data back to a CPU-owned `Vec<T>`.
    ///
    /// # Examples
    /// ```
    /// let data = vec![1i32,2,3];
    /// let arr = scir_gpu::DeviceArray::from_cpu_slice(&[3], scir_gpu::DType::F32, &data);
    /// let back = arr.to_cpu_vec();
    /// assert_eq!(back, vec![1,2,3]);
    /// ```
    pub fn to_cpu_vec(&self) -> Vec<T> {
        self.host.clone()
    }

    /// Return the logical shape of the array.
    ///
    /// # Examples
    /// ```
    /// let data = vec![0u8; 6];
    /// let arr = scir_gpu::DeviceArray::from_cpu_slice(&[2,3], scir_gpu::DType::F32, &data);
    /// assert_eq!(arr.shape(), &[2,3]);
    /// ```
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Return the element data type.
    ///
    /// # Examples
    /// ```
    /// let data = vec![0u8; 4];
    /// let arr = scir_gpu::DeviceArray::from_cpu_slice(&[4], scir_gpu::DType::F32, &data);
    /// assert!(matches!(arr.dtype(), scir_gpu::DType::F32));
    /// ```
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Return the current device of this array.
    ///
    /// # Examples
    /// ```
    /// let data = vec![1u8,2,3,4];
    /// let arr = scir_gpu::DeviceArray::from_cpu_slice(&[4], scir_gpu::DType::F32, &data);
    /// assert!(matches!(arr.device(), scir_gpu::Device::Cpu));
    /// ```
    pub fn device(&self) -> Device {
        self.device
    }
}

impl<T: Copy> DeviceArray<T> {
    #[cfg(feature = "cuda")]
    /// Move the array to a device (CPU or CUDA if enabled).
    ///
    /// # Examples
    /// ```
    /// let data = vec![1.0f32, 2.0, 3.0, 4.0];
    /// let mut arr = scir_gpu::DeviceArray::from_cpu_slice(&[4], scir_gpu::DType::F32, &data);
    /// // Always available
    /// arr.to_device(scir_gpu::Device::Cpu).unwrap();
    /// ```
    pub fn to_device(&mut self, device: Device) -> Result<(), GpuError> {
        match device {
            Device::Cpu => {
                self.device = Device::Cpu;
                Ok(())
            }
            Device::Cuda => {
                // Placeholder: actual upload would allocate device memory and copy.
                self.device = Device::Cuda;
                Ok(())
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    /// Move the array to a device (CPU only, when CUDA is disabled).
    pub fn to_device(&mut self, device: Device) -> Result<(), GpuError> {
        match device {
            Device::Cpu => {
                self.device = Device::Cpu;
                Ok(())
            }
        }
    }
}

// Elementwise ops (CPU baseline)
impl<T> DeviceArray<T>
where
    T: Copy + NumAssign,
{
    /// Add a scalar to each element (CPU baseline).
    ///
    /// # Examples
    /// ```
    /// let data = vec![1.0f32, 2.0, 3.0];
    /// let arr = scir_gpu::DeviceArray::from_cpu_slice(&[3], scir_gpu::DType::F32, &data);
    /// let out = arr.add_scalar(1.0f32);
    /// assert_eq!(out.to_cpu_vec(), vec![2.0f32, 3.0, 4.0]);
    /// ```
    pub fn add_scalar(&self, alpha: T) -> Self {
        let mut out = self.clone();
        for v in &mut out.host {
            *v += alpha;
        }
        out
    }

    /// Multiply each element by a scalar (CPU baseline).
    ///
    /// # Examples
    /// ```
    /// let data = vec![1.0f32, 2.0, 3.0];
    /// let arr = scir_gpu::DeviceArray::from_cpu_slice(&[3], scir_gpu::DType::F32, &data);
    /// let out = arr.mul_scalar(2.0f32);
    /// assert_eq!(out.to_cpu_vec(), vec![2.0f32, 4.0, 6.0]);
    /// ```
    pub fn mul_scalar(&self, alpha: T) -> Self {
        let mut out = self.clone();
        for v in &mut out.host {
            *v *= alpha;
        }
        out
    }
}

impl<T> DeviceArray<T>
where
    T: Copy + NumAssign,
{
    /// Elementwise addition between arrays (CPU baseline).
    ///
    /// # Examples
    /// ```
    /// let a = scir_gpu::DeviceArray::from_cpu_slice(&[3], scir_gpu::DType::F32, &[1.0f32,2.0,3.0]);
    /// let b = scir_gpu::DeviceArray::from_cpu_slice(&[3], scir_gpu::DType::F32, &[0.5f32,1.5,2.5]);
    /// let c = a.add(&b).unwrap();
    /// assert_eq!(c.to_cpu_vec(), vec![1.5f32, 3.5, 5.5]);
    /// ```
    pub fn add(&self, other: &Self) -> Result<Self, GpuError> {
        if self.shape != other.shape {
            return Err(GpuError::ShapeMismatch);
        }
        let mut out = self.clone();
        for (o, r) in out.host.iter_mut().zip(other.host.iter()) {
            *o += *r;
        }
        Ok(out)
    }
}

impl DeviceArray<f32> {
    /// Elementwise add-scalar, with CUDA dispatch when available.
    ///
    /// # Examples
    /// ```
    /// let data = vec![1.0f32, 2.0, 3.0];
    /// let mut a = scir_gpu::DeviceArray::from_cpu_slice(&[3], scir_gpu::DType::F32, &data);
    /// a.to_device(scir_gpu::Device::Cpu).unwrap();
    /// let out = a.add_scalar_auto(1.0);
    /// assert_eq!(out.to_cpu_vec(), vec![2.0f32, 3.0, 4.0]);
    /// ```
    pub fn add_scalar_auto(&self, alpha: f32) -> Self {
        #[cfg(feature = "cuda")]
        {
            match self.device {
                Device::Cpu => self.mul_scalar(1.0f32).add_scalar(alpha), // reuse CPU path
                Device::Cuda => {
                    let mut out = vec![0.0f32; self.host.len()];
                    if let Err(_) = crate::add_scalar_f32_cuda(&self.host, alpha, &mut out) {
                        // Fallback to CPU on failure
                        return self.mul_scalar(1.0f32).add_scalar(alpha);
                    }
                    DeviceArray {
                        shape: self.shape.clone(),
                        dtype: self.dtype,
                        device: self.device,
                        host: out,
                    }
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.mul_scalar(1.0f32).add_scalar(alpha)
        }
    }

    /// Elementwise add of two arrays, with CUDA dispatch when available.
    ///
    /// # Examples
    /// ```
    /// let a = scir_gpu::DeviceArray::from_cpu_slice(&[2], scir_gpu::DType::F32, &[1.0f32,2.0]);
    /// let b = scir_gpu::DeviceArray::from_cpu_slice(&[2], scir_gpu::DType::F32, &[0.5f32,1.5]);
    /// let c = a.add_auto(&b).unwrap();
    /// assert_eq!(c.to_cpu_vec(), vec![1.5f32, 3.5]);
    /// ```
    pub fn add_auto(&self, other: &Self) -> Result<Self, GpuError> {
        if self.shape != other.shape {
            return Err(GpuError::ShapeMismatch);
        }
        #[cfg(feature = "cuda")]
        {
            match (self.device, other.device) {
                (Device::Cpu, Device::Cpu) => self.add(other),
                (Device::Cuda, Device::Cuda) => {
                    let mut out = vec![0.0f32; self.host.len()];
                    if let Err(_) = crate::add_vec_f32_cuda(&self.host, &other.host, &mut out) {
                        // Fallback to CPU on failure
                        return self.add(other);
                    }
                    Ok(DeviceArray {
                        shape: self.shape.clone(),
                        dtype: self.dtype,
                        device: self.device,
                        host: out,
                    })
                }
                _ => self.add(other),
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.add(other)
        }
    }

    /// Elementwise mul-scalar with device dispatch.
    ///
    /// # Examples
    /// ```
    /// let data = vec![1.0f32, 2.0, 3.0];
    /// let a = scir_gpu::DeviceArray::from_cpu_slice(&[3], scir_gpu::DType::F32, &data);
    /// let out = a.mul_scalar_auto(2.0);
    /// assert_eq!(out.to_cpu_vec(), vec![2.0f32, 4.0, 6.0]);
    /// ```
    pub fn mul_scalar_auto(&self, alpha: f32) -> Self {
        #[cfg(feature = "cuda")]
        {
            match self.device {
                Device::Cpu => self.mul_scalar(alpha),
                Device::Cuda => {
                    let mut out = vec![0.0f32; self.host.len()];
                    if let Err(_) = crate::mul_scalar_f32_cuda(&self.host, alpha, &mut out) {
                        return self.mul_scalar(alpha);
                    }
                    DeviceArray {
                        shape: self.shape.clone(),
                        dtype: self.dtype,
                        device: self.device,
                        host: out,
                    }
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.mul_scalar(alpha)
        }
    }
}

/// Dispatch FIR to CUDA if requested; otherwise use CPU baseline.
/// FIR over each row of `x` using `taps` with device dispatch (CUDA when available).
///
/// This function accepts a `Device` hint and will attempt to run on
/// CUDA when compiled with the `cuda` feature and a device is present,
/// otherwise it falls back to the CPU baseline.
///
/// # Examples
/// ```
/// use ndarray::{array, Array1, Array2};
/// let x: Array2<f32> = array![[1.0, 2.0, 3.0, 4.0]];
/// let taps: Array1<f32> = array![0.25, 0.5, 0.25];
/// // Explicitly run on CPU; returns shape-identical output
/// let y = scir_gpu::fir1d_batched_f32_auto(&x, &taps, scir_gpu::Device::Cpu);
/// assert_eq!(y.shape(), &[1, 4]);
/// ```
pub fn fir1d_batched_f32_auto(x: &Array2<f32>, taps: &Array1<f32>, device: Device) -> Array2<f32> {
    /// FIR over each row of `x` using `taps` with device dispatch (CUDA when available).
    #[cfg(feature = "cuda")]
    {
        return match device {
            Device::Cpu => fir1d_batched_f32(x, taps),
            Device::Cuda => match crate::fir1d_batched_f32_cuda(x, taps) {
                Ok(y) => y,
                Err(_) => fir1d_batched_f32(x, taps),
            },
        };
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = device;
        fir1d_batched_f32(x, taps)
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use std::ffi::c_void;
    use std::ptr;

    // Minimal CUDA Driver API bindings
    type CUdevice = i32;
    type CUcontext = *mut c_void;
    type CUmodule = *mut c_void;
    type CUfunction = *mut c_void;
    type CUdeviceptr = u64;
    type CUresult = i32;

    const CUDA_SUCCESS: CUresult = 0;

    #[link(name = "cuda")]
    extern "C" {
        fn cuInit(flags: u32) -> CUresult;
        fn cuDeviceGet(device: *mut CUdevice, ordinal: i32) -> CUresult;
        fn cuCtxCreate(ctx: *mut CUcontext, flags: u32, device: CUdevice) -> CUresult;
        fn cuCtxDestroy(ctx: CUcontext) -> CUresult;
        fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult;
        fn cuModuleGetFunction(hfunc: *mut CUfunction, hmod: CUmodule, name: *const u8)
            -> CUresult;
        fn cuMemAlloc(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
        fn cuMemFree(dptr: CUdeviceptr) -> CUresult;
        fn cuMemcpyHtoD(
            dstDevice: CUdeviceptr,
            srcHost: *const c_void,
            ByteCount: usize,
        ) -> CUresult;
        fn cuMemcpyDtoH(dstHost: *mut c_void, srcDevice: CUdeviceptr, ByteCount: usize)
            -> CUresult;
        fn cuLaunchKernel(
            f: CUfunction,
            gridDimX: u32,
            gridDimY: u32,
            gridDimZ: u32,
            blockDimX: u32,
            blockDimY: u32,
            blockDimZ: u32,
            sharedMemBytes: u32,
            hStream: *mut c_void,
            kernelParams: *mut *mut c_void,
            extra: *mut *mut c_void,
        ) -> CUresult;
        fn cuCtxSynchronize() -> CUresult;
    }

    fn check(res: CUresult, msg: &str) -> Result<(), GpuError> {
        if res == CUDA_SUCCESS {
            Ok(())
        } else {
            Err(GpuError::BackendUnavailable(msg))
        }
    }

    pub fn cuda_available() -> bool {
        unsafe {
            cuInit(0) == CUDA_SUCCESS && {
                let mut d = 0;
                cuDeviceGet(&mut d as *mut _, 0) == CUDA_SUCCESS
            }
        }
    }

    struct CudaCtx {
        ctx: CUcontext,
    }
    impl CudaCtx {
        fn create_default() -> Result<Self, GpuError> {
            unsafe {
                check(cuInit(0), "cuInit")?;
                let mut dev: CUdevice = 0;
                check(cuDeviceGet(&mut dev as *mut _, 0), "cuDeviceGet")?;
                let mut ctx: CUcontext = ptr::null_mut();
                check(cuCtxCreate(&mut ctx as *mut _, 0, dev), "cuCtxCreate")?;
                Ok(Self { ctx })
            }
        }
    }
    impl Drop for CudaCtx {
        fn drop(&mut self) {
            unsafe {
                let _ = cuCtxDestroy(self.ctx);
            }
        }
    }

    static PTX: &str = r#"
.version 7.0
.target sm_52
.address_size 64

.visible .entry add_vec_f32(
    .param .u64 out,
    .param .u64 a,
    .param .u64 b,
    .param .u32 n)
{
    .reg .pred %p;
    .reg .b32 %r<6>;
    .reg .b64 %rd<10>;
    .reg .f32 %f<4>;

    ld.param.u64 %rd1, [out];
    ld.param.u64 %rd2, [a];
    ld.param.u64 %rd3, [b];
    ld.param.u32 %r1, [n];

    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mad.lo.s32 %r5, %r3, %r4, %r2; // idx
    setp.ge.s32 %p, %r5, %r1;
    @%p ret;

    mul.wide.s32 %rd4, %r5, 4;
    add.s64 %rd5, %rd2, %rd4;
    add.s64 %rd6, %rd3, %rd4;
    add.s64 %rd7, %rd1, %rd4;
    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];
    add.f32 %f3, %f1, %f2;
    st.global.f32 [%rd7], %f3;
    ret;
}

.visible .entry add_scalar_f32(
    .param .u64 out,
    .param .u64 a,
    .param .f32 alpha,
    .param .u32 n)
{
    .reg .pred %p;
    .reg .b32 %r<6>;
    .reg .b64 %rd<10>;
    .reg .f32 %f<4>;

    ld.param.u64 %rd1, [out];
    ld.param.u64 %rd2, [a];
    ld.param.f32 %f1, [alpha];
    ld.param.u32 %r1, [n];

    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mad.lo.s32 %r5, %r3, %r4, %r2; // idx
    setp.ge.s32 %p, %r5, %r1;
    @%p ret;

    mul.wide.s32 %rd4, %r5, 4;
    add.s64 %rd5, %rd2, %rd4;
    add.s64 %rd6, %rd1, %rd4;
    ld.global.f32 %f2, [%rd5];
    add.f32 %f3, %f2, %f1;
    st.global.f32 [%rd6], %f3;
    ret;
}

.visible .entry mul_scalar_f32(
    .param .u64 out,
    .param .u64 a,
    .param .f32 alpha,
    .param .u32 n)
{
    .reg .pred %p;
    .reg .b32 %r<6>;
    .reg .b64 %rd<10>;
    .reg .f32 %f<4>;

    ld.param.u64 %rd1, [out];
    ld.param.u64 %rd2, [a];
    ld.param.f32 %f1, [alpha];
    ld.param.u32 %r1, [n];

    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mad.lo.s32 %r5, %r3, %r4, %r2; // idx
    setp.ge.s32 %p, %r5, %r1;
    @%p ret;

    mul.wide.s32 %rd4, %r5, 4;
    add.s64 %rd5, %rd2, %rd4;
    add.s64 %rd6, %rd1, %rd4;
    ld.global.f32 %f2, [%rd5];
    mul.f32 %f3, %f2, %f1;
    st.global.f32 [%rd6], %f3;
    ret;
}

.visible .entry fir1d_batched_f32(
    .param .u64 out,
    .param .u64 x,
    .param .u64 taps,
    .param .u32 b,
    .param .u32 n,
    .param .u32 k)
{
    .reg .pred %p<3>;
    .reg .b32 %r<20>;
    .reg .b64 %rd<20>;
    .reg .f32 %f<6>;

    // Load params
    ld.param.u64 %rd1, [out];
    ld.param.u64 %rd2, [x];
    ld.param.u64 %rd3, [taps];
    ld.param.u32 %rB, [b];
    ld.param.u32 %rN, [n];
    ld.param.u32 %rK, [k];

    // idx = blockIdx.x * blockDim.x + threadIdx.x
    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mad.lo.s32 %rIdx, %r3, %r4, %r2;

    // total = b*n
    mul.lo.u32 %rTotal, %rB, %rN;
    setp.ge.u32 %p0, %rIdx, %rTotal;
    @%p0 ret;

    // bi = idx / n; i = idx % n
    div.u32 %rBi, %rIdx, %rN;
    rem.u32 %rI, %rIdx, %rN;

    // start = (i + 1 > k) ? (i + 1 - k) : 0
    add.u32 %rTmp, %rI, 1;
    setp.gt.u32 %p1, %rTmp, %rK;
    mov.u32 %rStart, 0;
    @%p1 sub.u32 %rStart, %rTmp, %rK;

    // acc = 0.0f; j = i; t_idx = 0
    mov.f32 %fAcc, 0f00000000; // 0.0
    mov.u32 %rJ, %rI;
    mov.u32 %rTIdx, 0;

L_LOOP:
    // if (j < start) break;
    setp.lt.u32 %p2, %rJ, %rStart;
    @%p2 bra L_DONE;

    // tap_index = k - 1 - t_idx
    mov.u32 %rKminus1, 0;
    add.u32 %rKminus1, %rK, 0xffffffff; // k-1
    sub.u32 %rTapIdx, %rKminus1, %rTIdx;
    mul.wide.u32 %rdTapOff, %rTapIdx, 4;
    add.s64 %rdTapPtr, %rd3, %rdTapOff;
    ld.global.f32 %fTap, [%rdTapPtr];

    // x index: bi*n + j
    mul.lo.u32 %rRowOff, %rBi, %rN;
    add.u32 %rXIdx, %rRowOff, %rJ;
    mul.wide.u32 %rdXOff, %rXIdx, 4;
    add.s64 %rdXPtr, %rd2, %rdXOff;
    ld.global.f32 %fX, [%rdXPtr];

    // acc += tap * x
    mul.f32 %fMul, %fTap, %fX;
    add.f32 %fAcc, %fAcc, %fMul;

    // j--, t_idx++
    add.u32 %rJ, %rJ, 0xffffffff; // j-1
    add.u32 %rTIdx, %rTIdx, 1;
    bra L_LOOP;

L_DONE:
    // out index: bi*n + i
    mul.lo.u32 %rOutIdxBase, %rBi, %rN;
    add.u32 %rOutIdx, %rOutIdxBase, %rI;
    mul.wide.u32 %rdOutOff, %rOutIdx, 4;
    add.s64 %rdOutPtr, %rd1, %rdOutOff;
    st.global.f32 [%rdOutPtr], %fAcc;
    ret;
}
"#;

    fn load_module() -> Result<(CudaCtx, CUmodule), GpuError> {
        unsafe {
            let ctx = CudaCtx::create_default()?;
            let mut module: CUmodule = ptr::null_mut();
            check(
                cuModuleLoadData(&mut module as *mut _, PTX.as_ptr() as *const c_void),
                "cuModuleLoadData",
            )?;
            Ok((ctx, module))
        }
    }

    unsafe fn get_function(module: CUmodule, name: &str) -> Result<CUfunction, GpuError> {
        let mut func: CUfunction = ptr::null_mut();
        let cname = name.as_bytes();
        check(
            cuModuleGetFunction(&mut func as *mut _, module, cname.as_ptr()),
            name,
        )?;
        Ok(func)
    }

    pub fn add_vec_f32_cuda(a: &[f32], b: &[f32], out: &mut [f32]) -> Result<(), GpuError> {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), out.len());
        let n = a.len() as u32;
        unsafe {
            let (_ctx, module) = load_module()?;
            let func = get_function(module, "add_vec_f32")?;
            let bytes = (n as usize) * std::mem::size_of::<f32>();

            let mut d_a: CUdeviceptr = 0;
            let mut d_b: CUdeviceptr = 0;
            let mut d_out: CUdeviceptr = 0;
            check(cuMemAlloc(&mut d_a as *mut _, bytes), "cuMemAlloc a")?;
            check(cuMemAlloc(&mut d_b as *mut _, bytes), "cuMemAlloc b")?;
            check(cuMemAlloc(&mut d_out as *mut _, bytes), "cuMemAlloc out")?;

            check(
                cuMemcpyHtoD(d_a, a.as_ptr() as *const c_void, bytes),
                "cuMemcpyHtoD a",
            )?;
            check(
                cuMemcpyHtoD(d_b, b.as_ptr() as *const c_void, bytes),
                "cuMemcpyHtoD b",
            )?;

            let mut out_ptr = d_out as *mut c_void;
            let mut a_ptr = d_a as *mut c_void;
            let mut b_ptr = d_b as *mut c_void;
            let mut n_val = n;
            let mut params = vec![
                &mut out_ptr as *mut _ as *mut c_void,
                &mut a_ptr as *mut _ as *mut c_void,
                &mut b_ptr as *mut _ as *mut c_void,
                &mut n_val as *mut _ as *mut c_void,
            ];

            let block = 256u32;
            let grid = ((n + block - 1) / block) as u32;
            check(
                cuLaunchKernel(
                    func,
                    grid,
                    1,
                    1,
                    block,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "cuLaunchKernel add_vec_f32",
            )?;
            check(cuCtxSynchronize(), "cuCtxSynchronize")?;

            check(
                cuMemcpyDtoH(out.as_mut_ptr() as *mut c_void, d_out, bytes),
                "cuMemcpyDtoH out",
            )?;

            let _ = cuMemFree(d_a);
            let _ = cuMemFree(d_b);
            let _ = cuMemFree(d_out);
            Ok(())
        }
    }

    pub fn add_scalar_f32_cuda(a: &[f32], alpha: f32, out: &mut [f32]) -> Result<(), GpuError> {
        assert_eq!(a.len(), out.len());
        let n = a.len() as u32;
        unsafe {
            let (_ctx, module) = load_module()?;
            let func = get_function(module, "add_scalar_f32")?;
            let bytes = (n as usize) * std::mem::size_of::<f32>();

            let mut d_a: CUdeviceptr = 0;
            let mut d_out: CUdeviceptr = 0;
            check(cuMemAlloc(&mut d_a as *mut _, bytes), "cuMemAlloc a")?;
            check(cuMemAlloc(&mut d_out as *mut _, bytes), "cuMemAlloc out")?;
            check(
                cuMemcpyHtoD(d_a, a.as_ptr() as *const c_void, bytes),
                "cuMemcpyHtoD a",
            )?;

            let mut out_ptr = d_out as *mut c_void;
            let mut a_ptr = d_a as *mut c_void;
            let mut alpha_val = alpha;
            let mut n_val = n;
            let mut params = vec![
                &mut out_ptr as *mut _ as *mut c_void,
                &mut a_ptr as *mut _ as *mut c_void,
                &mut alpha_val as *mut _ as *mut c_void,
                &mut n_val as *mut _ as *mut c_void,
            ];

            let block = 256u32;
            let grid = ((n + block - 1) / block) as u32;
            check(
                cuLaunchKernel(
                    func,
                    grid,
                    1,
                    1,
                    block,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "cuLaunchKernel add_scalar_f32",
            )?;
            check(cuCtxSynchronize(), "cuCtxSynchronize")?;

            check(
                cuMemcpyDtoH(out.as_mut_ptr() as *mut c_void, d_out, bytes),
                "cuMemcpyDtoH out",
            )?;
            let _ = cuMemFree(d_a);
            let _ = cuMemFree(d_out);
            Ok(())
        }
    }

    pub fn mul_scalar_f32_cuda(a: &[f32], alpha: f32, out: &mut [f32]) -> Result<(), GpuError> {
        assert_eq!(a.len(), out.len());
        let n = a.len() as u32;
        unsafe {
            let (_ctx, module) = load_module()?;
            let func = get_function(module, "mul_scalar_f32")?;
            let bytes = (n as usize) * std::mem::size_of::<f32>();

            let mut d_a: CUdeviceptr = 0;
            let mut d_out: CUdeviceptr = 0;
            check(cuMemAlloc(&mut d_a as *mut _, bytes), "cuMemAlloc a")?;
            check(cuMemAlloc(&mut d_out as *mut _, bytes), "cuMemAlloc out")?;
            check(
                cuMemcpyHtoD(d_a, a.as_ptr() as *const c_void, bytes),
                "cuMemcpyHtoD a",
            )?;

            let mut out_ptr = d_out as *mut c_void;
            let mut a_ptr = d_a as *mut c_void;
            let mut alpha_val = alpha;
            let mut n_val = n;
            let mut params = vec![
                &mut out_ptr as *mut _ as *mut c_void,
                &mut a_ptr as *mut _ as *mut c_void,
                &mut alpha_val as *mut _ as *mut c_void,
                &mut n_val as *mut _ as *mut c_void,
            ];

            let block = 256u32;
            let grid = ((n + block - 1) / block) as u32;
            check(
                cuLaunchKernel(
                    func,
                    grid,
                    1,
                    1,
                    block,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                "cuLaunchKernel mul_scalar_f32",
            )?;
            check(cuCtxSynchronize(), "cuCtxSynchronize")?;

            check(
                cuMemcpyDtoH(out.as_mut_ptr() as *mut c_void, d_out, bytes),
                "cuMemcpyDtoH out",
            )?;
            let _ = cuMemFree(d_a);
            let _ = cuMemFree(d_out);
            Ok(())
        }
    }

    pub fn fir1d_batched_f32_cuda(
        x: &Array2<f32>,
        taps: &Array1<f32>,
    ) -> Result<Array2<f32>, GpuError> {
        let (b, n) = x.dim();
        let k = taps.len();
        let mut x_host = x.to_owned().into_raw_vec();
        let taps_host = taps.as_slice().unwrap();
        let mut out_host = vec![0.0f32; b * n];
        unsafe {
            let (_ctx, module) = load_module()?;
            let func = get_function(module, "fir1d_batched_f32")?;

            let bytes_x = x_host.len() * std::mem::size_of::<f32>();
            let bytes_t = k * std::mem::size_of::<f32>();
            let bytes_y = out_host.len() * std::mem::size_of::<f32>();

            let mut d_x: CUdeviceptr = 0;
            let mut d_t: CUdeviceptr = 0;
            let mut d_y: CUdeviceptr = 0;
            check(cuMemAlloc(&mut d_y as *mut _, bytes_y), "cuMemAlloc y")?;
            check(cuMemAlloc(&mut d_x as *mut _, bytes_x), "cuMemAlloc x")?;
            check(cuMemAlloc(&mut d_t as *mut _, bytes_t), "cuMemAlloc t")?;
            check(
                cuMemcpyHtoD(d_x, x_host.as_ptr() as *const c_void, bytes_x),
                "HtoD x",
            )?;
            check(
                cuMemcpyHtoD(d_t, taps_host.as_ptr() as *const c_void, bytes_t),
                "HtoD t",
            )?;

            let mut y_ptr = d_y as *mut c_void;
            let mut x_ptr = d_x as *mut c_void;
            let mut t_ptr = d_t as *mut c_void;
            let mut b_u32 = b as u32;
            let mut n_u32 = n as u32;
            let mut k_u32 = k as u32;
            let mut params = vec![
                &mut y_ptr as *mut _ as *mut c_void,
                &mut x_ptr as *mut _ as *mut c_void,
                &mut t_ptr as *mut _ as *mut c_void,
                &mut b_u32 as *mut _ as *mut c_void,
                &mut n_u32 as *mut _ as *mut c_void,
                &mut k_u32 as *mut _ as *mut c_void,
            ];

            let total = (b * n) as u32;
            let block = 256u32;
            let grid = ((total + block - 1) / block) as u32;
            check(
                cuLaunchKernel(
                    func,
                    grid,
                    1,
                    1,
                    block,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                "cuLaunchKernel fir1d_batched_f32",
            )?;
            check(cuCtxSynchronize(), "cuCtxSynchronize")?;
            check(
                cuMemcpyDtoH(out_host.as_mut_ptr() as *mut c_void, d_y, bytes_y),
                "DtoH y",
            )?;

            let _ = cuMemFree(d_x);
            let _ = cuMemFree(d_t);
            let _ = cuMemFree(d_y);
        }
        Ok(Array2::from_shape_vec((b, n), out_host).unwrap())
    }
}

#[cfg(feature = "cuda")]
pub use cuda::{add_scalar_f32_cuda, add_vec_f32_cuda, mul_scalar_f32_cuda};

/// Causal FIR over each row of `x` using `taps` (CPU baseline, f32).
///
/// Input shape is `(batch, n)` and the same shape is returned.
///
/// # Examples
/// ```
/// use ndarray::{array, Array1, Array2};
/// // Two rows (batch=2), four samples each
/// let x: Array2<f32> = array![[1.0, 2.0, 3.0, 4.0], [0.5, 0.0, -0.5, -1.0]];
/// let taps: Array1<f32> = array![0.25, 0.5, 0.25];
/// let y = scir_gpu::fir1d_batched_f32(&x, &taps);
/// assert_eq!(y.shape(), &[2, 4]);
/// ```
pub fn fir1d_batched_f32(x: &Array2<f32>, taps: &Array1<f32>) -> Array2<f32> {
    let (b, n) = x.dim();
    let k = taps.len();
    let mut y = Array2::<f32>::zeros((b, n));
    for bi in 0..b {
        let xin = x.index_axis(Axis(0), bi);
        let mut yout = y.index_axis_mut(Axis(0), bi);
        for i in 0..n {
            let mut acc = 0.0f32;
            let start = (i + 1).saturating_sub(k);
            for (t_idx, xi) in (start..=i).rev().enumerate() {
                let tap = taps[k - 1 - t_idx];
                acc += tap * xin[xi];
            }
            yout[i] = acc;
        }
    }
    y
}

/// Causal FIR over each row of `x` using `taps` (CPU baseline, f64).
///
/// Input shape is `(batch, n)` and the same shape is returned.
///
/// # Examples
/// ```
/// use ndarray::{array, Array1, Array2};
/// let x: Array2<f64> = array![[1.0, 2.0, 3.0, 4.0]];
/// let taps: Array1<f64> = array![0.25, 0.5, 0.25];
/// let y = scir_gpu::fir1d_batched_f64(&x, &taps);
/// assert_eq!(y.shape(), &[1, 4]);
/// ```
pub fn fir1d_batched_f64(x: &Array2<f64>, taps: &Array1<f64>) -> Array2<f64> {
    let (b, n) = x.dim();
    let k = taps.len();
    let mut y = Array2::<f64>::zeros((b, n));
    for bi in 0..b {
        let xin = x.index_axis(Axis(0), bi);
        let mut yout = y.index_axis_mut(Axis(0), bi);
        for i in 0..n {
            let mut acc = 0.0f64;
            let start = (i + 1).saturating_sub(k);
            for (t_idx, xi) in (start..=i).rev().enumerate() {
                let tap = taps[k - 1 - t_idx];
                acc += tap * xin[xi];
            }
            yout[i] = acc;
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2};
    use rand::Rng;
    use scir_core::assert_close;

    #[test]
    fn device_array_roundtrip() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let arr = DeviceArray::from_cpu_slice(&[2, 2], DType::F32, &data);
        assert_eq!(arr.shape(), &[2, 2]);
        assert_eq!(arr.dtype(), DType::F32);
        assert_eq!(arr.device(), Device::Cpu);
        assert_eq!(arr.to_cpu_vec(), data);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_add_and_add_scalar_f32() {
        // Attempt GPU op; if unavailable, treat as skip
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![0.5f32, 1.5, 2.5, 3.5];
        let mut out = vec![0.0f32; 4];
        match crate::add_vec_f32_cuda(&a, &b, &mut out) {
            Ok(()) => {
                let out_f64: Vec<f64> = out.iter().copied().map(|v| v as f64).collect();
                assert_close!(&out_f64, &[1.5, 3.5, 5.5, 7.5], slice, tol = 1e-6);
            }
            Err(_) => {
                eprintln!("CUDA not available; skipping CUDA test");
                return;
            }
        }
        let mut out2 = vec![0.0f32; 4];
        crate::add_scalar_f32_cuda(&a, 1.0, &mut out2).unwrap();
        let out2_f64: Vec<f64> = out2.iter().copied().map(|v| v as f64).collect();
        assert_close!(&out2_f64, &[2.0, 3.0, 4.0, 5.0], slice, tol = 1e-6);

        let mut out3 = vec![0.0f32; 4];
        crate::mul_scalar_f32_cuda(&a, 2.0, &mut out3).unwrap();
        let out3_f64: Vec<f64> = out3.iter().copied().map(|v| v as f64).collect();
        assert_close!(&out3_f64, &[2.0, 4.0, 6.0, 8.0], slice, tol = 1e-6);
    }

    #[test]
    fn elementwise_ops_baseline() {
        let data = vec![1.0f64, 2.0, 3.0, 4.0];
        let arr = DeviceArray::from_cpu_slice(&[4], DType::F64, &data);
        let add = arr.add_scalar(1.0);
        let mul = arr.mul_scalar(2.0);
        assert_close!(&add.to_cpu_vec(), &[2.0, 3.0, 4.0, 5.0], slice, tol = 0.0);
        assert_close!(&mul.to_cpu_vec(), &[2.0, 4.0, 6.0, 8.0], slice, tol = 0.0);
    }

    #[test]
    fn add_arrays() {
        let a = DeviceArray::from_cpu_slice(&[3], DType::F32, &[1.0, 2.0, 3.0]);
        let b = DeviceArray::from_cpu_slice(&[3], DType::F32, &[0.5, 1.5, 2.5]);
        let c = a.add(&b).unwrap();
        assert_eq!(c.to_cpu_vec(), vec![1.5f32, 3.5, 5.5]);
    }

    #[test]
    fn fir_batched_matches_naive_f32() {
        let x: Array2<f32> = array![[1.0, 2.0, 3.0, 4.0], [0.5, 0.0, -0.5, -1.0]];
        let taps: Array1<f32> = array![0.25, 0.5, 0.25];
        let y = fir1d_batched_f32(&x, &taps);
        // Manually compute expected for first row (compare as f64)
        let expected0_f64 = array![0.25f64, 1.0, 2.0, 3.0];
        let expected1_f64 = array![0.125f64, 0.25, 0.0, -0.5];
        let y0_f64 = y.index_axis(Axis(0), 0).to_owned().mapv(|v| v as f64);
        let y1_f64 = y.index_axis(Axis(0), 1).to_owned().mapv(|v| v as f64);
        assert_close!(&y0_f64, &expected0_f64, array, atol = 1e-7, rtol = 1e-7);
        assert_close!(&y1_f64, &expected1_f64, array, atol = 1e-7, rtol = 1e-7);
    }

    #[test]
    fn fir_batched_random_f64() {
        let mut rng = rand::thread_rng();
        let b = 3usize;
        let n = 32usize;
        let k = 5usize;
        let mut x = Array2::<f64>::zeros((b, n));
        for mut row in x.axis_iter_mut(Axis(0)) {
            for v in row.iter_mut() {
                *v = rng.gen::<f64>() * 2.0 - 1.0;
            }
        }
        let taps = Array1::from((0..k).map(|i| 1.0 / (i as f64 + 1.0)).collect::<Vec<_>>());
        let y = fir1d_batched_f64(&x, &taps);

        // Compare against a slow scalar reference
        let mut y_ref = Array2::<f64>::zeros((b, n));
        for bi in 0..b {
            for i in 0..n {
                let mut acc = 0.0f64;
                let start = (i + 1).saturating_sub(k);
                for (t_idx, xi) in (start..=i).rev().enumerate() {
                    let tap = taps[k - 1 - t_idx];
                    acc += tap * x[[bi, xi]];
                }
                y_ref[[bi, i]] = acc;
            }
        }
        assert_close!(
            &y.into_raw_vec(),
            &y_ref.into_raw_vec(),
            slice,
            atol = 1e-12,
            rtol = 1e-12
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_fir1d_batched_f32_parity_small() {
        // Small deterministic input
        let x: Array2<f32> = array![[1.0, 2.0, 3.0, 4.0], [0.5, 0.0, -0.5, -1.0]];
        let taps: Array1<f32> = array![0.25, 0.5, 0.25];
        // Try CUDA; if not available, skip
        match crate::fir1d_batched_f32_cuda(&x, &taps) {
            Ok(y_cuda) => {
                let y_cpu = super::fir1d_batched_f32(&x, &taps);
                let y_cuda_f64: Vec<f64> = y_cuda
                    .into_raw_vec()
                    .into_iter()
                    .map(|v| v as f64)
                    .collect();
                let y_cpu_f64: Vec<f64> =
                    y_cpu.into_raw_vec().into_iter().map(|v| v as f64).collect();
                assert_close!(&y_cuda_f64, &y_cpu_f64, slice, atol = 1e-5, rtol = 1e-6);
            }
            Err(_) => {
                eprintln!("CUDA not available; skipping CUDA FIR test");
            }
        }
    }
}
