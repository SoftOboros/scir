//! wgpu (Vulkan/Metal/DX12) compute backend: device init + WGSL kernels for
//! elementwise add and 2D image resize. Mirrors the `cuda` module's per-call
//! context creation and `Result<_, GpuError>` dispatch style (see PLAN.md §6.4:
//! GPU output must always be verifiable against the CPU baseline).

use crate::GpuError;
use wgpu::util::DeviceExt;

const ELEMENTWISE_ADD_WGSL: &str = include_str!("kernels/elementwise_add.wgsl");
const RESIZE2D_WGSL: &str = include_str!("kernels/resize2d.wgsl");

/// Uniform parameters for the `resize2d` kernel. Field order and types must
/// match the WGSL `Params` struct exactly (std140 uniform layout).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Resize2dParams {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    mode: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Lazily-initialized wgpu device/queue context. Created fresh per call,
/// matching `cuda::CudaCtx`'s creation style rather than caching a global
/// context, for consistency with the existing CUDA dispatch functions.
pub(crate) struct WgpuCtx {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) adapter_name: String,
}

impl WgpuCtx {
    pub(crate) fn create_default() -> Result<Self, GpuError> {
        pollster::block_on(Self::create_default_async())
    }

    async fn create_default_async() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|e| GpuError::BackendUnavailable(format!("wgpu adapter: {e}")))?;
        let info = adapter.get_info();
        let adapter_name = format!("{} ({:?})", info.name, info.backend);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .map_err(|e| GpuError::BackendUnavailable(format!("wgpu device: {e}")))?;
        Ok(Self {
            device,
            queue,
            adapter_name,
        })
    }
}

/// Return the name (and backend) of the adapter wgpu selects on this
/// machine, for diagnostic/verification purposes.
///
/// # Errors
/// Returns [`GpuError::BackendUnavailable`] if no wgpu adapter/device is available.
pub fn adapter_name() -> Result<String, GpuError> {
    Ok(WgpuCtx::create_default()?.adapter_name)
}

/// Copy `buffer` (size `len_bytes`) back to a CPU-owned `Vec<f32>` via a
/// mapped staging buffer.
fn readback_f32(
    ctx: &WgpuCtx,
    buffer: &wgpu::Buffer,
    len_bytes: u64,
) -> Result<Vec<f32>, GpuError> {
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scir-gpu-staging"),
        size: len_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("scir-gpu-readback"),
        });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, len_bytes);
    ctx.queue.submit(Some(encoder.finish()));

    let (tx, rx) = std::sync::mpsc::channel();
    staging.map_async(wgpu::MapMode::Read, .., move |res| {
        let _ = tx.send(res);
    });
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| GpuError::BackendUnavailable(format!("wgpu poll: {e:?}")))?;
    rx.recv()
        .map_err(|_| GpuError::BackendUnavailable("wgpu map_async channel closed".to_string()))?
        .map_err(|e| GpuError::BackendUnavailable(format!("wgpu buffer map: {e:?}")))?;

    let data = staging
        .get_mapped_range(..)
        .map_err(|e| GpuError::BackendUnavailable(format!("wgpu get_mapped_range: {e:?}")))?;
    let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    Ok(out)
}

/// Elementwise vector addition on the wgpu backend: `out[i] = a[i] + b[i]`.
/// Parity baseline matching `cuda::add_vec_f32_cuda`.
///
/// # Errors
/// Returns [`GpuError::BackendUnavailable`] if no wgpu adapter is available
/// or a dispatch step fails.
pub fn add_vec_f32_wgpu(a: &[f32], b: &[f32], out: &mut [f32]) -> Result<(), GpuError> {
    if a.len() != b.len() || a.len() != out.len() {
        return Err(GpuError::ShapeMismatch);
    }
    let ctx = WgpuCtx::create_default()?;
    let len_bytes = (a.len() * std::mem::size_of::<f32>()) as u64;

    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("elementwise_add"),
            source: wgpu::ShaderSource::Wgsl(ELEMENTWISE_ADD_WGSL.into()),
        });

    let a_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("a"),
            contents: bytemuck::cast_slice(a),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let b_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("b"),
            contents: bytemuck::cast_slice(b),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out"),
        size: len_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("elementwise_add_pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("elementwise_add_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("elementwise_add_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("elementwise_add_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let groups = (a.len() as u32).div_ceil(256);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    ctx.queue.submit(Some(encoder.finish()));

    let result = readback_f32(&ctx, &out_buf, len_bytes)?;
    out.copy_from_slice(&result);
    Ok(())
}

/// 2D image resize (single-channel f32) on the wgpu backend.
///
/// `mode` is `0` for nearest-neighbor, `1` for bilinear — matching
/// `crate::ResizeMode` (converted by the caller in `lib.rs`).
///
/// # Errors
/// Returns [`GpuError::BackendUnavailable`] if no wgpu adapter is available
/// or a dispatch step fails.
pub fn resize2d_f32_wgpu(
    src: &[f32],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    mode: u32,
) -> Result<Vec<f32>, GpuError> {
    if src.len() != (src_w * src_h) as usize {
        return Err(GpuError::ShapeMismatch);
    }
    let ctx = WgpuCtx::create_default()?;
    let out_len_bytes = (dst_w as u64) * (dst_h as u64) * (std::mem::size_of::<f32>() as u64);

    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("resize2d"),
            source: wgpu::ShaderSource::Wgsl(RESIZE2D_WGSL.into()),
        });

    let params = Resize2dParams {
        src_w,
        src_h,
        dst_w,
        dst_h,
        mode,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let params_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("resize2d_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let src_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("resize2d_src"),
            contents: bytemuck::cast_slice(src),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let dst_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("resize2d_dst"),
        size: out_len_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("resize2d_pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("resize2d_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: src_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: dst_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("resize2d_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("resize2d_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let groups_x = dst_w.div_ceil(16);
        let groups_y = dst_h.div_ceil(16);
        pass.dispatch_workgroups(groups_x, groups_y, 1);
    }
    ctx.queue.submit(Some(encoder.finish()));

    readback_f32(&ctx, &dst_buf, out_len_bytes)
}
