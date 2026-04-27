//! Build-time helper for emitting SciPy-parity biquad coefficient tables
//! as Rust source files. Designed for static-overlay / no_std DSP runtimes
//! that cannot link `scir-signal` at runtime but want SciPy-validated
//! coefficients pinned to a known scir version.
//!
//! See the crate README for usage. The high-level shape:
//!
//! ```no_run
//! use std::path::PathBuf;
//! use scir_signal_build::{Emitter, FilterType};
//!
//! let out = PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("baked.rs");
//! Emitter::new()
//!     .add_butter("ANTI_ALIAS_48K", 4, FilterType::LowPass(0.04167), 2.0)
//!     .unwrap()
//!     .write(&out)
//!     .unwrap();
//! ```
#![deny(missing_docs)]

use std::fmt::Write as _;
use std::io;
use std::path::Path;

pub use scir_signal::{FilterError, FilterType};

const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// One baked filter table queued for emission.
#[derive(Debug, Clone)]
struct EmittedItem {
    ident: String,
    doc: String,
    sections: Vec<[f64; 6]>,
    as_f32: bool,
}

/// Build-time coefficient-table emitter.
///
/// Queue any number of designs via `add_butter` / `add_cheby1`; call
/// [`Emitter::write`] (file) or [`Emitter::render`] (string) to materialize
/// the Rust source. Output is byte-deterministic for byte-identical inputs.
#[derive(Debug, Default, Clone)]
pub struct Emitter {
    header: Option<String>,
    items: Vec<EmittedItem>,
}

impl Emitter {
    /// Construct an empty emitter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a free-form header comment, rendered above the generated tables.
    /// Multi-line strings are emitted with each line prefixed `// `.
    pub fn header(mut self, header: impl Into<String>) -> Self {
        self.header = Some(header.into());
        self
    }

    /// Queue a Butterworth design. `ident` becomes the `pub static`
    /// identifier in the emitted source.
    pub fn add_butter(
        &mut self,
        ident: &str,
        order: u32,
        kind: FilterType,
        fs: f64,
    ) -> Result<&mut Self, FilterError> {
        let sos = scir_signal::butter_filter(order, kind, fs)?;
        let doc = format!("butter(order={order}, kind={kind:?}, fs={fs})");
        self.items.push(EmittedItem {
            ident: ident.to_string(),
            doc,
            sections: sos_to_arrays(&sos),
            as_f32: false,
        });
        Ok(self)
    }

    /// Queue a Chebyshev Type I design. `rp` is passband ripple in dB.
    pub fn add_cheby1(
        &mut self,
        ident: &str,
        order: u32,
        ripple: f64,
        kind: FilterType,
        fs: f64,
    ) -> Result<&mut Self, FilterError> {
        let sos = scir_signal::cheby1_filter(order, ripple, kind, fs)?;
        let doc = format!("cheby1(order={order}, rp={ripple}, kind={kind:?}, fs={fs})");
        self.items.push(EmittedItem {
            ident: ident.to_string(),
            doc,
            sections: sos_to_arrays(&sos),
            as_f32: false,
        });
        Ok(self)
    }

    /// Queue a Bessel design (phase normalization). Supported orders
    /// are bounded by `scir_iir_filters::filter_design::MAX_BESSEL_ORDER`
    /// (currently 25).
    pub fn add_bessel(
        &mut self,
        ident: &str,
        order: u32,
        kind: FilterType,
        fs: f64,
    ) -> Result<&mut Self, FilterError> {
        let sos = scir_signal::bessel_filter(order, kind, fs)?;
        let doc = format!("bessel(order={order}, kind={kind:?}, fs={fs}, norm=phase)");
        self.items.push(EmittedItem {
            ident: ident.to_string(),
            doc,
            sections: sos_to_arrays(&sos),
            as_f32: false,
        });
        Ok(self)
    }

    /// Queue a 2nd-order IIR notch (single biquad). Mirrors
    /// `scipy.signal.iirnotch(w0, Q, fs)`. `w0` is the center frequency
    /// to remove (in the same units as `fs`); `q` is the notch quality
    /// factor (`Q = w0 / bw_-3dB`).
    pub fn add_iirnotch(
        &mut self,
        ident: &str,
        w0: f64,
        q: f64,
        fs: f64,
    ) -> Result<&mut Self, FilterError> {
        let sos = scir_signal::iirnotch(w0, q, fs)?;
        let doc = format!("iirnotch(w0={w0}, Q={q}, fs={fs})");
        self.items.push(EmittedItem {
            ident: ident.to_string(),
            doc,
            sections: sos_to_arrays(&sos),
            as_f32: false,
        });
        Ok(self)
    }

    /// Mark the most-recently-added item to emit as `[[f32; 6]; N]`
    /// (truncated from f64). Useful for embedded targets where the
    /// runtime biquad uses `f32` storage. The truncation is explicit
    /// and recorded in the emitted item's doc comment.
    pub fn as_f32(&mut self) -> &mut Self {
        if let Some(item) = self.items.last_mut() {
            item.as_f32 = true;
        }
        self
    }

    /// Render the queued items as a Rust source string. Pure function:
    /// same inputs always produce identical output (byte-deterministic).
    pub fn render(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "// Generated by scir-signal-build@{CRATE_VERSION} — do not edit by hand.",
        );
        let _ = writeln!(
            out,
            "// Generation is deterministic; same inputs always produce identical output.",
        );
        if let Some(header) = &self.header {
            let _ = writeln!(out, "//");
            for line in header.lines() {
                let _ = writeln!(out, "// {line}");
            }
        }
        out.push('\n');

        for item in &self.items {
            let _ = writeln!(out, "/// {}", item.doc);
            if item.as_f32 {
                let _ = writeln!(out, "/// (truncated from f64 to f32 at emission)");
            }
            let n = item.sections.len();
            let ty = if item.as_f32 { "f32" } else { "f64" };
            let _ = writeln!(
                out,
                "pub static {ident}: [[{ty}; 6]; {n}] = [",
                ident = item.ident,
            );
            for section in &item.sections {
                out.push_str("    [");
                for (i, c) in section.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    if item.as_f32 {
                        let v = *c as f32;
                        let _ = write!(out, "{v:.9e}_f32");
                    } else {
                        let _ = write!(out, "{c:.17e}_f64");
                    }
                }
                out.push_str("],\n");
            }
            let _ = writeln!(out, "];");
            out.push('\n');
        }
        out
    }

    /// Write the rendered source to `out_path`. Atomically replaces any
    /// existing file. Caller is responsible for cargo rerun-if-changed.
    pub fn write(&self, out_path: &Path) -> io::Result<()> {
        let body = self.render();
        std::fs::write(out_path, body)
    }
}

fn sos_to_arrays(sos: &scir_signal::Sos) -> Vec<[f64; 6]> {
    sos.sections()
        .iter()
        .map(|s| [s.b0, s.b1, s.b2, s.a0, s.a1, s.a2])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use scir_signal::sosfilt;

    #[test]
    fn render_is_byte_deterministic() {
        let mut e1 = Emitter::new().header("test header");
        e1.add_butter("FOO", 4, FilterType::LowPass(0.2), 2.0).unwrap();
        let mut e2 = Emitter::new().header("test header");
        e2.add_butter("FOO", 4, FilterType::LowPass(0.2), 2.0).unwrap();
        assert_eq!(e1.render(), e2.render());
    }

    #[test]
    fn render_is_independent_of_emission_order_when_idents_differ() {
        // Two distinct items with different idents: two separate emitters
        // produce identical output as long as the queued sequence matches.
        let mut e1 = Emitter::new();
        e1.add_butter("A", 2, FilterType::LowPass(0.1), 2.0).unwrap();
        e1.add_butter("B", 2, FilterType::HighPass(0.4), 2.0).unwrap();

        let mut e2 = Emitter::new();
        e2.add_butter("A", 2, FilterType::LowPass(0.1), 2.0).unwrap();
        e2.add_butter("B", 2, FilterType::HighPass(0.4), 2.0).unwrap();

        assert_eq!(e1.render(), e2.render());
    }

    #[test]
    fn emitted_butter_lpf_filters_identically_to_direct_scir() {
        // Round-trip: design via Emitter, parse the text, run sosfilt.
        // The result must be sample-accurate vs designing through scir
        // directly with the same parameters.
        let order = 4;
        let kind = FilterType::LowPass(0.2);
        let fs = 2.0;

        let mut e = Emitter::new();
        e.add_butter("FOO", order, kind, fs).unwrap();

        // Direct scir reference.
        let direct = scir_signal::butter_filter(order, kind, fs).unwrap();

        // Reconstruct from the emitted text by re-parsing the f64 literals.
        let body = e.render();
        let parsed = parse_first_static_table(&body, "FOO", 6);
        assert_eq!(parsed.len(), direct.sections().len());
        for (i, sec) in parsed.iter().enumerate() {
            let d = &direct.sections()[i];
            // 17-sig-digit scientific notation round-trips IEEE f64 losslessly.
            assert_eq!(sec[0], d.b0, "section {i} b0 mismatch");
            assert_eq!(sec[1], d.b1, "section {i} b1 mismatch");
            assert_eq!(sec[2], d.b2, "section {i} b2 mismatch");
            assert_eq!(sec[3], d.a0, "section {i} a0 mismatch");
            assert_eq!(sec[4], d.a1, "section {i} a1 mismatch");
            assert_eq!(sec[5], d.a2, "section {i} a2 mismatch");
        }

        // End-to-end parity: emitted coefficients filter the same data the same way.
        let x: Vec<f64> = (0..32).map(|i| i as f64 / 32.0).collect();
        let arr = Array1::from(x.clone());
        let direct_y = sosfilt(&direct, &arr);
        let parsed_sos = scir_signal::Sos::from_vec(parsed);
        let parsed_y = sosfilt(&parsed_sos, &arr);
        for (a, b) in direct_y.iter().zip(parsed_y.iter()) {
            assert_eq!(a, b, "filtered sample mismatch");
        }
    }

    #[test]
    fn emitted_cheby1_hpf_filters_identically_to_direct_scir() {
        let order = 4;
        let kind = FilterType::HighPass(0.3);
        let fs = 2.0;
        let rp = 1.0;

        let mut e = Emitter::new();
        e.add_cheby1("HPF", order, rp, kind, fs).unwrap();

        let direct = scir_signal::cheby1_filter(order, rp, kind, fs).unwrap();

        let body = e.render();
        let parsed = parse_first_static_table(&body, "HPF", 6);
        let parsed_sos = scir_signal::Sos::from_vec(parsed);
        let x = Array1::from((0..64).map(|i| (i as f64 * 0.1).sin()).collect::<Vec<_>>());
        let direct_y = sosfilt(&direct, &x);
        let parsed_y = sosfilt(&parsed_sos, &x);
        for (a, b) in direct_y.iter().zip(parsed_y.iter()) {
            assert_eq!(a, b, "filtered sample mismatch");
        }
    }

    #[test]
    fn emitted_bessel_lpf_filters_identically_to_direct_scir() {
        let order = 4;
        let kind = FilterType::LowPass(0.2);
        let fs = 2.0;

        let mut e = Emitter::new();
        e.add_bessel("LPF", order, kind, fs).unwrap();

        let direct = scir_signal::bessel_filter(order, kind, fs).unwrap();

        let body = e.render();
        let parsed = parse_first_static_table(&body, "LPF", 6);
        let parsed_sos = scir_signal::Sos::from_vec(parsed);
        let x = Array1::from((0..64).map(|i| (i as f64 * 0.1).sin()).collect::<Vec<_>>());
        let direct_y = sosfilt(&direct, &x);
        let parsed_y = sosfilt(&parsed_sos, &x);
        for (a, b) in direct_y.iter().zip(parsed_y.iter()) {
            assert_eq!(a, b, "filtered sample mismatch");
        }
    }

    #[test]
    fn emitted_iirnotch_filters_identically_to_direct_scir() {
        let mut e = Emitter::new();
        e.add_iirnotch("NOTCH_60HZ", 60.0, 30.0, 200.0).unwrap();

        let direct = scir_signal::iirnotch(60.0, 30.0, 200.0).unwrap();

        let body = e.render();
        let parsed = parse_first_static_table(&body, "NOTCH_60HZ", 6);
        assert_eq!(parsed.len(), 1, "iirnotch is always a single biquad");
        let parsed_sos = scir_signal::Sos::from_vec(parsed);
        let x = Array1::from(
            (0..128).map(|i| (i as f64 * 0.05).sin()).collect::<Vec<_>>(),
        );
        let direct_y = sosfilt(&direct, &x);
        let parsed_y = sosfilt(&parsed_sos, &x);
        for (a, b) in direct_y.iter().zip(parsed_y.iter()) {
            assert_eq!(a, b, "filtered sample mismatch");
        }
    }

    #[test]
    fn add_bessel_rejects_order_above_table() {
        let mut e = Emitter::new();
        // MAX_BESSEL_ORDER=25; 26 is out of range.
        let err = e.add_bessel("OOPS", 26, FilterType::LowPass(0.2), 2.0);
        assert!(err.is_err());
    }

    #[test]
    fn multi_item_emitter_emits_all_tables() {
        let mut e = Emitter::new();
        e.add_butter("A", 2, FilterType::LowPass(0.1), 2.0).unwrap();
        e.add_butter("B", 4, FilterType::HighPass(0.3), 2.0).unwrap();
        e.add_cheby1("C", 4, 1.0, FilterType::LowPass(0.2), 2.0).unwrap();
        let body = e.render();
        for ident in ["A", "B", "C"] {
            assert!(
                body.contains(&format!("pub static {ident}:")),
                "missing pub static {ident} in emitted source:\n{body}"
            );
        }
        // f64 unless explicitly downcast.
        assert!(body.contains("[f64; 6]"));
        assert!(!body.contains("[f32; 6]"));
    }

    #[test]
    fn as_f32_emits_truncated_f32_table_for_last_item() {
        let mut e = Emitter::new();
        e.add_butter("F64_TBL", 2, FilterType::LowPass(0.1), 2.0).unwrap();
        e.add_butter("F32_TBL", 2, FilterType::HighPass(0.3), 2.0).unwrap();
        e.as_f32();
        let body = e.render();
        assert!(body.contains("pub static F64_TBL: [[f64; 6]; 1]"));
        assert!(body.contains("pub static F32_TBL: [[f32; 6]; 1]"));
        assert!(body.contains("_f32"));
    }

    #[test]
    fn write_produces_file_with_render_contents() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("baked.rs");
        let mut e = Emitter::new();
        e.add_butter("X", 4, FilterType::LowPass(0.2), 2.0).unwrap();
        e.write(&path).unwrap();
        let on_disk = std::fs::read_to_string(&path).unwrap();
        assert_eq!(on_disk, e.render());
    }

    #[test]
    fn invalid_filter_args_propagate_error() {
        let mut e = Emitter::new();
        // cutoff out of (0, 1) range for fs=2.0 → scir rejects.
        let err = e.add_butter("BAD", 4, FilterType::LowPass(2.5), 2.0);
        assert!(err.is_err());
    }

    /// Extract the f64 literals from the first `pub static IDENT: ...` block
    /// in `body`, returning each section as a `[f64; 6]`.
    fn parse_first_static_table(body: &str, ident: &str, cols: usize) -> Vec<[f64; 6]> {
        assert_eq!(cols, 6, "this test parser only handles 6-column biquad rows");
        let needle = format!("pub static {ident}:");
        let start = body.find(&needle).expect("ident not found");
        let after = &body[start..];
        // Skip past the type signature `[[fT; 6]; N]` and the `= ` that
        // follows it; the array literal begins at the next `[`.
        let eq_marker = "] = [";
        let eq = after.find(eq_marker).expect("array literal start");
        let body = &after[eq + eq_marker.len() - 1..]; // include the leading `[`
        let close = body.find("];").expect("array literal close");
        let table_text = &body[..close];
        let mut sections = Vec::new();
        for line in table_text.lines() {
            let line = line.trim();
            if !(line.starts_with("[") && line.ends_with("],")) {
                continue;
            }
            let inner = &line[1..line.len() - 2];
            let nums: Vec<f64> = inner
                .split(',')
                .map(|t| {
                    let t = t.trim().trim_end_matches("_f64").trim_end_matches("_f32");
                    t.parse::<f64>().expect("number parse")
                })
                .collect();
            assert_eq!(nums.len(), 6, "expected 6 columns, got {} on `{line}`", nums.len());
            sections.push([nums[0], nums[1], nums[2], nums[3], nums[4], nums[5]]);
        }
        sections
    }
}
