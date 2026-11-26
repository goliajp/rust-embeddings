//! Reader for safetensors weight files: a little-endian u64 header length, a
//! JSON header giving each tensor's dtype, shape and byte range, then the
//! data. The file is mapped read-only, so f32 tensors are read in place.

use std::collections::HashMap;
use std::fs::File;
use std::ops::Deref;
use std::path::Path;
use std::sync::Arc;

use super::fail;
use crate::error::Result;
use crate::json::read::Cursor;

// the limit the reference implementation puts on the header
const MAX_HEADER: u64 = 100_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Dtype {
    F32,
    F16,
    Bf16,
    /// Any other dtype, with its element size when known.
    Other(Option<usize>),
}

impl Dtype {
    fn parse(s: &str) -> Self {
        match s {
            "F32" => Self::F32,
            "F16" => Self::F16,
            "BF16" => Self::Bf16,
            "F64" | "I64" | "U64" => Self::Other(Some(8)),
            "I32" | "U32" => Self::Other(Some(4)),
            "I16" | "U16" => Self::Other(Some(2)),
            "I8" | "U8" | "BOOL" | "F8_E4M3" | "F8_E5M2" => Self::Other(Some(1)),
            _ => Self::Other(None),
        }
    }

    fn size(self) -> Option<usize> {
        match self {
            Self::F32 => Some(4),
            Self::F16 | Self::Bf16 => Some(2),
            Self::Other(s) => s,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Info {
    pub(crate) dtype: Dtype,
    pub(crate) shape: Vec<usize>,
    /// Byte range in the file.
    pub(crate) start: usize,
    pub(crate) end: usize,
}

/// The whole file, mapped or read.
pub(crate) struct Bytes {
    ptr: *const u8,
    len: usize,
    #[cfg(not(all(unix, target_pointer_width = "64")))]
    _owned: Vec<u64>,
}

// safety: read-only memory owned by this value
unsafe impl Send for Bytes {}
unsafe impl Sync for Bytes {}

#[cfg(all(unix, target_pointer_width = "64"))]
mod sys {
    unsafe extern "C" {
        pub(super) fn mmap(
            addr: *mut u8,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            off: i64,
        ) -> *mut u8;
        pub(super) fn munmap(addr: *mut u8, len: usize) -> i32;
    }
    // the same values on linux and the bsds, macos included
    pub(super) const PROT_READ: i32 = 1;
    pub(super) const MAP_PRIVATE: i32 = 2;
}

impl Bytes {
    #[cfg(all(unix, target_pointer_width = "64"))]
    fn open(file: &File, len: usize) -> std::io::Result<Self> {
        use std::os::fd::AsRawFd;
        // safety: a private read-only mapping of an open file; the model
        // cache never rewrites a file in place
        let ptr = unsafe {
            sys::mmap(
                std::ptr::null_mut(),
                len,
                sys::PROT_READ,
                sys::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr as isize == -1 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(Self { ptr, len })
    }

    #[cfg(not(all(unix, target_pointer_width = "64")))]
    fn open(mut file: &File, len: usize) -> std::io::Result<Self> {
        use std::io::Read;
        let mut owned = vec![0u64; len.div_ceil(8)];
        // safety: the u64 buffer covers len bytes
        let bytes = unsafe { std::slice::from_raw_parts_mut(owned.as_mut_ptr() as *mut u8, len) };
        file.read_exact(bytes)?;
        Ok(Self {
            ptr: owned.as_ptr() as *const u8,
            len,
            _owned: owned,
        })
    }

    fn get(&self) -> &[u8] {
        // safety: ptr covers len readable bytes for the life of self
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

#[cfg(all(unix, target_pointer_width = "64"))]
impl Drop for Bytes {
    fn drop(&mut self) {
        // safety: the mapping made in open, unmapped once
        unsafe { sys::munmap(self.ptr as *mut u8, self.len) };
    }
}

/// f32 values, either read in place from the file or converted into memory.
pub(crate) enum F32s {
    Mapped {
        bytes: Arc<Bytes>,
        start: usize,
        len: usize,
    },
    Owned(Vec<f32>),
}

impl Deref for F32s {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        match self {
            // safety: start is 4-aligned within a page-aligned mapping and
            // start + 4 * len is inside it, checked in `Safetensors::f32s`
            Self::Mapped { bytes, start, len } => unsafe {
                std::slice::from_raw_parts(bytes.ptr.add(*start) as *const f32, *len)
            },
            Self::Owned(v) => v,
        }
    }
}

pub(crate) struct Safetensors {
    bytes: Arc<Bytes>,
    tensors: HashMap<String, Info>,
}

fn header_error(what: &str) -> crate::error::Error {
    fail(format!("not a safetensors file: {what}"))
}

impl Safetensors {
    pub(crate) fn open(path: &Path) -> Result<Self> {
        let file =
            File::open(path).map_err(|e| fail(format!("cannot open {}: {e}", path.display())))?;
        let len = file
            .metadata()
            .map_err(|e| fail(format!("cannot read {}: {e}", path.display())))?
            .len();
        if len < 8 {
            return Err(header_error("shorter than 8 bytes"));
        }
        let len = usize::try_from(len).map_err(|_| header_error("too large for this platform"))?;
        let bytes = Bytes::open(&file, len)
            .map_err(|e| fail(format!("cannot map {}: {e}", path.display())))?;
        let tensors = parse(bytes.get())?;
        Ok(Self {
            bytes: Arc::new(bytes),
            tensors,
        })
    }

    pub(crate) fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// A tensor of exactly `shape`, as f32 (f16 and bf16 are widened, which is exact).
    pub(crate) fn f32s(&self, name: &str, shape: &[usize]) -> Result<F32s> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| fail(format!("the weights have no tensor {name}")))?;
        if info.shape != shape {
            return Err(fail(format!(
                "tensor {name} has shape {:?}, expected {shape:?}",
                info.shape
            )));
        }
        let raw = &self.bytes.get()[info.start..info.end];
        let n: usize = shape.iter().product();
        Ok(match info.dtype {
            Dtype::F32 if cfg!(target_endian = "little") && info.start % 4 == 0 => F32s::Mapped {
                bytes: self.bytes.clone(),
                start: info.start,
                len: n,
            },
            Dtype::F32 => F32s::Owned(
                raw.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect(),
            ),
            Dtype::F16 => F32s::Owned(
                raw.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                    .collect(),
            ),
            Dtype::Bf16 => F32s::Owned(
                raw.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                    .collect(),
            ),
            Dtype::Other(_) => {
                return Err(fail(format!(
                    "tensor {name} is not a float tensor (F32, F16 or BF16)"
                )));
            }
        })
    }
}

/// The tensor table, every entry checked against the file length.
fn parse(file: &[u8]) -> Result<HashMap<String, Info>> {
    let n = u64::from_le_bytes(file[..8].try_into().expect("8 bytes"));
    if n > MAX_HEADER || n > (file.len() - 8) as u64 {
        return Err(header_error("header length out of range"));
    }
    let data = 8 + n as usize;
    type Raw = (String, String, Vec<u64>, Vec<u64>);
    let mut raw: Vec<Raw> = Vec::new();
    let mut c = Cursor::new(&file[8..data]);
    c.object(|c, key| {
        if key == "__metadata__" {
            return c.skip();
        }
        let (mut dtype, mut shape, mut offsets) = (String::new(), Vec::new(), Vec::new());
        c.object(|c, field| match field {
            "dtype" => {
                dtype = c.str()?.into_owned();
                Ok(())
            }
            "shape" => c.array(|c| c.u64().map(|v| shape.push(v))),
            "data_offsets" => c.array(|c| c.u64().map(|v| offsets.push(v))),
            _ => c.skip(),
        })?;
        raw.push((key.to_string(), dtype, shape, offsets));
        Ok(())
    })
    .and_then(|_| c.end())
    .map_err(|f| header_error(&format!("{} at byte {}", f.what, f.offset + 8)))?;

    let mut tensors = HashMap::with_capacity(raw.len());
    for (name, dtype, shape, offsets) in raw {
        let bad = |what: &str| fail(format!("tensor {name} in the weights header: {what}"));
        let [begin, end] = offsets[..] else {
            return Err(bad("data_offsets is not two numbers"));
        };
        let (start, stop) = (data as u64)
            .checked_add(begin)
            .zip((data as u64).checked_add(end))
            .ok_or_else(|| bad("offset overflow"))?;
        if begin > end || stop > file.len() as u64 {
            return Err(bad("data outside the file"));
        }
        let shape: Vec<usize> = shape
            .into_iter()
            .map(usize::try_from)
            .collect::<std::result::Result<_, _>>()
            .map_err(|_| bad("shape too large"))?;
        let dtype = Dtype::parse(&dtype);
        if let Some(size) = dtype.size() {
            let want = shape
                .iter()
                .try_fold(size as u64, |a, &d| a.checked_mul(d as u64));
            if want != Some(end - begin) {
                return Err(bad("byte length does not match dtype and shape"));
            }
        }
        tensors.insert(
            name,
            Info {
                dtype,
                shape,
                start: start as usize,
                end: stop as usize,
            },
        );
    }
    Ok(tensors)
}

pub(crate) fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match exp {
        0 if man == 0 => sign,
        0 => {
            // subnormal: man * 2^-24, exact in f32
            let v = man as f32 * (1.0 / 16_777_216.0);
            return if sign != 0 { -v } else { v };
        }
        31 => sign | 0x7f80_0000 | (man << 13),
        _ => sign | ((exp + 112) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

pub(crate) fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

#[cfg(test)]
#[path = "safetensors_tests.rs"]
mod tests;
