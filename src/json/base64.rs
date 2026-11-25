//! Standard base64 of little-endian `f32` values, decoded straight into the
//! result buffer.

const INVALID: u8 = 0xff;

static DECODE: [u8; 256] = {
    let mut t = [INVALID; 256];
    let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut i = 0;
    while i < 64 {
        t[alphabet[i] as usize] = i as u8;
        i += 1;
    }
    t
};

fn quad(c: &[u8; 4]) -> Option<u32> {
    let (a, b, x, y) = (
        DECODE[c[0] as usize],
        DECODE[c[1] as usize],
        DECODE[c[2] as usize],
        DECODE[c[3] as usize],
    );
    if (a | b | x | y) & 0x80 != 0 {
        return None;
    }
    Some((a as u32) << 18 | (b as u32) << 12 | (x as u32) << 6 | y as u32)
}

/// Decode `s` (padding optional) and append the `f32` values to `out`.
/// Returns how many values were appended.
pub(crate) fn decode_f32_le(s: &[u8], out: &mut Vec<f32>) -> Result<usize, &'static str> {
    let s = match s {
        [rest @ .., b'=', b'='] => rest,
        [rest @ .., b'='] => rest,
        _ => s,
    };
    let tail = s.len() % 4;
    if tail == 1 {
        return Err("base64 length is not valid");
    }
    let bytes = s.len() / 4 * 3 + tail.saturating_sub(1);
    if bytes % 4 != 0 {
        return Err("base64 vector is not a whole number of f32 values");
    }
    let n = bytes / 4;
    out.reserve(n);
    // 16 characters are 12 bytes, three whole values
    let (chunks, rest) = s.as_chunks::<16>();
    for c in chunks {
        let mut b = [0u8; 12];
        for k in 0..4 {
            let q = quad(c[4 * k..4 * k + 4].try_into().expect("4 bytes")).ok_or(BAD_BYTE)?;
            b[3 * k] = (q >> 16) as u8;
            b[3 * k + 1] = (q >> 8) as u8;
            b[3 * k + 2] = q as u8;
        }
        out.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
        out.push(f32::from_le_bytes([b[4], b[5], b[6], b[7]]));
        out.push(f32::from_le_bytes([b[8], b[9], b[10], b[11]]));
    }
    let mut b = [0u8; 12];
    let mut len = 0;
    let (quads, part) = rest.as_chunks::<4>();
    for c in quads {
        let q = quad(c).ok_or(BAD_BYTE)?;
        b[len..len + 3].copy_from_slice(&[(q >> 16) as u8, (q >> 8) as u8, q as u8]);
        len += 3;
    }
    if !part.is_empty() {
        let mut c = [b'A'; 4];
        c[..part.len()].copy_from_slice(part);
        let q = quad(&c).ok_or(BAD_BYTE)?;
        let got = [(q >> 16) as u8, (q >> 8) as u8, q as u8];
        let keep = part.len() - 1;
        b[len..len + keep].copy_from_slice(&got[..keep]);
        len += keep;
    }
    for v in b[..len].as_chunks::<4>().0 {
        out.push(f32::from_le_bytes(*v));
    }
    Ok(n)
}

const BAD_BYTE: &str = "invalid base64 byte";

#[cfg(test)]
pub(crate) fn encode(bytes: &[u8]) -> String {
    const A: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::new();
    for c in bytes.chunks(3) {
        let n = (c[0] as u32) << 16
            | (*c.get(1).unwrap_or(&0) as u32) << 8
            | *c.get(2).unwrap_or(&0) as u32;
        for i in 0..4 {
            if i <= c.len() {
                s.push(A[(n >> (18 - 6 * i) & 63) as usize] as char);
            } else {
                s.push('=');
            }
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(s: &str) -> Result<Vec<f32>, &'static str> {
        let mut out = vec![9.0];
        let n = decode_f32_le(s.as_bytes(), &mut out)?;
        assert_eq!(n, out.len() - 1);
        Ok(out[1..].to_vec())
    }

    #[test]
    fn round_trips_every_length() {
        let values: Vec<f32> = (0..40)
            .map(|i| f32::from_bits(0x9E37_79B9u32.wrapping_mul(i + 1)))
            .collect();
        for n in 0..values.len() {
            let bytes: Vec<u8> = values[..n].iter().flat_map(|v| v.to_le_bytes()).collect();
            let s = encode(&bytes);
            let got = decode(&s).unwrap();
            let want: Vec<u32> = values[..n].iter().map(|v| v.to_bits()).collect();
            let got: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
            assert_eq!(got, want, "{n}");
            // padding is optional
            let got = decode(s.trim_end_matches('=')).unwrap();
            assert_eq!(got.len(), n);
        }
    }

    #[test]
    fn known_value() {
        // 1.0f32 little-endian is 00 00 80 3f
        assert_eq!(decode("AACAPw==").unwrap(), vec![1.0]);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(decode("AACA!w==").is_err());
        assert!(decode("A").is_err());
        assert!(decode("Zm9vYmE=").is_err());
        assert!(decode("AACAPwAACAPwAACAPw\"A").is_err());
        let mut long = "A".repeat(16);
        long.replace_range(13..14, "-");
        assert!(decode(&long).is_err());
    }

    #[test]
    fn every_byte_value_survives() {
        let bytes: Vec<u8> = (0..=255u8).chain((0..=255u8).rev()).collect();
        let s = encode(&bytes);
        let got: Vec<u8> = decode(&s)
            .unwrap()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(got, bytes);
    }
}
