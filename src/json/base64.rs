//! Standard base64 of little-endian `f32` values, decoded straight into the
//! result buffer.

// set in a table entry for a byte outside the alphabet; the 24 bits below it
// hold the decoded bytes
const INVALID: u32 = 1 << 31;

/// One table per position in a group of four characters: the bits the
/// character contributes to the three decoded bytes, first byte lowest.
static DECODE: [[u32; 256]; 4] = {
    let mut t = [[INVALID; 256]; 4];
    let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut i = 0;
    while i < 64 {
        let mut k = 0;
        while k < 4 {
            let big = (i as u32) << (18 - 6 * k);
            t[k][alphabet[i] as usize] = (big >> 16) | (big & 0xff00) | (big & 0xff) << 16;
            k += 1;
        }
        i += 1;
    }
    t
};

fn quad(c: &[u8; 4]) -> u32 {
    DECODE[0][c[0] as usize]
        | DECODE[1][c[1] as usize]
        | DECODE[2][c[2] as usize]
        | DECODE[3][c[3] as usize]
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
    let start = out.len();
    out.resize(start + n, 0.0);
    let (groups, tail_values) = out[start..].as_chunks_mut::<3>();
    // 16 characters are 12 bytes, three whole values
    let (chunks, rest) = s.as_chunks::<16>();
    let mut seen = 0;
    for (g, c) in groups.iter_mut().zip(chunks) {
        let [c0, c1, c2, c3] = c.as_chunks::<4>().0 else {
            unreachable!("16 bytes are four groups of four")
        };
        let (w0, w1, w2, w3) = (quad(c0), quad(c1), quad(c2), quad(c3));
        seen |= w0 | w1 | w2 | w3;
        *g = [
            f32::from_bits(w0 | w1 << 24),
            f32::from_bits(w1 >> 8 | w2 << 16),
            f32::from_bits(w2 >> 16 | w3 << 8),
        ];
    }
    let mut b = [0u8; 12];
    for (b, c) in b.as_chunks_mut::<3>().0.iter_mut().zip(rest.chunks(4)) {
        let mut padded = [b'A'; 4];
        padded[..c.len()].copy_from_slice(c);
        let w = quad(&padded);
        seen |= w;
        b.copy_from_slice(&w.to_le_bytes()[..3]);
    }
    for (v, b) in tail_values.iter_mut().zip(b.as_chunks::<4>().0) {
        *v = f32::from_le_bytes(*b);
    }
    if seen & INVALID != 0 {
        return Err(BAD_BYTE);
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
