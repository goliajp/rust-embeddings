//! One embedding vector as providers send it: a JSON array of numbers, or a
//! base64 string of little-endian `f32` values.

use std::fmt;

use serde::de::{self, Deserialize, Deserializer, SeqAccess, Visitor};

#[derive(Debug, PartialEq)]
pub(crate) struct Vector(pub Vec<f32>);

impl<'de> Deserialize<'de> for Vector {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_any(VectorVisitor)
    }
}

struct VectorVisitor;

impl<'de> Visitor<'de> for VectorVisitor {
    type Value = Vector;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("an array of numbers or a base64 string of little-endian f32 values")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Vector, A::Error> {
        let mut v = Vec::with_capacity(seq.size_hint().unwrap_or(1024));
        while let Some(x) = seq.next_element::<f32>()? {
            v.push(x);
        }
        Ok(Vector(v))
    }

    fn visit_str<E: de::Error>(self, s: &str) -> Result<Vector, E> {
        decode_f32_le(s).map(Vector).map_err(E::custom)
    }
}

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

/// Decode standard base64 (padding optional) into bytes.
fn decode_base64(s: &[u8]) -> Result<Vec<u8>, String> {
    let s = match s {
        [rest @ .., b'=', b'='] => rest,
        [rest @ .., b'='] => rest,
        _ => s,
    };
    if s.len() % 4 == 1 {
        return Err(format!("base64 length {} is not valid", s.len()));
    }
    let mut out = Vec::with_capacity(s.len() / 4 * 3 + 2);
    let (chunks, rest) = s.as_chunks::<4>();
    for c in chunks {
        let a = sextet(c[0])?;
        let b = sextet(c[1])?;
        let x = sextet(c[2])?;
        let y = sextet(c[3])?;
        let n = (a << 18) | (b << 12) | (x << 6) | y;
        out.extend_from_slice(&[(n >> 16) as u8, (n >> 8) as u8, n as u8]);
    }
    if !rest.is_empty() {
        let mut n = 0u32;
        for (i, &ch) in rest.iter().enumerate() {
            n |= sextet(ch)? << (18 - 6 * i);
        }
        out.push((n >> 16) as u8);
        if rest.len() == 3 {
            out.push((n >> 8) as u8);
        }
    }
    Ok(out)
}

fn sextet(ch: u8) -> Result<u32, String> {
    match DECODE[ch as usize] {
        INVALID => Err(format!("invalid base64 byte {ch:#04x}")),
        v => Ok(v as u32),
    }
}

fn decode_f32_le(s: &str) -> Result<Vec<f32>, String> {
    let bytes = decode_base64(s.as_bytes())?;
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "base64 vector has {} bytes, not a whole number of f32 values",
            bytes.len()
        ));
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode(bytes: &[u8]) -> String {
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

    #[test]
    fn base64_known_vectors() {
        // rfc 4648 section 10
        for (plain, coded) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ] {
            assert_eq!(decode_base64(coded.as_bytes()).unwrap(), plain.as_bytes());
            assert_eq!(encode(plain.as_bytes()), coded);
        }
        assert_eq!(decode_base64(b"Zm9vYg").unwrap(), b"foob");
    }

    #[test]
    fn base64_round_trips_every_byte() {
        let bytes: Vec<u8> = (0..=255u8).chain((0..=255u8).rev()).collect();
        for len in 0..bytes.len() {
            assert_eq!(
                decode_base64(encode(&bytes[..len]).as_bytes()).unwrap(),
                &bytes[..len]
            );
        }
    }

    #[test]
    fn base64_rejects_bad_input() {
        assert!(decode_base64(b"Zm9v!A==").is_err());
        assert!(decode_base64(b"Z").is_err());
        assert!(decode_f32_le("Zm9vYmE=").is_err());
    }

    #[test]
    fn vector_from_floats_and_base64_agree() {
        let values = [0.25f32, -1.5, 3.0e-7, f32::MAX, -0.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let from_b64: Vector = serde_json::from_str(&format!("\"{}\"", encode(&bytes))).unwrap();
        let from_floats: Vector =
            serde_json::from_str("[0.25, -1.5, 3.0e-7, 3.4028235e38, -0.0]").unwrap();
        assert_eq!(from_b64.0, values);
        assert_eq!(from_floats.0, values);
    }

    #[test]
    fn vector_rejects_other_json() {
        assert!(serde_json::from_str::<Vector>("{}").is_err());
        assert!(serde_json::from_str::<Vector>("[\"a\"]").is_err());
    }
}
