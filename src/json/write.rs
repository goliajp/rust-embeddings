//! Request body writer: flat objects of strings, string arrays and integers.
//!
//! Output is byte-identical to `serde_json::to_vec` for the same values, so
//! requests do not change shape between releases.

pub(crate) struct Writer {
    buf: Vec<u8>,
}

const LO: u64 = 0x0101_0101_0101_0101;
const HI: u64 = 0x8080_8080_8080_8080;

// true when any of the eight bytes is a control byte, '"' or '\'
fn needs_escape(w: u64) -> bool {
    let ctl = w.wrapping_sub(LO * 0x20) & !w;
    let q = w ^ (LO * b'"' as u64);
    let q = q.wrapping_sub(LO) & !q;
    let b = w ^ (LO * b'\\' as u64);
    let b = b.wrapping_sub(LO) & !b;
    (ctl | q | b) & HI != 0
}

// the escape serde_json writes for a byte, 0 when it needs none
static ESCAPE: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0;
    while i < 0x20 {
        t[i] = b'u';
        i += 1;
    }
    t[0x08] = b'b';
    t[0x09] = b't';
    t[0x0a] = b'n';
    t[0x0c] = b'f';
    t[0x0d] = b'r';
    t[b'"' as usize] = b'"';
    t[b'\\' as usize] = b'\\';
    t
};

impl Writer {
    pub(crate) fn with_capacity(n: usize) -> Self {
        Self {
            buf: Vec::with_capacity(n),
        }
    }

    /// Bytes written as they are: punctuation and keys known to need no escaping.
    pub(crate) fn raw(&mut self, s: &str) {
        self.buf.extend_from_slice(s.as_bytes());
    }

    pub(crate) fn byte(&mut self, b: u8) {
        self.buf.push(b);
    }

    pub(crate) fn uint(&mut self, v: u64) {
        let mut tmp = [0u8; 20];
        let mut i = tmp.len();
        let mut v = v;
        loop {
            i -= 1;
            tmp[i] = b'0' + (v % 10) as u8;
            v /= 10;
            if v == 0 {
                break;
            }
        }
        self.buf.extend_from_slice(&tmp[i..]);
    }

    /// `"key":`
    pub(crate) fn key(&mut self, k: &str) {
        self.buf.push(b'"');
        self.buf.extend_from_slice(k.as_bytes());
        self.buf.extend_from_slice(b"\":");
    }

    /// A JSON string, quoted and escaped.
    pub(crate) fn str(&mut self, s: &str) {
        let b = s.as_bytes();
        self.buf.reserve(b.len() + 2);
        self.buf.push(b'"');
        let mut start = 0;
        let mut i = 0;
        while i + 8 <= b.len() {
            let w = u64::from_le_bytes(b[i..i + 8].try_into().expect("8 bytes"));
            if needs_escape(w) {
                for j in i..i + 8 {
                    if ESCAPE[b[j] as usize] != 0 {
                        self.escape(&b[start..j], b[j]);
                        start = j + 1;
                    }
                }
            }
            i += 8;
        }
        for j in i..b.len() {
            if ESCAPE[b[j] as usize] != 0 {
                self.escape(&b[start..j], b[j]);
                start = j + 1;
            }
        }
        self.buf.extend_from_slice(&b[start..]);
        self.buf.push(b'"');
    }

    fn escape(&mut self, run: &[u8], c: u8) {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        self.buf.extend_from_slice(run);
        match ESCAPE[c as usize] {
            b'u' => self.buf.extend_from_slice(&[
                b'\\',
                b'u',
                b'0',
                b'0',
                HEX[(c >> 4) as usize],
                HEX[(c & 15) as usize],
            ]),
            e => self.buf.extend_from_slice(&[b'\\', e]),
        }
    }

    pub(crate) fn finish(self) -> Vec<u8> {
        self.buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quoted(s: &str) -> String {
        let mut w = Writer::with_capacity(0);
        w.str(s);
        String::from_utf8(w.finish()).unwrap()
    }

    #[test]
    fn matches_serde_json_on_every_byte_class() {
        let mut all = String::new();
        for c in (0u32..0x800).filter_map(char::from_u32) {
            all.push(c);
        }
        all.push_str("😀 \u{2028} \u{feff} \u{10ffff}");
        for len in 0..all.len() {
            if !all.is_char_boundary(len) {
                continue;
            }
            let s = &all[..len];
            assert_eq!(quoted(s), serde_json::to_string(s).unwrap(), "{len}");
        }
    }

    #[test]
    fn swar_finds_escapes_in_every_lane() {
        for special in [0u8, 0x1f, b'"', b'\\', b'\n'] {
            for lane in 0..19 {
                let mut bytes = vec![b'a'; 19];
                bytes[lane] = special;
                let s = String::from_utf8(bytes).unwrap();
                assert_eq!(quoted(&s), serde_json::to_string(&s).unwrap());
            }
        }
        // bytes at and above 0x80 are not control bytes
        let s = "\u{7f}\u{80}\u{ff}ÿÿÿÿÿÿÿÿ";
        assert_eq!(quoted(s), serde_json::to_string(s).unwrap());
    }

    #[test]
    fn integers() {
        for v in [0u64, 7, 10, 256, 4096, u32::MAX as u64, u64::MAX] {
            let mut w = Writer::with_capacity(0);
            w.uint(v);
            assert_eq!(w.finish(), v.to_string().into_bytes());
        }
    }
}
