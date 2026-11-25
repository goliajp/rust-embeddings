//! placeholder until the fast parser lands: grammar scan, then std's parser
pub(crate) fn parse_f32(s: &[u8], pos: &mut usize) -> Option<f32> {
    let start = *pos;
    let mut i = start;
    let digits = |i: &mut usize| {
        let b = *i;
        while let Some(b'0'..=b'9') = s.get(*i) {
            *i += 1;
        }
        *i - b
    };
    if s.get(i) == Some(&b'-') {
        i += 1;
    }
    let int_start = i;
    let n = digits(&mut i);
    if n == 0 || (n > 1 && s[int_start] == b'0') {
        return None;
    }
    if s.get(i) == Some(&b'.') {
        i += 1;
        if digits(&mut i) == 0 {
            return None;
        }
    }
    if let Some(b'e' | b'E') = s.get(i) {
        i += 1;
        if let Some(b'+' | b'-') = s.get(i) {
            i += 1;
        }
        if digits(&mut i) == 0 {
            return None;
        }
    }
    let v = std::str::from_utf8(&s[start..i]).ok()?.parse().ok()?;
    *pos = i;
    Some(v)
}
