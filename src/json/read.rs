//! A cursor over a JSON document. Each response parser walks the keys it
//! needs and skips the rest, so added fields never break a parse.

use std::borrow::Cow;

use super::{base64, float};

/// Where and why a parse stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Fail {
    pub(crate) offset: usize,
    pub(crate) what: &'static str,
}

pub(crate) type R<T> = Result<T, Fail>;

const MAX_DEPTH: u32 = 64;

pub(crate) struct Cursor<'a> {
    s: &'a [u8],
    pos: usize,
    // length of the last base64 vector; the next one usually ends there too
    base64_len: usize,
}

impl<'a> Cursor<'a> {
    pub(crate) fn new(s: &'a [u8]) -> Self {
        Self {
            s,
            pos: 0,
            base64_len: 0,
        }
    }

    pub(crate) fn fail<T>(&self, what: &'static str) -> R<T> {
        Err(Fail {
            offset: self.pos,
            what,
        })
    }

    fn ws(&mut self) {
        while let Some(b' ' | b'\n' | b'\r' | b'\t') = self.s.get(self.pos) {
            self.pos += 1;
        }
    }

    /// Next non-whitespace byte, not consumed.
    pub(crate) fn peek(&mut self) -> Option<u8> {
        self.ws();
        self.s.get(self.pos).copied()
    }

    fn eat(&mut self, b: u8) -> bool {
        if self.peek() == Some(b) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, b: u8, what: &'static str) -> R<()> {
        if self.eat(b) { Ok(()) } else { self.fail(what) }
    }

    /// Only whitespace may follow the document.
    pub(crate) fn end(&mut self) -> R<()> {
        match self.peek() {
            None => Ok(()),
            Some(_) => self.fail("trailing characters after the document"),
        }
    }

    /// `null` is consumed and reported as true.
    pub(crate) fn null(&mut self) -> R<bool> {
        if self.peek() != Some(b'n') {
            return Ok(false);
        }
        self.literal(b"null")?;
        Ok(true)
    }

    fn literal(&mut self, lit: &'static [u8]) -> R<()> {
        if self.s[self.pos..].starts_with(lit) {
            self.pos += lit.len();
            Ok(())
        } else {
            self.fail("invalid literal")
        }
    }

    /// Calls `f` with each key, the cursor placed on its value; `f` must
    /// consume the value.
    pub(crate) fn object(&mut self, mut f: impl FnMut(&mut Self, &str) -> R<()>) -> R<()> {
        self.expect(b'{', "expected an object")?;
        if self.eat(b'}') {
            return Ok(());
        }
        loop {
            if self.peek() != Some(b'"') {
                return self.fail("expected a key");
            }
            let key = self.str()?;
            self.expect(b':', "expected ':'")?;
            f(self, &key)?;
            if self.eat(b',') {
                continue;
            }
            return self.expect(b'}', "expected ',' or '}'");
        }
    }

    /// Calls `f` for each element, the cursor placed on it; `f` must consume it.
    pub(crate) fn array(&mut self, mut f: impl FnMut(&mut Self) -> R<()>) -> R<()> {
        self.expect(b'[', "expected an array")?;
        if self.eat(b']') {
            return Ok(());
        }
        loop {
            f(self)?;
            if self.eat(b',') {
                continue;
            }
            return self.expect(b']', "expected ',' or ']'");
        }
    }

    /// The raw bytes between the quotes of a string, and whether it holds escapes.
    fn str_raw(&mut self) -> R<(&'a [u8], bool)> {
        self.expect(b'"', "expected a string")?;
        let start = self.pos;
        let mut escaped = false;
        loop {
            match self.s.get(self.pos) {
                None => return self.fail("unterminated string"),
                Some(b'"') => break,
                Some(b'\\') => {
                    escaped = true;
                    self.pos += 2;
                }
                Some(&c) if c < 0x20 => return self.fail("control character in string"),
                Some(_) => self.pos += 1,
            }
        }
        if self.pos > self.s.len() {
            return self.fail("unterminated string");
        }
        let raw = &self.s[start..self.pos];
        self.pos += 1;
        Ok((raw, escaped))
    }

    pub(crate) fn str(&mut self) -> R<Cow<'a, str>> {
        let start = self.pos;
        let (raw, escaped) = self.str_raw()?;
        let bad = |what| Fail {
            offset: start,
            what,
        };
        if !escaped {
            return std::str::from_utf8(raw)
                .map(Cow::Borrowed)
                .map_err(|_| bad("string is not utf-8"));
        }
        unescape(raw)
            .map(Cow::Owned)
            .ok_or(bad("invalid escape in string"))
    }

    pub(crate) fn u64(&mut self) -> R<u64> {
        self.ws();
        let start = self.pos;
        let mut v: u64 = 0;
        while let Some(&c @ b'0'..=b'9') = self.s.get(self.pos) {
            v = match v
                .checked_mul(10)
                .and_then(|v| v.checked_add((c - b'0') as u64))
            {
                Some(v) => v,
                None => return self.fail("integer out of range"),
            };
            self.pos += 1;
        }
        let digits = self.pos - start;
        if digits == 0 || (digits > 1 && self.s[start] == b'0') {
            return self.fail("expected a non-negative integer");
        }
        if let Some(b'.' | b'e' | b'E') = self.s.get(self.pos) {
            return self.fail("expected an integer");
        }
        Ok(v)
    }

    /// A count: a non-negative integer, also written with a zero fraction
    /// (`2.0`), as providers that type counts as JSON numbers may send.
    pub(crate) fn count(&mut self) -> R<u64> {
        self.ws();
        let start = self.pos;
        let mut end = start;
        while let Some(b'0'..=b'9') = self.s.get(end) {
            end += 1;
        }
        if self.s.get(end) != Some(&b'.') {
            return self.u64();
        }
        let mut frac = end + 1;
        while let Some(b'0') = self.s.get(frac) {
            frac += 1;
        }
        match self.s.get(frac) {
            _ if frac == end + 1 => self.fail("expected digits after the decimal point"),
            Some(b'1'..=b'9' | b'e' | b'E') => self.fail("expected an integer"),
            _ => {
                let mut whole = Cursor::new(&self.s[..end]);
                whole.pos = start;
                let v = whole.u64()?;
                self.pos = frac;
                Ok(v)
            }
        }
    }

    /// One vector, as a JSON array of numbers or a base64 string of
    /// little-endian `f32`, appended to `out`. Returns its length.
    pub(crate) fn vector(&mut self, out: &mut Vec<f32>) -> R<usize> {
        match self.peek() {
            Some(b'"') => {
                let start = self.pos + 1;
                let before = out.len();
                let end = start + self.base64_len;
                // a quote or escape inside the guessed range fails to decode
                if self.base64_len > 0 && self.s.get(end) == Some(&b'"') {
                    if let Ok(n) = base64::decode_f32_le(&self.s[start..end], out) {
                        self.pos = end + 1;
                        return Ok(n);
                    }
                    out.truncate(before);
                }
                let (raw, escaped) = self.str_raw()?;
                let fail = |what| Fail {
                    offset: start,
                    what,
                };
                self.base64_len = raw.len();
                // some encoders write '/' as "\/"
                let plain;
                let raw = if escaped {
                    plain = unescape(raw).ok_or(fail("invalid escape in string"))?;
                    plain.as_bytes()
                } else {
                    raw
                };
                base64::decode_f32_le(raw, out).map_err(fail)
            }
            Some(b'[') => {
                let before = out.len();
                self.pos += 1;
                if self.eat(b']') {
                    return Ok(0);
                }
                loop {
                    self.ws();
                    match float::parse_f32(self.s, &mut self.pos) {
                        Some(v) => out.push(v),
                        None => return self.fail("expected a number"),
                    }
                    match self.s.get(self.pos) {
                        Some(b',') => self.pos += 1,
                        Some(b']') => {
                            self.pos += 1;
                            return Ok(out.len() - before);
                        }
                        _ => {
                            if self.eat(b',') {
                                continue;
                            }
                            self.expect(b']', "expected ',' or ']'")?;
                            return Ok(out.len() - before);
                        }
                    }
                }
            }
            _ => self.fail("expected an array of numbers or a base64 string"),
        }
    }

    /// Skip any value.
    pub(crate) fn skip(&mut self) -> R<()> {
        self.skip_at(0)
    }

    fn skip_at(&mut self, depth: u32) -> R<()> {
        if depth > MAX_DEPTH {
            return self.fail("nesting too deep");
        }
        match self.peek() {
            Some(b'{') => self.object(|c, _| c.skip_at(depth + 1)),
            Some(b'[') => self.array(|c| c.skip_at(depth + 1)),
            Some(b'"') => self.str_raw().map(drop),
            Some(b't') => self.literal(b"true"),
            Some(b'f') => self.literal(b"false"),
            Some(b'n') => self.literal(b"null"),
            Some(b'-' | b'0'..=b'9') => match float::parse_f32(self.s, &mut self.pos) {
                Some(_) => Ok(()),
                None => self.fail("invalid number"),
            },
            _ => self.fail("expected a value"),
        }
    }
}

fn hex4(s: &[u8]) -> Option<u32> {
    let mut v = 0;
    for &c in s.get(..4)? {
        v = v << 4 | (c as char).to_digit(16)?;
    }
    Some(v)
}

fn unescape(raw: &[u8]) -> Option<String> {
    let mut out = Vec::with_capacity(raw.len());
    let mut i = 0;
    while i < raw.len() {
        let c = raw[i];
        if c != b'\\' {
            out.push(c);
            i += 1;
            continue;
        }
        let e = *raw.get(i + 1)?;
        i += 2;
        let simple = match e {
            b'"' => b'"',
            b'\\' => b'\\',
            b'/' => b'/',
            b'b' => 0x08,
            b'f' => 0x0c,
            b'n' => b'\n',
            b'r' => b'\r',
            b't' => b'\t',
            b'u' => 0,
            _ => return None,
        };
        if e != b'u' {
            out.push(simple);
            continue;
        }
        let hi = hex4(&raw[i..])?;
        i += 4;
        let code = match hi {
            0xd800..=0xdbff => {
                if raw.get(i..i + 2)? != b"\\u" {
                    return None;
                }
                let lo = hex4(&raw[i + 2..])?;
                if !(0xdc00..=0xdfff).contains(&lo) {
                    return None;
                }
                i += 6;
                0x10000 + ((hi - 0xd800) << 10) + (lo - 0xdc00)
            }
            0xdc00..=0xdfff => return None,
            _ => hi,
        };
        let mut buf = [0u8; 4];
        out.extend_from_slice(char::from_u32(code)?.encode_utf8(&mut buf).as_bytes());
    }
    String::from_utf8(out).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(json: &str) -> R<String> {
        let mut c = Cursor::new(json.as_bytes());
        let v = c.str()?.into_owned();
        c.end()?;
        Ok(v)
    }

    #[test]
    fn strings_match_serde_json() {
        for json in [
            r#""plain""#,
            r#""esc \" \\ \/ \b \f \n \r \t""#,
            r#""Aé中😀""#,
            "\"raw utf-8 \u{4e2d}\u{1f600}\"",
            r#""""#,
        ] {
            assert_eq!(
                s(json).unwrap(),
                serde_json::from_str::<String>(json).unwrap()
            );
        }
        for bad in [
            r#""\x""#,
            r#""\ud83d""#,
            r#""\ude00""#,
            r#""\ud83dA""#,
            "\"tab\there\"",
            r#""unterminated"#,
            r#""\"#,
            r#""\u12""#,
        ] {
            assert!(s(bad).is_err(), "{bad}");
            assert!(serde_json::from_str::<String>(bad).is_err(), "{bad}");
        }
        assert!(Cursor::new(b"\"\xff\"").str().is_err());
    }

    #[test]
    fn skip_any_value() {
        let json = br#"{"a":[1,-2.5e3,true,false,null,{"b":"c\"d"}],"e":{}, "f" : [ ] }"#;
        let mut c = Cursor::new(json);
        c.skip().unwrap();
        c.end().unwrap();
        let deep = "[".repeat(100) + &"]".repeat(100);
        assert!(Cursor::new(deep.as_bytes()).skip().is_err());
        for bad in ["[1,]", "{\"a\"}", "{\"a\":1,}", "tru", "[1 2]", "01", "-"] {
            let mut c = Cursor::new(bad.as_bytes());
            assert!(c.skip().and_then(|_| c.end()).is_err(), "{bad}");
        }
    }

    #[test]
    fn integers() {
        assert_eq!(Cursor::new(b" 42").u64().unwrap(), 42);
        assert_eq!(Cursor::new(b"0").u64().unwrap(), 0);
        for bad in ["-1", "01", "1.0", "1e3", "x", "18446744073709551616"] {
            assert!(Cursor::new(bad.as_bytes()).u64().is_err(), "{bad}");
        }
    }

    #[test]
    fn counts() {
        for (ok, want) in [("42", 42), (" 2.0", 2), ("0.000", 0), ("18.0", 18)] {
            assert_eq!(Cursor::new(ok.as_bytes()).count().unwrap(), want, "{ok}");
        }
        let mut c = Cursor::new(b"[3.0,4]");
        let mut got = vec![];
        c.array(|c| {
            got.push(c.count()?);
            Ok(())
        })
        .unwrap();
        assert_eq!(got, [3, 4]);
        for bad in [
            "-1", "-1.0", "01.0", "1.", "1.5", "1.01", "1.0e3", "1e3", ".0", "x",
        ] {
            assert!(Cursor::new(bad.as_bytes()).count().is_err(), "{bad}");
        }
    }

    #[test]
    fn vectors_as_numbers_or_base64() {
        let mut out = Vec::new();
        let mut c = Cursor::new(b"[ 0.5 , -1e-3,2 ]");
        assert_eq!(c.vector(&mut out).unwrap(), 3);
        assert_eq!(out, vec![0.5, -1e-3, 2.0]);
        let mut c = Cursor::new(b"\"AACAPw==\"");
        assert_eq!(c.vector(&mut out).unwrap(), 1);
        assert_eq!(out[3], 1.0);
        assert_eq!(Cursor::new(b"[]").vector(&mut out).unwrap(), 0);
        for bad in ["[1,]", "[1 2]", "[\"a\"]", "{}", "\"AA\\u0041\"", "[1,2"] {
            assert!(
                Cursor::new(bad.as_bytes()).vector(&mut out).is_err(),
                "{bad}"
            );
        }
    }
}
