//! Correctly rounded decimal to `f32` for JSON numbers, in one pass over the
//! bytes. Results are bit-identical to `str::parse::<f32>`.

use super::pow5::{LARGEST_POWER_OF_FIVE, POWER_OF_FIVE_128, SMALLEST_POWER_OF_FIVE};

const POW10_F64: [f64; 23] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16,
    1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
];

const SIG_BITS: i32 = 23;
const INFINITE_POWER: i32 = 0xFF;
const EXP_MIN: i32 = -126;
const MIN_EXPONENT_ROUND_TO_EVEN: i64 = -17;
const MAX_EXPONENT_ROUND_TO_EVEN: i64 = 10;

/// Parses one JSON number (`-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?`)
/// at `s[*pos]` and advances `*pos` past it. `None` when the bytes there are
/// not a JSON number; a leading zero followed by a digit (`01`) is rejected.
pub(crate) fn parse_f32(s: &[u8], pos: &mut usize) -> Option<f32> {
    let start = *pos;
    let mut i = start;
    let neg = s.get(i) == Some(&b'-');
    i += neg as usize;

    let digits_start = i;
    let mut w = 0u64;
    match s.get(i) {
        Some(b'0') => {
            i += 1;
            if matches!(s.get(i), Some(b'0'..=b'9')) {
                return None;
            }
        }
        Some(b'1'..=b'9') => i = digits(s, i, &mut w),
        _ => return None,
    }
    let mut n_frac = 0usize;
    if s.get(i) == Some(&b'.') {
        i += 1;
        let frac_start = i;
        i = digits(s, i, &mut w);
        n_frac = i - frac_start;
        if n_frac == 0 {
            return None;
        }
    }
    let digits_end = i;
    let mut exp = 0i64;
    if let Some(b'e' | b'E') = s.get(i) {
        i += 1;
        let exp_neg = match s.get(i) {
            Some(b'-') => {
                i += 1;
                true
            }
            Some(b'+') => {
                i += 1;
                false
            }
            _ => false,
        };
        let exp_start = i;
        while let Some(&c) = s.get(i) {
            let d = c.wrapping_sub(b'0');
            if d > 9 {
                break;
            }
            exp = exp * 10 + d as i64;
            i += 1;
            if exp >= 0x10000 {
                // far outside f32's range; std decides the remaining digits
                return finish_slow(s, start, i, pos);
            }
        }
        if i == exp_start {
            return None;
        }
        if exp_neg {
            exp = -exp;
        }
    }
    *pos = i;

    let mut q = exp - n_frac as i64;
    let mut many_digits = false;
    if digits_end - digits_start - (n_frac != 0) as usize > 19 {
        (w, q, many_digits) = truncate_19(&s[digits_start..digits_end], q);
    }
    if w == 0 {
        return Some(if neg { -0.0 } else { 0.0 });
    }

    if !many_digits && w <= 1 << 53 && (-22..=22).contains(&q) {
        // w and 10^|q| are exact f64s, so r is the correctly rounded f64;
        // rounding r to f32 is then correct unless r sits exactly on an f32
        // midpoint (low 29 mantissa bits 1000..0); r is always an f32 normal
        let r = if q < 0 {
            w as f64 / POW10_F64[(-q) as usize]
        } else {
            w as f64 * POW10_F64[q as usize]
        };
        if r.to_bits() & 0x1FFF_FFFF != 0x1000_0000 {
            let f = r as f32;
            return Some(if neg { -f } else { f });
        }
    }

    let mut bits = eisel_lemire(q, w);
    if many_digits && bits.is_some() && bits != eisel_lemire(q, w + 1) {
        bits = None;
    }
    match bits {
        Some(b) => Some(f32::from_bits(b | (neg as u32) << 31)),
        None => std_parse(&s[start..i]),
    }
}

// reads ascii digits from s[i..] into w (wrapping), returns the end index
#[inline(always)]
fn digits(s: &[u8], mut i: usize, w: &mut u64) -> usize {
    while let Some(chunk) = s.get(i..i + 8) {
        let v = u64::from_le_bytes(chunk.try_into().expect("8-byte slice"));
        if !is_8_digits(v) {
            break;
        }
        *w = w.wrapping_mul(100_000_000).wrapping_add(parse_8_digits(v));
        i += 8;
    }
    while let Some(&c) = s.get(i) {
        let d = c.wrapping_sub(b'0');
        if d > 9 {
            break;
        }
        *w = w.wrapping_mul(10).wrapping_add(d as u64);
        i += 1;
    }
    i
}

#[inline(always)]
fn is_8_digits(v: u64) -> bool {
    let a = v & 0xF0F0_F0F0_F0F0_F0F0;
    let b = (v.wrapping_add(0x0606_0606_0606_0606) & 0xF0F0_F0F0_F0F0_F0F0) >> 4;
    a | b == 0x3333_3333_3333_3333
}

// eight ascii digits, first digit in the lowest byte
#[inline(always)]
fn parse_8_digits(mut v: u64) -> u64 {
    const MASK: u64 = 0x0000_00FF_0000_00FF;
    const MUL1: u64 = 0x000F_4240_0000_0064; // 100 + (1000000 << 32)
    const MUL2: u64 = 0x0000_2710_0000_0001; // 1 + (10000 << 32)
    v = v.wrapping_sub(0x3030_3030_3030_3030);
    v = v.wrapping_mul(10).wrapping_add(v >> 8);
    let lo = (v & MASK).wrapping_mul(MUL1);
    let hi = ((v >> 16) & MASK).wrapping_mul(MUL2);
    lo.wrapping_add(hi) >> 32
}

// the first 19 significant digits of `digits` (ascii digits and at most one
// '.') with the exponent adjusted for the dropped ones; the flag says whether
// any digit was dropped
#[cold]
fn truncate_19(digits: &[u8], q: i64) -> (u64, i64, bool) {
    let mut w = 0u64;
    let mut taken = 0u32;
    let mut dropped = 0i64;
    for &c in digits.iter().filter(|&&c| c != b'.') {
        if taken == 19 {
            dropped += 1;
        } else if w != 0 || c != b'0' {
            w = w * 10 + (c - b'0') as u64;
            taken += 1;
        }
    }
    (w, q + dropped, dropped != 0)
}

#[cold]
#[inline(never)]
fn finish_slow(s: &[u8], start: usize, mut i: usize, pos: &mut usize) -> Option<f32> {
    while matches!(s.get(i), Some(b'0'..=b'9')) {
        i += 1;
    }
    *pos = i;
    std_parse(&s[start..i])
}

// json number syntax is a subset of what `f32::from_str` accepts
#[cold]
#[inline(never)]
fn std_parse(number: &[u8]) -> Option<f32> {
    core::str::from_utf8(number).ok()?.parse().ok()
}

// Eisel-Lemire for f32: the biased exponent and mantissa as f32 bits without
// the sign, or `None` when the truncated product cannot decide the rounding
#[inline]
fn eisel_lemire(q: i64, mut w: u64) -> Option<u32> {
    if q < SMALLEST_POWER_OF_FIVE as i64 {
        return Some(0);
    }
    if q > LARGEST_POWER_OF_FIVE as i64 {
        return Some((INFINITE_POWER as u32) << SIG_BITS);
    }
    let lz = w.leading_zeros();
    w <<= lz;
    let (lo, hi) = product_approx(q, w);
    if lo == u64::MAX && !(-27..=55).contains(&q) {
        return None;
    }
    let upperbit = (hi >> 63) as i32;
    let shift = upperbit + 64 - SIG_BITS - 3;
    let mut mantissa = hi >> shift;
    let mut power2 = power(q as i32) + upperbit - lz as i32 - EXP_MIN + 1;
    if power2 <= 0 {
        if -power2 + 1 >= 64 {
            return Some(0);
        }
        mantissa >>= -power2 + 1;
        mantissa += mantissa & 1;
        mantissa >>= 1;
        // a carry into bit 23 turns the subnormal into the smallest normal
        return Some(mantissa as u32);
    }
    if lo <= 1
        && (MIN_EXPONENT_ROUND_TO_EVEN..=MAX_EXPONENT_ROUND_TO_EVEN).contains(&q)
        && mantissa & 3 == 1
        && (mantissa << shift) == hi
    {
        mantissa &= !1;
    }
    mantissa += mantissa & 1;
    mantissa >>= 1;
    if mantissa >= 2 << SIG_BITS {
        mantissa = 1 << SIG_BITS;
        power2 += 1;
    }
    mantissa &= !(1 << SIG_BITS);
    if power2 >= INFINITE_POWER {
        return Some((INFINITE_POWER as u32) << SIG_BITS);
    }
    Some(mantissa as u32 | (power2 as u32) << SIG_BITS)
}

// floor(log2(10^q)) + 63
#[inline(always)]
fn power(q: i32) -> i32 {
    (q.wrapping_mul(152_170 + 65536) >> 16) + 63
}

// high 128 bits of w * 5^q, with the second multiply only when the bits
// below the f32 mantissa plus three guard bits are all ones
#[inline(always)]
fn product_approx(q: i64, w: u64) -> (u64, u64) {
    const MASK: u64 = u64::MAX >> (SIG_BITS + 3);
    let (hi5, lo5) = POWER_OF_FIVE_128[(q - SMALLEST_POWER_OF_FIVE as i64) as usize];
    let first = w as u128 * hi5 as u128;
    let mut first_lo = first as u64;
    let mut first_hi = (first >> 64) as u64;
    if first_hi & MASK == MASK {
        let second_hi = ((w as u128 * lo5 as u128) >> 64) as u64;
        first_lo = first_lo.wrapping_add(second_hi);
        if second_hi > first_lo {
            first_hi += 1;
        }
    }
    (first_lo, first_hi)
}

#[cfg(test)]
mod tests {
    use super::parse_f32;
    use proptest::prelude::*;

    fn ours(s: &str) -> Option<(u32, usize)> {
        let mut pos = 0;
        parse_f32(s.as_bytes(), &mut pos).map(|f| (f.to_bits(), pos))
    }

    fn assert_std(s: &str) {
        let expected = s.parse::<f32>().unwrap().to_bits();
        assert_eq!(ours(s), Some((expected, s.len())), "{s}");
    }

    // exact decimal expansion of an f64, which holds every f32 midpoint
    fn exact(x: f64) -> String {
        let s = format!("{x:.800e}");
        let (m, e) = s.split_once('e').unwrap();
        let m = m.trim_end_matches('0').trim_end_matches('.');
        format!("{m}e{e}")
    }

    #[test]
    fn edge_cases_match_std() {
        let cases = [
            "0",
            "-0",
            "0e0",
            "0.0",
            "-0.0e-0",
            "0e99999999999",
            "-0E+5",
            "1",
            "-1",
            "1.5",
            "1e10",
            "1E+10",
            "1e-10",
            "123456.789e-3",
            // min/max normal, min subnormal, max
            "1.17549435e-38",
            "1.1754942e-38",
            "1e-45",
            "1.401298464324817e-45",
            "3.4028235e38",
            "-3.4028235e38",
            "3.4028234663852886e38",
            // exact halfway cases: ties to even, and one digit either side
            "16777217",
            "16777219",
            "16777217.000000001",
            "16777216.999999999",
            "1.000000059604644775390625",
            "1.000000059604644775390626",
            "1.000000059604644775390624",
            "1.0000001788139343261718750",
            // overflow threshold f32::MAX + half ulp, and just below
            "340282356779733661637539395458142568448",
            "340282356779733661637539395458142568447",
            "3.40282356779733661637539395458142568448e38",
            // underflow threshold 2^-150 (ties to zero) and just above
            "7.00649232162408535461864791644958065640130970938257885878534141944895541342930300743319094181060791015625e-46",
            "7.00649232162408535461864791644958065640130970938257885878534141944895541342930300743319094181060791015626e-46",
            // f64 rounding lands exactly on an f32 midpoint
            "17.00392436981201",
            "2.642103131899219e20",
            "-6.450396305701556e35",
            // overflow and underflow
            "1e39",
            "-1e39",
            "3.5e38",
            "1e-46",
            "-1e-46",
            "1e-50",
            "1e-400",
            "1e400",
            // 19, 20 and 40+ digit significands
            "1234567890123456789",
            "12345678901234567890",
            "9999999999999999999",
            "10000000000000000000",
            "18446744073709551615",
            "18446744073709551616",
            "0.1234567890123456789012345678901234567890123",
            "1234567890123456789012345678901234567890e-30",
            "3.40282113390818417817494308091127136256e38",
            // leading zeros in the fraction
            "0.00000000000000000000000001",
            "0.000000000000000000000000000000000000000000001",
            "0.0000000000000000000000000000000000000000000014012984643248170709237295832899161312802619418765157717570682838897910826858606014866381883621215820312",
            // exponents with many digits
            "1e0000000000000000000000000000038",
            "1e-00000000000000000000045",
            "1e99999999999999999999",
            "1e-99999999999999999999",
            "-1e65535",
            "1e65536",
            // long fraction compensated by a large exponent
            "0.00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001e100",
        ];
        for s in cases {
            assert_std(s);
        }
        assert_std(&format!("0.{}1e1000", "0".repeat(999)));
        assert_std(&format!("1{}e-1000", "0".repeat(1000)));
    }

    #[test]
    fn stops_after_the_number() {
        for (s, n) in [
            ("1.5,2", 3),
            ("-0]", 2),
            ("12e3}", 4),
            ("7 ", 1),
            ("0.25x", 4),
            ("0x10", 1),
        ] {
            let mut pos = 0;
            let f = parse_f32(s.as_bytes(), &mut pos).unwrap();
            assert_eq!((f, pos), (s[..n].parse::<f32>().unwrap(), n), "{s}");
        }
        let mut pos = 4;
        assert_eq!(parse_f32(b"[1, 2.5]", &mut pos), Some(2.5));
        assert_eq!(pos, 7);
    }

    #[test]
    fn invalid_grammar_is_none() {
        let cases = [
            "", "01", "-01", "00", "1.", "1.e5", ".5", "-.5", "+1", "1e", "1e+", "1E-", "-", "--1",
            "NaN", "nan", "inf", "-inf", "Infinity", " 1", "e5", "-e5",
        ];
        for s in cases {
            assert_eq!(ours(s), None, "{s}");
        }
    }

    // 1..=40 significant digits, exponent -60..=50, json grammar
    fn decimal() -> impl Strategy<Value = String> {
        (
            any::<bool>(),
            1..=9u8,
            prop::collection::vec(0..=9u8, 0..40),
            any::<prop::sample::Index>(),
            0..4usize,
            prop::option::of((-60..=50i32, any::<bool>())),
        )
            .prop_map(|(neg, first, rest, point, zeros, exp)| {
                let mut ds = vec![b'0' + first];
                ds.extend(rest.iter().map(|d| b'0' + d));
                let ds = String::from_utf8(ds).unwrap();
                let k = point.index(ds.len() + 1);
                let mut s = String::from(if neg { "-" } else { "" });
                if k == 0 {
                    s += "0.";
                    s += &"0".repeat(zeros);
                    s += &ds;
                } else if k == ds.len() {
                    s += &ds;
                } else {
                    s += &ds[..k];
                    s += ".";
                    s += &ds[k..];
                }
                if let Some((e, upper)) = exp {
                    s += if upper { "E" } else { "e" };
                    s += &e.to_string();
                }
                s
            })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100_000))]

        #[test]
        fn random_decimals_match_std(s in decimal()) {
            let expected = s.parse::<f32>().unwrap().to_bits();
            prop_assert_eq!(ours(&s), Some((expected, s.len())), "{}", s);
        }

        #[test]
        fn f32_display_round_trips(sign in 0..2u32, exp in 0..255u32, mantissa in 0..1u32 << 23) {
            let bits = sign << 31 | exp << 23 | mantissa;
            let f = f32::from_bits(bits);
            for s in [format!("{f}"), format!("{f:e}"), format!("{}", f as f64), format!("{f:.8e}")] {
                prop_assert_eq!(ours(&s), Some((bits, s.len())), "{}", s);
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10_000))]

        #[test]
        fn halfway_cases_match_std(bits in 0..0x7f7f_ffffu32, side in 0..3usize) {
            let lo = f32::from_bits(bits) as f64;
            let hi = f32::from_bits(bits + 1) as f64;
            let mid = (lo + hi) / 2.0;
            assert_std(&exact([mid, mid.next_up(), mid.next_down()][side]));
            assert_std(&format!("{mid:e}"));
        }
    }
}
