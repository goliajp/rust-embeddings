use super::*;

/// A file in the temp directory, removed when dropped.
struct Temp(std::path::PathBuf);

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

impl std::ops::Deref for Temp {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.0
    }
}

fn temp(name: &str, bytes: &[u8]) -> Temp {
    let path = std::env::temp_dir().join(format!("embedrs-st-{}-{name}", std::process::id()));
    std::fs::write(&path, bytes).unwrap();
    Temp(path)
}

fn file(name: &str, header: &str, data: &[u8]) -> Temp {
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header.as_bytes());
    bytes.extend(data);
    temp(name, &bytes)
}

fn err(path: &Path) -> String {
    match Safetensors::open(path) {
        Ok(_) => panic!("{} opened", path.display()),
        Err(e) => e.to_string(),
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[test]
fn reads_f32_f16_bf16() {
    let mut data = f32_bytes(&[1.0, -2.5, 3.25, 0.0]);
    data.extend([0x00, 0x3c, 0x00, 0xc1]); // f16 1.0, -2.5
    data.extend([0x80, 0x3f, 0x20, 0xc0]); // bf16 1.0, -2.5
    let h = r#"{"__metadata__":{"format":"pt"},"a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},
        "h":{"dtype":"F16","shape":[2],"data_offsets":[16,20]},"b":{"dtype":"BF16","shape":[2],"data_offsets":[20,24]}}"#;
    // pad the header to a multiple of 4 so the f32 tensor is mapped in place
    let h = format!("{h}{}", " ".repeat((4 - h.len() % 4) % 4));
    let st = Safetensors::open(&file("ok.safetensors", &h, &data)).unwrap();
    let a = st.f32s("a", &[2, 2]).unwrap();
    assert!(matches!(a, F32s::Mapped { .. }));
    assert_eq!(&a[..], &[1.0, -2.5, 3.25, 0.0]);
    assert_eq!(&st.f32s("h", &[2]).unwrap()[..], &[1.0, -2.5]);
    assert_eq!(&st.f32s("b", &[2]).unwrap()[..], &[1.0, -2.5]);
    assert!(st.contains("a") && !st.contains("c"));
    let e = st.f32s("a", &[4]).err().unwrap().to_string();
    assert!(e.contains("shape [2, 2]"), "{e}");
    let e = st.f32s("c", &[1]).err().unwrap().to_string();
    assert!(e.contains("no tensor c"), "{e}");
}

#[test]
fn unaligned_f32_is_copied() {
    let h = r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[1,5]}}"#;
    let h = format!("{h}{}", " ".repeat((4 - h.len() % 4) % 4));
    let mut data = vec![0u8];
    data.extend(f32_bytes(&[7.5]));
    let st = Safetensors::open(&file("unaligned.safetensors", &h, &data)).unwrap();
    let a = st.f32s("a", &[1]).unwrap();
    assert!(matches!(a, F32s::Owned(_)));
    assert_eq!(&a[..], &[7.5]);
}

#[test]
fn rejects_broken_files() {
    assert!(err(&temp("short", &[1, 2, 3])).contains("shorter than 8 bytes"));
    let mut long = 1000u64.to_le_bytes().to_vec();
    long.extend(b"{}");
    assert!(err(&temp("long", &long)).contains("header length"));

    let cases = [
        ("json", r#"{"a":"#, "at byte"),
        (
            "outside",
            r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
            "outside the file",
        ),
        (
            "length",
            r#"{"a":{"dtype":"F32","shape":[3],"data_offsets":[0,4]}}"#,
            "does not match",
        ),
        (
            "offsets",
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0]}}"#,
            "two numbers",
        ),
        (
            "reversed",
            r#"{"a":{"dtype":"F32","shape":[0],"data_offsets":[4,0]}}"#,
            "outside the file",
        ),
    ];
    for (name, h, want) in cases {
        let e = err(&file(name, h, &[0; 4]));
        assert!(e.contains(want), "{name}: {e}");
    }
    let st = Safetensors::open(&file(
        "int",
        r#"{"a":{"dtype":"I64","shape":[1],"data_offsets":[0,8]}}"#,
        &[0; 8],
    ))
    .unwrap();
    assert!(
        st.f32s("a", &[1])
            .err()
            .unwrap()
            .to_string()
            .contains("not a float tensor")
    );
}

// every f16 bit pattern against its value computed in f64
#[test]
fn f16_conversion_is_exact() {
    for h in 0..=u16::MAX {
        let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
        let exp = ((h >> 10) & 0x1f) as i32;
        let man = (h & 0x3ff) as f64;
        let got = f16_to_f32(h);
        match exp {
            31 if man == 0.0 => assert_eq!(got, sign as f32 * f32::INFINITY),
            31 => assert!(got.is_nan()),
            0 => assert_eq!(got as f64, sign * man * 2f64.powi(-24), "{h:#06x}"),
            _ => assert_eq!(
                got as f64,
                sign * (1.0 + man / 1024.0) * 2f64.powi(exp - 15),
                "{h:#06x}"
            ),
        }
        if h == 0x8000 {
            assert!(got == 0.0 && got.is_sign_negative());
        }
    }
    assert_eq!(bf16_to_f32(0x3f80), 1.0);
    assert_eq!(bf16_to_f32(0xc020), -2.5);
}
