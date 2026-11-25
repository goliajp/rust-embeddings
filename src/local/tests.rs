use super::*;
use crate::rt::block_on;

fn plan(model: &str, dimensions: Option<u32>) -> Plan {
    Plan {
        model: model.into(),
        dimensions,
        input_type: None,
        truncate: Truncate::Provider,
        backoff: None,
    }
}

fn client() -> crate::Client {
    let handle = Handle {
        info: &models()[0],
        model: Arc::new(OnceLock::new()),
    };
    crate::Client::with_backend(crate::client::Backend::Local(handle), None)
}

fn handle(c: &crate::Client) -> &Handle {
    match &c.backend {
        crate::client::Backend::Local(h) => h,
        _ => unreachable!(),
    }
}

#[test]
fn foreign_model_is_rejected_before_loading() {
    let c = client();
    let texts = Inputs::from(["hi"]);
    let err = block_on(embed(
        &c,
        handle(&c),
        &plan("text-embedding-3-large", None),
        &texts,
    ))
    .unwrap_err();
    assert!(matches!(err, Error::UnknownModel(ref m) if m == "text-embedding-3-large"));
    assert!(handle(&c).model.get().is_none());
}

#[test]
fn unsupported_dimensions_are_rejected_before_loading() {
    let c = client();
    let texts = Inputs::from(["hi"]);
    let err = block_on(embed(
        &c,
        handle(&c),
        &plan("all-MiniLM-L6-v2", Some(64)),
        &texts,
    ))
    .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("384") && msg.contains("dimensions(64)"),
        "{msg}"
    );
    assert!(handle(&c).model.get().is_none());
}

#[test]
fn native_dimensions_and_empty_input_need_no_model() {
    let c = client();
    let empty: Vec<String> = vec![];
    let r = block_on(embed(
        &c,
        handle(&c),
        &plan("all-MiniLM-L6-v2", Some(384)),
        &Inputs::from(empty),
    ))
    .unwrap();
    assert!(r.is_empty());
    assert_eq!(r.usage.total_tokens, Some(0));
    assert_eq!(r.model, "all-MiniLM-L6-v2");
}

#[test]
fn without_a_transport_nothing_is_downloaded() {
    let c = client();
    let e = block_on(embed(
        &c,
        handle(&c),
        &plan("gte-small", None),
        &Inputs::from(["hi"]),
    ))
    .unwrap_err();
    assert!(matches!(e, Error::Unsupported { .. }), "{e:?}");
}

#[test]
fn model_errors_get_the_model_name() {
    let e = named(fail("boom".into()), "gte-small");
    assert_eq!(e.to_string(), "local model gte-small: boom");
    assert!(matches!(
        LocalModel::from_files("nope", Path::new("/nonexistent")),
        Err(Error::UnknownModel(_))
    ));
    let e = LocalModel::from_files("gte-small", Path::new("/nonexistent/model.safetensors"))
        .unwrap_err();
    assert!(e.to_string().starts_with("local model gte-small:"), "{e}");
}
