#![allow(dead_code, reason = "each test binary uses a different part")]

use std::sync::Arc;
use std::time::Duration;

use embedrs::transport::{BoxFuture, HttpRequest, HttpResponse, Transport, TransportError};

/// A transport for tests that only build requests.
pub struct Unused;

impl Transport for Unused {
    fn send(&self, _: HttpRequest) -> BoxFuture<'_, Result<HttpResponse, TransportError>> {
        panic!("this test sends nothing")
    }
    fn sleep(&self, _: Duration) -> BoxFuture<'static, ()> {
        Box::pin(std::future::pending())
    }
}

pub fn unused() -> Arc<dyn Transport> {
    Arc::new(Unused)
}

pub fn ok(body: &str) -> HttpResponse {
    HttpResponse::new(200, vec![], body.as_bytes().to_vec())
}
