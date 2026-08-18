//! Bounded retry for transient provider failures.
//!
//! Cloud endpoints routinely flake with 429 / 5xx / dropped connections, but
//! a retry is pure waste for a 4xx contract or billing error (400, 401, 402
//! "out of credits", 403, 404): the server will refuse again. So we retry
//! only transient statuses with a short linear backoff and fail fast on
//! everything else.

use crate::{ProviderError, Result};
use std::time::Duration;

/// How many attempts (including the first) before giving up on a transient
/// failure.
const MAX_ATTEMPTS: u32 = 3;

/// Base delay before the first retry; multiplied by the attempt number.
const BASE_BACKOFF: Duration = Duration::from_millis(500);

/// Whether `err` is worth re-attempting, and the base delay to use.
fn retry_delay(err: &ProviderError) -> Option<Duration> {
    match err {
        // Dropped connection / DNS / TLS glitch — almost always transient.
        ProviderError::Transport(_) => Some(BASE_BACKOFF),
        ProviderError::Http { status, .. } => match status {
            // 408/425/429 request timing / rate limiting — wait, then retry.
            408 | 425 | 429 => Some(BASE_BACKOFF),
            // 5xx = the provider's problem, likely fixed in a moment.
            500..=599 => Some(BASE_BACKOFF),
            // All other 4xx (400/401/402/403/404/...): no retry.
            _ => None,
        },
        _ => None,
    }
}

/// Run `attempt` up to [`MAX_ATTEMPTS`] times, retrying only transient
/// failures (network errors, 429/408/5xx) with linear backoff. Hard 4xx
/// errors — e.g. 402 out-of-credits — surface immediately.
pub(crate) async fn with_retry<F, Fut, T>(mut attempt: F) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    let mut attempt_no = 0u32;
    loop {
        match attempt().await {
            Ok(value) => return Ok(value),
            Err(err) if attempt_no + 1 < MAX_ATTEMPTS => {
                let Some(base) = retry_delay(&err) else {
                    return Err(err);
                };
                tokio::time::sleep(base.saturating_mul(attempt_no + 1)).await;
                attempt_no += 1;
            }
            Err(err) => return Err(err),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[tokio::test]
    async fn retries_transient_until_success() {
        let calls = Cell::new(0);
        let out = with_retry(|| async {
            calls.set(calls.get() + 1);
            if calls.get() < 3 {
                Err(ProviderError::Http {
                    status: 500,
                    message: "boom".into(),
                })
            } else {
                Ok(42u32)
            }
        })
        .await
        .unwrap();
        assert_eq!(out, 42);
        assert_eq!(calls.get(), 3);
    }

    #[tokio::test]
    async fn fails_fast_on_hard_4xx() {
        let calls = Cell::new(0);
        let err = with_retry(|| async {
            calls.set(calls.get() + 1);
            Err::<(), _>(ProviderError::Http {
                status: 402,
                message: "out of credits".into(),
            })
        })
        .await
        .unwrap_err();
        assert!(matches!(err, ProviderError::Http { status: 402, .. }));
        assert_eq!(calls.get(), 1, "402 must not be retried");
    }

    #[tokio::test]
    async fn retries_transport_errors() {
        let calls = Cell::new(0);
        let out = with_retry(|| async {
            calls.set(calls.get() + 1);
            if calls.get() == 1 {
                Err(ProviderError::Transport("connection reset".into()))
            } else {
                Ok(true)
            }
        })
        .await
        .unwrap();
        assert!(out);
        assert_eq!(calls.get(), 2);
    }

    #[tokio::test]
    async fn gives_up_after_max_attempts() {
        let calls = Cell::new(0);
        let err = with_retry(|| async {
            calls.set(calls.get() + 1);
            Err::<(), _>(ProviderError::Http {
                status: 503,
                message: "down".into(),
            })
        })
        .await
        .unwrap_err();
        assert!(matches!(err, ProviderError::Http { status: 503, .. }));
        assert_eq!(calls.get(), MAX_ATTEMPTS as usize);
    }
}
