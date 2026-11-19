package tech.noetzold.context_api.resilience;

/** Upstream answered with a retryable failure (5xx, transport error or timeout). */
public class UpstreamUnavailableException extends RuntimeException {

    public UpstreamUnavailableException(String message) {
        super(message);
    }

    public UpstreamUnavailableException(String message, Throwable cause) {
        super(message, cause);
    }
}
