package tech.noetzold.context_api.resilience;

/** Raised when a call is rejected because the breaker is open. */
public class CircuitOpenException extends RuntimeException {

    private final String breakerName;
    private final long retryAfterMillis;

    public CircuitOpenException(String breakerName, long retryAfterMillis) {
        super("circuit '" + breakerName + "' is open; retry in " + retryAfterMillis + "ms");
        this.breakerName = breakerName;
        this.retryAfterMillis = retryAfterMillis;
    }

    public String breakerName() {
        return breakerName;
    }

    public long retryAfterMillis() {
        return retryAfterMillis;
    }
}
