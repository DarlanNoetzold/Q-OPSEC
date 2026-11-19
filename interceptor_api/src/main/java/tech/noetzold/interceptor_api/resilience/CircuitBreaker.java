package tech.noetzold.interceptor_api.resilience;

import java.time.Duration;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Failure-threshold circuit breaker with a half-open probe window.
 *
 * Deliberately dependency-free so every service can use it without adding a new
 * artifact to its build. Thread-safe: all state transitions are synchronized.
 */
public final class CircuitBreaker {

    public enum State { CLOSED, OPEN, HALF_OPEN }

    private final String name;
    private final int failureThreshold;
    private final long resetTimeoutMillis;
    private final int halfOpenMaxCalls;

    private State state = State.CLOSED;
    private int failures;
    private int halfOpenCalls;
    private long openedAtMillis;
    private String lastError;

    private final AtomicInteger rejections = new AtomicInteger();
    private final AtomicInteger successes = new AtomicInteger();
    private final AtomicLong lastFailureAt = new AtomicLong();

    public CircuitBreaker(String name, int failureThreshold, Duration resetTimeout, int halfOpenMaxCalls) {
        if (failureThreshold < 1) {
            throw new IllegalArgumentException("failureThreshold must be positive");
        }
        if (halfOpenMaxCalls < 1) {
            throw new IllegalArgumentException("halfOpenMaxCalls must be positive");
        }
        this.name = name;
        this.failureThreshold = failureThreshold;
        this.resetTimeoutMillis = Math.max(1L, resetTimeout.toMillis());
        this.halfOpenMaxCalls = halfOpenMaxCalls;
    }

    public static CircuitBreaker of(String name) {
        return new CircuitBreaker(name, 5, Duration.ofSeconds(30), 1);
    }

    public synchronized void acquire() {
        if (state == State.OPEN) {
            if (System.currentTimeMillis() - openedAtMillis >= resetTimeoutMillis) {
                state = State.HALF_OPEN;
                halfOpenCalls = 0;
            } else {
                rejections.incrementAndGet();
                throw new CircuitOpenException(name, retryAfterMillis());
            }
        }
        if (state == State.HALF_OPEN) {
            if (halfOpenCalls >= halfOpenMaxCalls) {
                rejections.incrementAndGet();
                throw new CircuitOpenException(name, retryAfterMillis());
            }
            halfOpenCalls++;
        }
    }

    public synchronized void recordSuccess() {
        successes.incrementAndGet();
        failures = 0;
        state = State.CLOSED;
        openedAtMillis = 0L;
        halfOpenCalls = 0;
    }

    public synchronized void recordFailure(Throwable error) {
        failures++;
        lastFailureAt.set(System.currentTimeMillis());
        lastError = error == null ? null : (error.getClass().getSimpleName() + ": " + error.getMessage());
        if (state == State.HALF_OPEN || failures >= failureThreshold) {
            state = State.OPEN;
            openedAtMillis = System.currentTimeMillis();
            halfOpenCalls = 0;
        }
    }

    public synchronized State state() {
        return state;
    }

    public synchronized boolean isOpen() {
        return state == State.OPEN;
    }

    public synchronized long retryAfterMillis() {
        if (state != State.OPEN || openedAtMillis == 0L) {
            return 0L;
        }
        long elapsed = System.currentTimeMillis() - openedAtMillis;
        return Math.max(0L, resetTimeoutMillis - elapsed);
    }

    public synchronized String lastError() {
        return lastError;
    }

    public int rejections() {
        return rejections.get();
    }

    public int successes() {
        return successes.get();
    }

    public long lastFailureAt() {
        return lastFailureAt.get();
    }

    public String name() {
        return name;
    }

    public synchronized void reset() {
        state = State.CLOSED;
        failures = 0;
        openedAtMillis = 0L;
        halfOpenCalls = 0;
        lastError = null;
    }

    @Override
    public synchronized String toString() {
        return "CircuitBreaker{" + name + ", state=" + state + ", failures=" + failures
                + ", rejections=" + rejections.get() + ", successes=" + successes.get() + '}';
    }
}
