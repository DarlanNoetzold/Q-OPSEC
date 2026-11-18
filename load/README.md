# Load test — async pipeline

Plan: `qopsec_pipeline_async.jmx` (JMeter 5.5). It posts to `/run-pipeline/async` with a unique
`Idempotency-Key` per request and asserts HTTP 202.

```bash
jmeter -n -t load/qopsec_pipeline_async.jmx \
  -Jhost=127.0.0.1 -Jport=8081 \
  -Jthreads=50 -Jramp=30 -Jloops=20 \
  -l load/results.jtl -e -o load/report
```

Parameters: `host`, `port`, `protocol`, `threads`, `ramp`, `loops`.

Read the result from `load/report/index.html` (throughput, p95, error rate) and from
`GET /pipeline/jobs/stats` on the orchestrator (queued/running/succeeded/failed) while the run is active.

Baseline to watch:

- error rate must stay at 0 with 429 only when the rate limit is intentionally exceeded;
- `qopsec_pipeline_jobs_queued` should drain back to 0 after the ramp-down;
- `qopsec_circuit_breakers_open` must stay at 0; a non-zero value means an upstream is failing.

To test the rate limit on purpose, run with `-Jthreads=200 -Jramp=1` and
`QOPSEC_RATE_LIMIT_REQUESTS=600` (default): the excess must come back as 429 with `X-RateLimit-*` headers.
