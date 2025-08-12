package tech.noetzold.context_api.controller;

import org.slf4j.MDC;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestControllerAdvice;

import java.util.Map;
import java.util.concurrent.TimeoutException;

@RestControllerAdvice
public class ErrorHandler {

    @ExceptionHandler(TimeoutException.class)
    @ResponseStatus(HttpStatus.GATEWAY_TIMEOUT)
    public Map<String,Object> handleTimeout(TimeoutException ex) {
        return Map.of(
                "code","UPSTREAM_TIMEOUT",
                "message","Risk or Confidentiality service timed out",
                "retry_after_ms",200,
                "trace_id", MDC.get("trace_id")
        );
    }
}