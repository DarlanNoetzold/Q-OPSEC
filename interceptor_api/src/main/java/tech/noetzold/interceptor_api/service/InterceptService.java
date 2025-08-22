package tech.noetzold.interceptor_api.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import tech.noetzold.interceptor_api.client.ContextApiClient;

// service/InterceptService.java
@Service
public class InterceptService {
    @Autowired
    private ContextApiClient contextApiClient;

    public InterceptResponse intercept(InterceptRequest req) {
        ContextRequest ctxReq = new ContextRequest();
        ctxReq.setSourceId(req.getSourceId());
        ctxReq.setDestinationId(req.getDestinationId());
        ctxReq.setContent(req.getMessage());
        ctxReq.setMetadata(req.getMetadata());

        ContextResponse context = contextApiClient.getContext(ctxReq);

        InterceptResponse resp = new InterceptResponse();
        resp.setStatus("OK");
        resp.setContext(context);
        return resp;
    }
}
