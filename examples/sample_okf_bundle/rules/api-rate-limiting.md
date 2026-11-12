---
id: "WIKI-RULE-002"
type: "rule"
title: "Distributed Redis Token Bucket Rate Limiting for Public APIs"
category: "api_security"
target_services: ["checkout-api", "auth-service", "gateway-api"]
target_languages: ["golang", "python", "typescript"]
approved_by: "api-arch@taryn.cloud.corp"
approved_at: "2026-08-20"
valid_until: "2028-01-01"
status: "APPROVED"
tags: ["rate-limit", "redis", "api-gateway", "ddos"]
---

# Architecture Rule: Public API Rate Limiting

1. Any public HTTP endpoint exposed without mTLS must implement token-bucket rate limiting backed by cluster Redis.
2. Default quota: 100 requests per minute per authenticated API key; 20 requests per minute per anonymous client IP.
3. Exceeded requests must respond with HTTP status `429 Too Many Requests` containing headers `Retry-After` and `X-RateLimit-Reset`.

# LLM Reviewer Directive:
When reviewing new endpoint handlers, verify that middleware `limiter.RequireRateLimit()` wraps the route registration.
