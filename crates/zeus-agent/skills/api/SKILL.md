---
name: api
description: Design and build HTTP APIs — REST or RPC, validation, auth, error contracts, docs, and tests.
version: 1.0.0
tags: [api, rest, http, backend, openapi]
depends_on: [database]
---

# API Engineering

Build APIs that other code (your frontend, other services) can consume without
guessing.

## Design first
- List resources and their actions before writing routes.
- **REST conventions**: nouns for resources, `GET` for reads (no side
  effects), `POST` for creation with non-idempotency, `PUT`/`PATCH` for
  updates (PUT whole, PATCH partial), `DELETE` for removal.
- Return consistent, documented shapes: `{ data: ... }` or plain objects, and
  a stable error body (`{ "error": { "code", "message" } }`).

## Request handling
- **Validate every boundary input**: types, ranges, formats, max lengths.
  Fail fast with 400 and a readable message.
- Paginate list endpoints (`page`/`limit` or cursor). Cap `limit`.
- **Auth**: never invent crypto. Use the framework/session/JWT middleware.
  Protect every route with the right authn/authz check — verify each handler,
  not just the router.

## Error contract
- Map exceptions to correct status codes: 400 client error, 401 unauthn,
  403 forbidden, 404 missing, 409 conflict, 422 validation, 500 internal.
- Never leak stack traces, connection strings, or internals to clients in
  500s. Log the detail server-side only.

## Docs
- Keep OpenAPI/Swagger in sync with the routes (generate from code where
  possible). A dead doc is worse than none.

## Verification
- Test each route: happy path, validation failure, auth failure, missing
  resource. Use the codebase's existing test harness — do NOT add a framework.
- Verify with a live request (curl/httpie) when a server is reachable.