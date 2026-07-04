# Bug report: `gcloud compute start-iap-tunnel` hangs (`Bad file descriptor`) with OpenSSL 3.5/3.6

> Ready to file at Google's public Issue Tracker (Cloud SDK component) or via
> `gcloud feedback`. Everything below was reproduced and verified on 2026-06-19.

## Title

`gcloud compute start-iap-tunnel` hangs at "Testing if tunnel connection works." —
unhandled `OSError: [Errno 9] Bad file descriptor` in
`iap_tunnel_lightweight_websocket.py` when the client links OpenSSL 3.5/3.6

## Summary

`gcloud compute start-iap-tunnel` never finishes establishing: it prints
`Testing if tunnel connection works.` and hangs forever, never reaching
`Listening on port [...]`. With `--verbosity=debug`, the WebSocket to
`wss://tunnel.cloudproxy.app/v4/connect` connects and then immediately tears down
with `OSError: [Errno 9] Bad file descriptor` raised from
`iap_tunnel_lightweight_websocket.py`.

The IAP service, the instance, the firewall, IAM, and the network are all
correct — I proved the tunnel establishes server-side by performing the WebSocket
handshake manually (see "Proof"). The defect is entirely client-side in gcloud's
"lightweight websocket" and is exposed by the new OpenSSL 3.5/3.6 non-blocking
write semantics.

## Environment

- gcloud: **Google Cloud SDK 573.0.0** (also reproduced with 540.0.0 and 430.0.0)
- OS: **macOS 26.5** (build 25F71), Apple Silicon (arm64)
- Python / TLS (reproduced under both):
  - CPython 3.14.5 (gcloud Homebrew-cask venv) — **OpenSSL 3.6.2 (7 Apr 2026)**
  - CPython 3.12.13 (uv) — **OpenSSL 3.5.6 (7 Apr 2026)**
- Install method: Homebrew cask `google-cloud-sdk`
- NumPy installed in gcloud's Python (does not help)

## Reproduction

1. macOS, gcloud ≥ 430, a Python 3.12+ that links **OpenSSL ≥ 3.5**.
2. Create a Compute Engine instance (with or without an external IP) and a firewall
   rule allowing `35.235.240.0/20` to the target port.
3. Run:
   ```
   gcloud compute start-iap-tunnel INSTANCE 22 --local-host-port=localhost:2222 --zone ZONE --verbosity=debug
   ```
4. Observed: hangs at `Testing if tunnel connection works.`; debug log shows:
   ```
   INFO  Connecting with URL ['wss://tunnel.cloudproxy.app/v4/connect?...&port=22&...']
   DEBUG CLOSE
   INFO  Unable to send WebSocket Close message [[Errno 9] Bad file descriptor].
   INFO  Error while receiving from WebSocket.
   ...
   File ".../api_lib/compute/iap_tunnel_lightweight_websocket.py", line 343, in _wait_for_socket_to_ready
   OSError: [Errno 9] Bad file descriptor
   ```

Expected: tunnel reaches `Listening on port [2222] on localhost.` and forwards.

## Root cause (from the shipped source)

File: `lib/googlecloudsdk/api_lib/compute/iap_tunnel_lightweight_websocket.py`

1. **Unhandled `OSError` in `_wait_for_socket_to_ready`** (~line 337):
   ```python
   _ = select.select([self.sock], (), (), timeout)
   ```
   Only `TypeError` is caught. When the SSL socket's fd is closed by the other
   thread between the `fileno() != -1` guard and the `select`, `select` raises
   `OSError: [Errno 9] Bad file descriptor`, which propagates and kills the
   connection (then loops forever on reconnect → the hang).

2. **Identity-vs-isinstance bugs in `_throw_on_non_retriable_exception`** (~line 306):
   ```python
   if e is ssl.SSLError:          # always False; should be isinstance(e, ssl.SSLError)
       ...
   elif e is socket.error:        # always False; should be isinstance(e, (socket.error, OSError))
       ...
       if error_code != errno.EAGAIN or error_code != errno.EWOULDBLOCK:  # tautology; should be 'and' / 'not in'
           raise e
   ```
   Transient `SSL_ERROR_WANT_WRITE` / `EWOULDBLOCK` from a non-blocking SSL socket
   are therefore not handled as intended. OpenSSL 3.5/3.6 changed non-blocking
   write signaling relative to 1.1.1 / 3.0–3.2, which is why this only surfaces on
   very recent OpenSSL.

## Proof the server side is correct (not network/IAM/firewall)

A manual WebSocket handshake to the same endpoint, using gcloud's exact origin
(`bot:iap-tunneler`, from `iap_tunnel_websocket_helper.py:34`), subprotocol
(`relay.tunnel.cloudproxy.app`), and a valid bearer token, succeeds:

```
HTTP/1.1 101 Switching Protocols
Sec-WebSocket-Protocol: relay.tunnel.cloudproxy.app
<binary frame 0x82 ... 0x0001 ...>   # SUBPROTOCOL_TAG_CONNECT_SUCCESS_SID + session id
```

i.e. the IAP relay accepts the tunnel and holds it open; only gcloud's client
mishandles the established connection.

## Suggested fix

- `_wait_for_socket_to_ready`: guard `self.sock is None or self.sock.fileno() == -1`,
  and catch `(OSError, ValueError)` around `select.select`, raising
  `WebSocketConnectionClosedException` instead of propagating.
- `_throw_on_non_retriable_exception`: replace `is` with `isinstance`, include
  `OSError`, and fix the `!= ... or != ...` tautology to `not in (EAGAIN, EWOULDBLOCK)`.

## Workarounds

- Reach the instance over standard SSH (external IP) instead of IAP TCP forwarding.
- Run gcloud under a Python linked against OpenSSL ≤ 3.2.
