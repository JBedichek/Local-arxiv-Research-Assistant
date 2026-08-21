/* Talking to the server. */

export async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error(`${res.status} ${(await res.text()).slice(0, 160)}`);
  return res.json();
}

/* A request with a JSON body. The method/Content-Type/JSON.stringify triplet was written
 * out at twelve call sites, which is twelve chances to send an object as "[object Object]"
 * by forgetting the last part.
 *
 * The endpoints that still call `fetch` directly do so deliberately: /api/ask and
 * /api/synthesize need the Response itself to read an SSE stream, not a parsed body. */
export function send(method, path, body) {
  return api(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}
