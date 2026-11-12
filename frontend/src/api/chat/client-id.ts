const CLIENT_ID_KEY = "client-id";

let clientId: string | undefined;

// There is no login: the browser is identified by an opaque id made once and kept in localStorage
export function getClientId(): string {
  if (clientId) return clientId;
  let id: string | null = null;
  try {
    id = localStorage.getItem(CLIENT_ID_KEY);
  } catch {
    // Storage is blocked: the id lives until the page is closed
  }
  if (!id) {
    id = crypto.randomUUID();
    try {
      localStorage.setItem(CLIENT_ID_KEY, id);
    } catch {
      // See above
    }
  }
  clientId = id;
  return id;
}
