// Tiny safe localStorage helpers. All Nelke keys live under the "nelke:"
// prefix so they don't collide with other apps on the same origin. Access is
// wrapped in try/catch because localStorage can throw in private browsing
// modes, when disabled by the user, or when the quota is exceeded — none of
// which should crash the UI.

const PREFIX = "nelke:";

/** Read a string value, or `null` if missing / unreadable. */
export function loadString(key: string): string | null {
  try {
    return window.localStorage.getItem(PREFIX + key);
  } catch {
    return null;
  }
}

/** Write a string value. Silently no-ops on unavailable storage. */
export function saveString(key: string, value: string): void {
  try {
    window.localStorage.setItem(PREFIX + key, value);
  } catch {
    // ignore — persistence is best-effort
  }
}

/** Remove a value. Silently no-ops on unavailable storage. */
export function removeString(key: string): void {
  try {
    window.localStorage.removeItem(PREFIX + key);
  } catch {
    // ignore
  }
}
