import { FormEvent, useEffect, useState } from "react";
import type { SessionUser } from "./accounts";

export function AuthBar({
  user,
  error,
  openSignal,
  onLogin,
  onLogout,
}: {
  user: SessionUser | null;
  error: string | null;
  openSignal: number;
  onLogin: (username: string) => Promise<void>;
  onLogout: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [username, setUsername] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (openSignal) setOpen(true);
  }, [openSignal]);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    try {
      await onLogin(username.trim());
      setOpen(false);
    } catch {
      /* parent renders error */
    } finally {
      setBusy(false);
    }
  }

  if (user) {
    return (
      <span className="auth-who">
        <span className="quiet">{user.username}</span>
        <button type="button" className="tab-link" onClick={onLogout}>
          Log out
        </button>
      </span>
    );
  }

  return (
    <div className="auth-bar">
      <button
        type="button"
        className="tab-link"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        Log in
      </button>
      {open && (
        <form className="auth-form" onSubmit={onSubmit}>
          <input
            autoFocus
            autoComplete="username"
            placeholder="Your name"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            aria-label="Name"
          />
          {error && <p className="warn">{error}</p>}
          <button type="submit" disabled={busy || username.trim().length < 2}>
            Log in
          </button>
          <p className="lede-note">Separates your model uploads. No password.</p>
        </form>
      )}
    </div>
  );
}
