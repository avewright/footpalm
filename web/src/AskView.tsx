import { FormEvent, useEffect, useRef, useState } from "react";
import type { KeyboardEvent as ReactKey } from "react";
import { AskMarkdown, CardView, splitViz, type Card } from "./AskViz";
import {
  activeId,
  addReply,
  asAssistant,
  deleteThread,
  emptyThread,
  listThreads,
  loadMode,
  loadThread,
  payloadMessages,
  pickReply,
  saveMode,
  saveThread,
  setActive,
  shown,
  type AskMode,
  type ChatMessage,
  type Thread,
} from "./chats";
import type { UserModel } from "./mymodel";

const INSERTS = [
  { label: "This week's slate", text: "What are the best bets this week?" },
  { label: "My model vs the board", text: "Compare my model to the board this week." },
];

const FILE_CAP = 60_000;

type AskBody = { error?: string; text?: string; cards?: Card[]; tools?: string[]; picks?: UserModel };

type StreamEvent =
  | { type: "delta"; text?: string }
  | { type: "tools" }
  | { type: "done"; text?: string; cards?: Card[]; tools?: string[]; picks?: UserModel }
  | { type: "error"; error?: string };

function Icon({
  name,
  label,
}: {
  name: "side" | "compose" | "search" | "plus" | "copy" | "retry";
  label?: string;
}) {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.75"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden={label ? undefined : true}
    >
      {label && <title>{label}</title>}
      {name === "side" && (
        <>
          <rect x="3" y="4" width="18" height="16" rx="2" />
          <path d="M9 4v16" />
        </>
      )}
      {name === "compose" && (
        <>
          <path d="M12 20H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v6" />
          <path d="M14 14.5 20.5 8 22 9.5 15.5 16H14z" />
        </>
      )}
      {name === "search" && (
        <>
          <circle cx="11" cy="11" r="6" />
          <path d="m16 16 4 4" />
        </>
      )}
      {name === "plus" && <path d="M12 5v14M5 12h14" />}
      {name === "copy" && (
        <>
          <rect x="8" y="8" width="12" height="12" rx="2" />
          <path d="M4 16V6a2 2 0 0 1 2-2h10" />
        </>
      )}
      {name === "retry" && <path d="M3 12a9 9 0 1 0 3-6.7M3 4v5h5" />}
    </svg>
  );
}

function Rail({
  list,
  active,
  q,
  onQ,
  onNew,
  onOpen,
  onDelete,
  onHide,
}: {
  list: Thread[];
  active: string;
  q: string;
  onQ: (v: string) => void;
  onNew: () => void;
  onOpen: (id: string) => void;
  onDelete: (id: string) => void;
  onHide: () => void;
}) {
  const [find, setFind] = useState(false);
  const needle = q.trim().toLowerCase();
  const rows = needle ? list.filter((t) => t.title.toLowerCase().includes(needle)) : list;
  return (
    <aside className="ask-nav">
      <div className="ask-nav-top">
        <button type="button" className="ask-new" onClick={onNew}>
          New chat
        </button>
        <div className="ask-nav-tools">
          <button
            type="button"
            className="ask-icon"
            aria-label="Search chats"
            aria-pressed={find}
            onClick={() => setFind((v) => !v)}
          >
            <Icon name="search" />
          </button>
          <button type="button" className="ask-icon" aria-label="Hide chats" onClick={onHide}>
            <Icon name="side" />
          </button>
        </div>
      </div>
      {find && (
        <input
          type="search"
          className="ask-nav-find"
          placeholder="Search"
          value={q}
          aria-label="Search chats"
          autoFocus
          onChange={(e) => onQ(e.target.value)}
        />
      )}
      <p className="ask-nav-label">Recents</p>
      <div className="ask-nav-list">
        {rows.length === 0 && <p className="lede-note">No chats yet.</p>}
        {rows.map((t) => (
          <div key={t.id} className="ask-nav-row">
            <button type="button" aria-current={t.id === active} onClick={() => onOpen(t.id)} title={t.title}>
              {t.title}
            </button>
            <button type="button" className="ask-nav-x" aria-label={`Delete ${t.title}`} onClick={() => onDelete(t.id)}>
              ×
            </button>
          </div>
        ))}
      </div>
    </aside>
  );
}

async function readStream(response: Response, onEvent: (ev: StreamEvent) => void): Promise<AskBody> {
  const type = response.headers.get("content-type") ?? "";
  if (!type.includes("event-stream")) {
    const raw = await response.text();
    let payload: AskBody = {};
    if (raw.trim()) {
      try {
        payload = JSON.parse(raw);
      } catch {
        throw new Error(`ask ${response.status}`);
      }
    }
    if (!response.ok) throw new Error(payload.error || (raw.trim() ? `ask ${response.status}` : "Ask server is not running"));
    if (!raw.trim()) throw new Error("Ask server is not running");
    return payload;
  }
  if (!response.body) throw new Error("ask stream failed");
  const reader = response.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  let done: AskBody | null = null;
  while (true) {
    const { value, done: end } = await reader.read();
    if (end) break;
    buf += dec.decode(value, { stream: true });
    const blocks = buf.split("\n\n");
    buf = blocks.pop() ?? "";
    for (const block of blocks) {
      const line = block.split("\n").find((row) => row.startsWith("data:"));
      if (!line) continue;
      let ev: StreamEvent;
      try {
        ev = JSON.parse(line.slice(5).trim()) as StreamEvent;
      } catch {
        continue;
      }
      if (ev.type === "error") throw new Error(ev.error || "ask failed");
      if (ev.type === "done") {
        done = ev;
        continue;
      }
      onEvent(ev);
    }
  }
  if (!done) throw new Error("ask stream stopped");
  return done;
}

export function AskView({
  season,
  picks,
  onPicks,
  onOpenTeam,
}: {
  season: number;
  picks?: UserModel | null;
  onPicks?: (next: UserModel) => void;
  onOpenTeam?: (team: string) => void;
}) {
  const [thread, setThread] = useState<Thread>(() => {
    const id = activeId();
    return (id && loadThread(id)) || emptyThread(season);
  });
  const [list, setList] = useState(listThreads);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState<number | null>(null);
  const [plus, setPlus] = useState(false);
  const [q, setQ] = useState("");
  const [mode, setMode] = useState<AskMode>(loadMode);
  const [rail, setRail] = useState(() => window.matchMedia("(min-width: 800px)").matches);
  const end = useRef<HTMLDivElement>(null);
  const box = useRef<HTMLTextAreaElement>(null);
  const file = useRef<HTMLInputElement>(null);
  const messages = thread.messages;
  const empty = messages.length === 0 && !busy;

  useEffect(() => {
    end.current?.scrollIntoView({ block: "end" });
  }, [messages, busy]);

  useEffect(() => {
    box.current?.focus();
  }, [thread.id]);

  useEffect(() => {
    if (!plus) return;
    function close(e: MouseEvent) {
      if (!(e.target instanceof Element) || e.target.closest(".ask-plus-wrap")) return;
      setPlus(false);
    }
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [plus]);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "b") {
        e.preventDefault();
        setRail((v) => !v);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  function commit(next: Thread) {
    const saved = saveThread(next);
    setThread(saved);
    setList(listThreads());
  }

  function open(id: string) {
    if (busy) return;
    const hit = loadThread(id);
    if (!hit) return;
    setActive(id);
    setThread(hit);
    setError(null);
    setPlus(false);
  }

  function fresh() {
    if (busy) return;
    setThread(emptyThread(season));
    setError(null);
    setPlus(false);
    setDraft("");
    if (box.current) box.current.style.height = "auto";
  }

  function remove(id: string) {
    if (busy) return;
    const next = deleteThread(id);
    setList(listThreads());
    if (thread.id === id) setThread((next && loadThread(next)) || emptyThread(season));
  }

  function pickMode(next: AskMode) {
    setMode(next);
    saveMode(next);
  }

  function finish(payload: AskBody): ChatMessage {
    const parsed = splitViz(payload.text || "");
    if (payload.picks?.picks && Object.keys(payload.picks.picks).length && onPicks) {
      onPicks(payload.picks);
    }
    return asAssistant({
      content: parsed.text,
      cards: [...(payload.cards || []), ...parsed.cards],
      tools: payload.tools || [],
    });
  }

  async function ask(history: ChatMessage[], live?: (text: string) => void): Promise<ChatMessage> {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({
        season,
        mode,
        stream: true,
        picks: picks ?? undefined,
        messages: payloadMessages(history),
      }),
    });
    let acc = "";
    const payload = await readStream(response, (ev) => {
      if (ev.type === "delta" && ev.text) {
        acc += ev.text;
        live?.(acc);
      }
      if (ev.type === "tools") {
        acc = "";
        live?.("");
      }
    });
    return finish(payload);
  }

  async function send(text: string) {
    const question = text.trim();
    if (!question || busy) return;
    const history: ChatMessage[] = [...messages, { role: "user", content: question }];
    setThread({ ...thread, messages: history });
    setDraft("");
    setBusy(true);
    setError(null);
    setPlus(false);
    if (box.current) box.current.style.height = "auto";
    const paint = (next: string) => {
      setThread({
        ...thread,
        messages: next ? [...history, asAssistant({ content: next })] : history,
      });
    };
    try {
      const reply = await ask(history, paint);
      commit({ ...thread, messages: [...history, reply] });
    } catch (err) {
      commit({ ...thread, messages: history });
      setError(err instanceof Error ? err.message : "ask failed");
    } finally {
      setBusy(false);
      box.current?.focus();
    }
  }

  async function regenerate(i: number) {
    const msg = messages[i];
    if (busy || msg?.role !== "assistant") return;
    const history = messages.slice(0, i);
    setBusy(true);
    setError(null);
    const paint = (next: string) => {
      setThread({
        ...thread,
        messages: next ? [...history, asAssistant({ content: next })] : history,
      });
    };
    try {
      const reply = shown(await ask(history, paint));
      commit({ ...thread, messages: [...history, addReply(msg, reply)] });
    } catch (err) {
      setError(err instanceof Error ? err.message : "ask failed");
    } finally {
      setBusy(false);
      box.current?.focus();
    }
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    void send(draft);
  }

  function onKey(e: ReactKey<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void send(draft);
    }
  }

  function onDraft(el: HTMLTextAreaElement) {
    setDraft(el.value);
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }

  function insert(text: string) {
    const next = draft.trim() ? `${draft.replace(/\s+$/, "")}\n${text}` : text;
    setDraft(next);
    setPlus(false);
    requestAnimationFrame(() => {
      if (!box.current) return;
      box.current.style.height = "auto";
      box.current.style.height = `${Math.min(box.current.scrollHeight, 160)}px`;
      box.current.focus();
    });
  }

  function onFile(list: FileList | null) {
    const picked = list?.[0];
    if (file.current) file.current.value = "";
    if (!picked) return;
    const reader = new FileReader();
    reader.onload = () => {
      const body = String(reader.result ?? "");
      if (!body.trim()) return;
      const clipped = body.length > FILE_CAP ? `${body.slice(0, FILE_CAP)}\n…` : body;
      insert(`[${picked.name}]\n${clipped}`);
    };
    reader.readAsText(picked);
  }

  function turn(i: number, at: number) {
    commit({ ...thread, messages: messages.map((m, j) => (j === i ? pickReply(m, at) : m)) });
  }

  function copy(i: number, text: string) {
    void navigator.clipboard.writeText(text);
    setCopied(i);
    window.setTimeout(() => setCopied((cur) => (cur === i ? null : cur)), 1200);
  }

  const composer = (
    <form className="ask-form" onSubmit={onSubmit}>
      <div className="ask-composer">
        <div className="ask-plus-wrap">
          <button type="button" className="ask-plus" aria-label="Add" aria-expanded={plus} onClick={() => setPlus((v) => !v)}>
            <Icon name="plus" />
          </button>
          {plus && (
            <div className="ask-plus-menu">
              <button type="button" onClick={() => file.current?.click()}>
                Attach a file
              </button>
              {INSERTS.map((item) => (
                <button key={item.label} type="button" onClick={() => insert(item.text)}>
                  {item.label}
                </button>
              ))}
            </div>
          )}
        </div>
        <input ref={file} type="file" hidden accept=".txt,.csv,.json,.md,.tsv" onChange={(e) => onFile(e.target.files)} />
        <textarea
          ref={box}
          rows={1}
          value={draft}
          onChange={(e) => onDraft(e.target)}
          onKeyDown={onKey}
          placeholder="Ask FootPalm"
          aria-label="Message"
        />
        <button
          type="button"
          className="ask-mode"
          aria-label="Response mode"
          onClick={() => pickMode(mode === "instant" ? "deep" : "instant")}
        >
          {mode === "deep" ? "Deep" : "Instant"}
        </button>
        <button type="submit" className="ask-send" disabled={busy || !draft.trim()} aria-label="Send">
          ↑
        </button>
      </div>
    </form>
  );

  return (
    <div className={`ask${rail ? " has-rail" : ""}`}>
      {rail && (
        <Rail
          list={list}
          active={thread.id}
          q={q}
          onQ={setQ}
          onNew={fresh}
          onOpen={open}
          onDelete={remove}
          onHide={() => setRail(false)}
        />
      )}
      <div className={`ask-main${empty ? " is-empty" : ""}`}>
        {!rail && (
          <div className="ask-iconrail">
            <button type="button" className="ask-icon" aria-label="Show chats" onClick={() => setRail(true)}>
              <Icon name="side" />
            </button>
            <button type="button" className="ask-icon" aria-label="New chat" onClick={fresh}>
              <Icon name="compose" />
            </button>
          </div>
        )}
        <div className="ask-log">
          <div className="ask-thread">
            {empty && <p className="ask-empty">What’s on the board?</p>}
            {messages.map((msg, i) => {
              if (msg.role === "user") {
                return (
                  <article key={i} className="ask-msg user">
                    <div className="ask-bubble">{msg.content}</div>
                  </article>
                );
              }
              const reply = shown(msg);
              const n = msg.replies?.length ?? 1;
              const at = msg.replyAt ?? n - 1;
              const streaming = busy && i === messages.length - 1 && !reply.cards?.length;
              return (
                <article key={i} className="ask-msg assistant">
                  <div className="ask-reply">
                    {reply.cards?.map((card, j) => (
                      <CardView key={j} card={card} onOpen={onOpenTeam} />
                    ))}
                    {reply.content && <AskMarkdown text={reply.content} />}
                    {!streaming && (
                      <div className="ask-actions">
                        {reply.content && (
                          <button type="button" aria-label={copied === i ? "Copied" : "Copy"} onClick={() => copy(i, reply.content)}>
                            <Icon name="copy" label={copied === i ? "Copied" : "Copy"} />
                          </button>
                        )}
                        <button type="button" aria-label="Retry" disabled={busy} onClick={() => void regenerate(i)}>
                          <Icon name="retry" label="Retry" />
                        </button>
                        {n > 1 && (
                          <span className="ask-pager">
                            <button type="button" disabled={at <= 0} onClick={() => turn(i, at - 1)} aria-label="Previous reply">
                              ‹
                            </button>
                            {at + 1}/{n}
                            <button type="button" disabled={at >= n - 1} onClick={() => turn(i, at + 1)} aria-label="Next reply">
                              ›
                            </button>
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                </article>
              );
            })}
            {busy && messages[messages.length - 1]?.role !== "assistant" && (
              <div className="ask-wait" aria-live="polite">
                <i />
                <i />
                <i />
              </div>
            )}
            {error && <p className="warn">{error}</p>}
            <div ref={end} />
          </div>
        </div>
        {composer}
      </div>
    </div>
  );
}
