import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";
import { AskMarkdown, CardView, splitViz, type Card } from "./AskViz";

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  cards?: Card[];
  tools?: string[];
};

export function AskView({ season }: { season: number }) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState<number | null>(null);
  const end = useRef<HTMLDivElement>(null);
  const box = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    end.current?.scrollIntoView({ block: "end", behavior: "smooth" });
  }, [messages, busy]);

  useEffect(() => {
    box.current?.focus();
  }, []);

  async function send(text: string) {
    const question = text.trim();
    if (!question || busy) return;
    const next: ChatMessage[] = [...messages, { role: "user", content: question }];
    setMessages(next);
    setDraft("");
    setBusy(true);
    setError(null);
    if (box.current) box.current.style.height = "auto";
    try {
      const response = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          season,
          messages: next.map(({ role, content }) => ({ role, content })),
        }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || `ask ${response.status}`);
      const parsed = splitViz(payload.text || "");
      setMessages([
        ...next,
        {
          role: "assistant",
          content: parsed.text,
          cards: [...(payload.cards || []), ...parsed.cards],
          tools: payload.tools || [],
        },
      ]);
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

  function onKey(e: KeyboardEvent<HTMLTextAreaElement>) {
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

  function copy(i: number, text: string) {
    void navigator.clipboard.writeText(text);
    setCopied(i);
    window.setTimeout(() => setCopied((cur) => (cur === i ? null : cur)), 1200);
  }

  function fresh() {
    setMessages([]);
    setError(null);
    box.current?.focus();
  }

  return (
    <div className="ask">
      {messages.length > 0 && (
        <button type="button" className="ask-new" onClick={fresh}>
          New chat
        </button>
      )}
      <div className="ask-log">
        {messages.length === 0 && !busy && <p className="ask-empty">Ask anything</p>}
        {messages.map((msg, i) => (
          <article key={i} className={`ask-msg ${msg.role}`}>
            {msg.role === "user" ? (
              <div className="ask-bubble">{msg.content}</div>
            ) : (
              <div className="ask-reply">
                {msg.content && <AskMarkdown text={msg.content} />}
                {msg.cards?.map((card, j) => (
                  <CardView key={j} card={card} />
                ))}
                {!!msg.tools?.length && <p className="ask-tools">{msg.tools.join(" · ")}</p>}
                {msg.content && (
                  <button type="button" className="ask-copy" onClick={() => copy(i, msg.content)}>
                    {copied === i ? "Copied" : "Copy"}
                  </button>
                )}
              </div>
            )}
          </article>
        ))}
        {busy && (
          <div className="ask-wait" aria-live="polite">
            <i />
            <i />
            <i />
          </div>
        )}
        {error && <p className="warn">{error}</p>}
        <div ref={end} />
      </div>
      <form className="ask-form" onSubmit={onSubmit}>
        <div className="ask-composer">
          <textarea
            ref={box}
            rows={1}
            value={draft}
            onChange={(e) => onDraft(e.target)}
            onKeyDown={onKey}
            placeholder="Ask FootPalm"
            aria-label="Message"
          />
          <button type="submit" disabled={busy || !draft.trim()} aria-label="Send">
            ↑
          </button>
        </div>
      </form>
    </div>
  );
}
