import type { Card } from "./AskViz";

const STORE = "footpalm.ask.v1";
const MAX = 40;

export type Reply = {
  content: string;
  cards?: Card[];
  tools?: string[];
};

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  cards?: Card[];
  tools?: string[];
  replies?: Reply[];
  replyAt?: number;
};

export type Thread = {
  id: string;
  title: string;
  season: number;
  updated: number;
  messages: ChatMessage[];
};

type Store = { threads: Thread[]; active: string | null };

function read(): Store {
  try {
    const raw = localStorage.getItem(STORE);
    if (!raw) return { threads: [], active: null };
    const parsed = JSON.parse(raw) as Store;
    if (!parsed || !Array.isArray(parsed.threads)) return { threads: [], active: null };
    return { threads: parsed.threads, active: parsed.active ?? null };
  } catch {
    return { threads: [], active: null };
  }
}

function write(store: Store) {
  try {
    localStorage.setItem(STORE, JSON.stringify(store));
  } catch {
    /* quota */
  }
}

export function newId() {
  return `t-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;
}

export function emptyThread(season: number): Thread {
  return { id: newId(), title: "New chat", season, updated: Date.now(), messages: [] };
}

export function titleFrom(messages: ChatMessage[]) {
  const first = messages.find((m) => m.role === "user")?.content ?? "";
  const one = first.replace(/\s+/g, " ").trim();
  if (!one) return "New chat";
  return one.length > 42 ? `${one.slice(0, 41)}…` : one;
}

export function shown(msg: ChatMessage): Reply {
  if (msg.role === "user" || !msg.replies?.length) {
    return { content: msg.content, cards: msg.cards, tools: msg.tools };
  }
  const i = Math.min(Math.max(msg.replyAt ?? msg.replies.length - 1, 0), msg.replies.length - 1);
  return msg.replies[i];
}

export function asAssistant(reply: Reply): ChatMessage {
  return { role: "assistant", ...reply, replies: [reply], replyAt: 0 };
}

export function addReply(msg: ChatMessage, reply: Reply): ChatMessage {
  const prev = msg.replies?.length ? msg.replies : [{ content: msg.content, cards: msg.cards, tools: msg.tools }];
  const replies = [...prev, reply];
  return { role: "assistant", ...reply, replies, replyAt: replies.length - 1 };
}

export function pickReply(msg: ChatMessage, at: number): ChatMessage {
  const replies = msg.replies ?? [];
  const i = Math.min(Math.max(at, 0), Math.max(replies.length - 1, 0));
  const reply = replies[i] ?? { content: msg.content, cards: msg.cards, tools: msg.tools };
  return { ...msg, ...reply, replies, replyAt: i };
}

export function listThreads(): Thread[] {
  return [...read().threads].sort((a, b) => b.updated - a.updated);
}

export function loadThread(id: string): Thread | null {
  return read().threads.find((t) => t.id === id) ?? null;
}

export function activeId(): string | null {
  return read().active;
}

export function setActive(id: string | null) {
  const store = read();
  store.active = id;
  write(store);
}

export function saveThread(thread: Thread) {
  if (!thread.messages.length) return thread;
  const store = read();
  const next = { ...thread, title: titleFrom(thread.messages), updated: Date.now() };
  const i = store.threads.findIndex((t) => t.id === next.id);
  if (i >= 0) store.threads[i] = next;
  else store.threads.unshift(next);
  store.threads = store.threads.sort((a, b) => b.updated - a.updated).slice(0, MAX);
  store.active = next.id;
  write(store);
  return next;
}

export function deleteThread(id: string): string | null {
  const store = read();
  store.threads = store.threads.filter((t) => t.id !== id);
  if (store.active === id) store.active = store.threads[0]?.id ?? null;
  write(store);
  return store.active;
}

export function payloadMessages(messages: ChatMessage[]) {
  return messages.map((m) => ({ role: m.role, content: shown(m).content }));
}

const MODE = "footpalm.ask.mode";

export type AskMode = "instant" | "deep";

export function loadMode(): AskMode {
  try {
    return localStorage.getItem(MODE) === "deep" ? "deep" : "instant";
  } catch {
    return "instant";
  }
}

export function saveMode(mode: AskMode) {
  try {
    localStorage.setItem(MODE, mode);
  } catch {
    /* quota */
  }
}
