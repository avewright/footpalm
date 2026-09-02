import type { UserModel, UserScore } from "./mymodel";

export type SessionUser = {
  id: string;
  username: string;
  role: "user" | "admin" | string;
};

export type ModelKind = "you" | "community" | "admin";

export type ModelCard = {
  id: string;
  name: string;
  owner: string;
  kind: ModelKind;
  season: number;
  source: string;
  uploaded_at: string;
  published: boolean;
  active: boolean;
  matched: number;
  unmatched: number;
  score: UserScore | null;
};

async function call<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(path, {
    credentials: "include",
    ...init,
    headers: {
      ...(init.body ? { "Content-Type": "application/json" } : {}),
      ...(init.headers ?? {}),
    },
  });
  const type = res.headers.get("content-type") ?? "";
  const payload = type.includes("json") ? ((await res.json()) as { error?: string }) : {};
  if (!res.ok) {
    throw new Error(
      payload.error || (res.status === 502 ? "Accounts server is not running." : `Request failed (${res.status})`),
    );
  }
  return payload as T;
}

export async function fetchMe(): Promise<SessionUser | null> {
  try {
    const { user } = await call<{ user: SessionUser | null }>("/api/auth/me");
    return user;
  } catch {
    return null;
  }
}

export async function login(username: string): Promise<SessionUser> {
  const { user } = await call<{ user: SessionUser }>("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ username }),
  });
  return user;
}

export async function logout(): Promise<void> {
  await call("/api/auth/logout", { method: "POST", body: "{}" });
}

export async function fetchCatalog(season: number): Promise<ModelCard[]> {
  const { models } = await call<{ models: ModelCard[] }>(`/api/models?season=${season}`);
  return models;
}

export async function fetchMine(season: number): Promise<UserModel[]> {
  const { models } = await call<{ models: UserModel[] }>(`/api/models?season=${season}&mine=1`);
  return models;
}

export async function fetchActive(season: number): Promise<UserModel | null> {
  const { model } = await call<{ model: UserModel | null }>(`/api/models/active?season=${season}`);
  return model;
}

export async function createModel(model: UserModel): Promise<UserModel> {
  const { model: saved } = await call<{ model: UserModel }>("/api/models", {
    method: "POST",
    body: JSON.stringify({
      name: model.name,
      season: model.season,
      source: model.source,
      picks: model.picks,
      unmatched: model.unmatched,
      published: model.published ?? true,
      active: true,
    }),
  });
  return saved;
}

export async function patchModel(
  id: string,
  patch: { name?: string; published?: boolean; active?: boolean },
): Promise<UserModel> {
  const { model } = await call<{ model: UserModel }>(`/api/models/${id}`, {
    method: "PATCH",
    body: JSON.stringify(patch),
  });
  return model;
}

export async function deleteModel(id: string): Promise<void> {
  await call(`/api/models/${id}`, { method: "DELETE" });
}
