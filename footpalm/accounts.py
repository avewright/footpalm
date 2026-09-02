"""User accounts, sessions, and uploaded prediction models."""

from __future__ import annotations

import json
import re
import secrets
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

COOKIE = "footpalm"
SESSION_DAYS = 30
USERNAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]{1,23}$")
RESERVED = {"footpalm", "admin", "models"}
ADMIN_ENGINES = (
    ("lightgbm", "FootPalm LightGBM"),
    ("xgboost", "FootPalm XGBoost"),
    ("tabpfn", "FootPalm TabPFN"),
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(ts: datetime | None = None) -> str:
    return (ts or utcnow()).replace(microsecond=0).isoformat()


def game_key(game: dict) -> str:
    gid = game.get("game_id")
    if gid is not None and str(gid).strip() != "":
        return str(gid)
    return f"{game.get('season')}-{game.get('week')}-{game.get('away')}-{game.get('home')}"


def is_final(game: dict) -> bool:
    return game.get("actual_home") is not None and game.get("actual_away") is not None


def score_picks(games: list[dict], picks: dict[str, dict]) -> dict[str, Any]:
    played = [g for g in games if is_final(g) and game_key(g) in picks]
    su_w = 0
    brier = 0.0
    brier_n = 0
    mae = 0.0
    residual = 0.0
    for game in played:
        pick = picks[game_key(game)]
        pred_home = float(pick["pred_home"]) > float(pick["pred_away"])
        actual_home = bool(game.get("home_won"))
        if pred_home == actual_home:
            su_w += 1
        p = pick.get("home_win_prob")
        if p is not None and game.get("home_won") is not None:
            brier += (float(p) - float(game["home_won"])) ** 2
            brier_n += 1
        if game.get("actual_margin") is not None:
            margin = float(pick["pred_home"]) - float(pick["pred_away"])
            mae += abs(margin - float(game["actual_margin"]))
            residual += float(game["actual_margin"]) - margin
    n = len(played)
    return {
        "n": n,
        "suW": su_w,
        "suL": n - su_w,
        "brier": (brier / brier_n) if brier_n else None,
        "mae": (mae / n) if n else None,
        "residual": (residual / n) if n else None,
    }


class Store:
    def __init__(self, root: Path):
        self.root = root
        self.dir = root / "data" / "accounts"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = self.dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        self.db_path = self.dir / "accounts.sqlite"
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init()

    def _init(self) -> None:
        with self.lock:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS models (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    season INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    uploaded_at TEXT NOT NULL,
                    published INTEGER NOT NULL DEFAULT 1,
                    active INTEGER NOT NULL DEFAULT 0,
                    matched INTEGER NOT NULL DEFAULT 0,
                    unmatched INTEGER NOT NULL DEFAULT 0
                );
                """
            )
            self.conn.commit()

    def user_by_id(self, user_id: str) -> dict | None:
        with self.lock:
            row = self.conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None

    def user_by_username(self, username: str) -> dict | None:
        with self.lock:
            row = self.conn.execute(
                "SELECT * FROM users WHERE lower(username) = ?", (username.lower(),)
            ).fetchone()
        return dict(row) if row else None

    def create_user(self, username: str, role: str = "user") -> dict:
        if not USERNAME_RE.match(username):
            raise ValueError("Use 2–24 letters, numbers, or underscores, starting with a letter.")
        if username.lower() in RESERVED and role != "admin":
            raise ValueError("That name is reserved.")
        if self.user_by_username(username):
            raise ValueError("That name is taken.")
        user = {
            "id": secrets.token_hex(8),
            "username": username,
            "password_hash": "",
            "salt": "",
            "role": role,
            "created_at": iso(),
        }
        with self.lock:
            self.conn.execute(
                "INSERT INTO users (id, username, password_hash, salt, role, created_at) VALUES (?,?,?,?,?,?)",
                (user["id"], user["username"], "", "", role, user["created_at"]),
            )
            self.conn.commit()
        return user

    def claim(self, username: str) -> tuple[dict, str]:
        name = username.strip()
        user = self.user_by_username(name)
        if not user:
            user = self.create_user(name)
        token = secrets.token_urlsafe(32)
        expires = iso(utcnow() + timedelta(days=SESSION_DAYS))
        with self.lock:
            self.conn.execute(
                "INSERT INTO sessions (token, user_id, expires_at) VALUES (?,?,?)",
                (token, user["id"], expires),
            )
            self.conn.commit()
        return user, token

    def logout(self, token: str | None) -> None:
        if not token:
            return
        with self.lock:
            self.conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
            self.conn.commit()

    def user_from_token(self, token: str | None) -> dict | None:
        if not token:
            return None
        with self.lock:
            row = self.conn.execute("SELECT * FROM sessions WHERE token = ?", (token,)).fetchone()
        if not row:
            return None
        if row["expires_at"] < iso():
            self.logout(token)
            return None
        return self.user_by_id(row["user_id"])

    def public_user(self, user: dict) -> dict:
        return {"id": user["id"], "username": user["username"], "role": user["role"]}

    def _picks_path(self, model_id: str) -> Path:
        return self.models_dir / f"{model_id}.json"

    def load_picks(self, model_id: str) -> dict[str, dict]:
        path = self._picks_path(model_id)
        if not path.exists():
            return {}
        raw = json.loads(path.read_text())
        return raw if isinstance(raw, dict) else {}

    def save_picks(self, model_id: str, picks: dict[str, dict]) -> None:
        self._picks_path(model_id).write_text(json.dumps(picks, separators=(",", ":")))

    def load_games(self, season: int) -> list[dict]:
        path = self.root / "data" / "processed" / f"predictions-{season}.json"
        if not path.exists():
            path = self.root / "web" / "public" / "data" / f"predictions-{season}.json"
        if not path.exists():
            return []
        payload = json.loads(path.read_text())
        games = payload.get("games") if isinstance(payload, dict) else payload
        return [g for g in (games or []) if g.get("fbs_fbs", True)]

    def list_rows(self, user_id: str | None = None, season: int | None = None) -> list[dict]:
        sql = "SELECT * FROM models"
        args: list[Any] = []
        clauses: list[str] = []
        if user_id:
            clauses.append("user_id = ?")
            args.append(user_id)
        if season is not None:
            clauses.append("season = ?")
            args.append(season)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY uploaded_at DESC"
        with self.lock:
            rows = self.conn.execute(sql, args).fetchall()
        return [dict(row) for row in rows]

    def model_row(self, model_id: str) -> dict | None:
        with self.lock:
            row = self.conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
        return dict(row) if row else None

    def set_active(self, user_id: str, season: int, model_id: str) -> None:
        with self.lock:
            self.conn.execute(
                "UPDATE models SET active = 0 WHERE user_id = ? AND season = ?",
                (user_id, season),
            )
            self.conn.execute(
                "UPDATE models SET active = 1 WHERE id = ? AND user_id = ?",
                (model_id, user_id),
            )
            self.conn.commit()

    def create_model(
        self,
        user: dict,
        name: str,
        season: int,
        source: str,
        picks: dict[str, dict],
        unmatched: int = 0,
        published: bool = True,
        active: bool = True,
    ) -> dict:
        if not picks:
            raise ValueError("Model has no matched picks.")
        if season < 2014 or season > 2100:
            raise ValueError("Bad season.")
        cleaned: dict[str, dict] = {}
        for key, pick in picks.items():
            if not isinstance(pick, dict):
                continue
            try:
                away = float(pick["pred_away"])
                home = float(pick["pred_home"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Each pick needs pred_away and pred_home.") from exc
            p = pick.get("home_win_prob")
            cleaned[str(key)] = {
                "pred_away": away,
                "pred_home": home,
                "home_win_prob": None if p is None else float(p),
            }
        if not cleaned:
            raise ValueError("Model has no matched picks.")
        label = (name or source or "Uploaded model").strip()[:80] or "Uploaded model"
        model_id = secrets.token_hex(8)
        now = iso()
        with self.lock:
            others = self.conn.execute(
                "SELECT COUNT(*) AS n FROM models WHERE user_id = ? AND season = ?",
                (user["id"], season),
            ).fetchone()["n"]
            is_active = 1 if active or others == 0 else 0
            if is_active:
                self.conn.execute(
                    "UPDATE models SET active = 0 WHERE user_id = ? AND season = ?",
                    (user["id"], season),
                )
            self.conn.execute(
                """INSERT INTO models
                   (id, user_id, name, season, source, uploaded_at, published, active, matched, unmatched)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    model_id,
                    user["id"],
                    label,
                    int(season),
                    (source or "upload")[:120],
                    now,
                    1 if published else 0,
                    is_active,
                    len(cleaned),
                    int(unmatched),
                ),
            )
            self.conn.commit()
        self.save_picks(model_id, cleaned)
        return self.model_row(model_id) or {}

    def patch_model(self, user: dict, model_id: str, patch: dict) -> dict:
        row = self.model_row(model_id)
        if not row or row["user_id"] != user["id"]:
            raise KeyError("Model not found.")
        name = row["name"]
        if "name" in patch and str(patch["name"]).strip():
            name = str(patch["name"]).strip()[:80]
        published = row["published"]
        if "published" in patch:
            published = 1 if patch["published"] else 0
        with self.lock:
            self.conn.execute(
                "UPDATE models SET name = ?, published = ? WHERE id = ?",
                (name, published, model_id),
            )
            self.conn.commit()
        if patch.get("active"):
            self.set_active(user["id"], row["season"], model_id)
        updated = self.model_row(model_id)
        assert updated
        return updated

    def delete_model(self, user: dict, model_id: str) -> None:
        row = self.model_row(model_id)
        if not row or row["user_id"] != user["id"]:
            raise KeyError("Model not found.")
        with self.lock:
            self.conn.execute("DELETE FROM models WHERE id = ?", (model_id,))
            if row["active"]:
                nxt = self.conn.execute(
                    "SELECT id FROM models WHERE user_id = ? AND season = ? ORDER BY uploaded_at DESC LIMIT 1",
                    (user["id"], row["season"]),
                ).fetchone()
                if nxt:
                    self.conn.execute("UPDATE models SET active = 1 WHERE id = ?", (nxt["id"],))
            self.conn.commit()
        path = self._picks_path(model_id)
        if path.exists():
            path.unlink()

    def payload(self, row: dict, include_picks: bool = False) -> dict:
        owner = self.user_by_id(row["user_id"])
        body = {
            "id": row["id"],
            "name": row["name"],
            "owner": owner["username"] if owner else "unknown",
            "owner_id": row["user_id"],
            "kind": "admin" if owner and owner["role"] == "admin" else "community",
            "season": row["season"],
            "source": row["source"],
            "uploaded_at": row["uploaded_at"],
            "published": bool(row["published"]),
            "active": bool(row["active"]),
            "matched": row["matched"],
            "unmatched": row["unmatched"],
        }
        if include_picks:
            body["picks"] = self.load_picks(row["id"])
        return body

    def admin_models(self, season: int, games: list[dict]) -> list[dict]:
        if not games:
            return []
        out = [self._engine_model("ensemble", "FootPalm", season, games)]
        for engine, label in ADMIN_ENGINES:
            if any((g.get("models") or {}).get(engine) for g in games):
                out.append(self._engine_model(engine, label, season, games))
        return [row for row in out if row["matched"]]

    def _engine_model(self, engine: str, name: str, season: int, games: list[dict]) -> dict:
        picks: dict[str, dict] = {}
        for game in games:
            if engine == "ensemble":
                if game.get("pred_home") is None or game.get("pred_away") is None:
                    continue
                picks[game_key(game)] = {
                    "pred_away": float(game["pred_away"]),
                    "pred_home": float(game["pred_home"]),
                    "home_win_prob": game.get("home_win_prob"),
                }
                continue
            pick = (game.get("models") or {}).get(engine)
            if not pick:
                continue
            margin = float(pick["pred_margin"])
            total = float(game.get("pred_home", 0) or 0) + float(game.get("pred_away", 0) or 0)
            if total <= 0:
                total = 50.0
            picks[game_key(game)] = {
                "pred_away": (total - margin) / 2,
                "pred_home": (total + margin) / 2,
                "home_win_prob": pick.get("home_win_prob"),
            }
        return {
            "id": f"admin:{engine}:{season}",
            "name": name,
            "owner": "FootPalm",
            "owner_id": "footpalm",
            "kind": "admin",
            "season": season,
            "source": "board",
            "uploaded_at": iso(),
            "published": True,
            "active": False,
            "matched": len(picks),
            "unmatched": 0,
            "picks": picks,
        }

    def catalog(self, season: int, viewer: dict | None) -> list[dict]:
        games = self.load_games(season)
        cards: list[dict] = []
        for admin in self.admin_models(season, games):
            picks = admin.pop("picks")
            admin["score"] = score_picks(games, picks)
            admin["kind"] = "admin"
            cards.append(admin)
        for row in self.list_rows(season=season):
            owner = self.user_by_id(row["user_id"])
            mine = bool(viewer and row["user_id"] == viewer["id"])
            if not row["published"] and not mine:
                continue
            card = self.payload(row)
            if mine:
                card["kind"] = "you"
            elif owner and owner["role"] == "admin":
                card["kind"] = "admin"
            else:
                card["kind"] = "community"
            card["score"] = score_picks(games, self.load_picks(row["id"]))
            cards.append(card)
        return cards

    def active_model(self, user: dict, season: int) -> dict | None:
        with self.lock:
            row = self.conn.execute(
                "SELECT * FROM models WHERE user_id = ? AND season = ? AND active = 1",
                (user["id"], season),
            ).fetchone()
            if row is None:
                row = self.conn.execute(
                    "SELECT * FROM models WHERE user_id = ? AND season = ? ORDER BY uploaded_at DESC LIMIT 1",
                    (user["id"], season),
                ).fetchone()
        if not row:
            return None
        return self.payload(dict(row), include_picks=True)

    def mine(self, user: dict, season: int) -> list[dict]:
        return [self.payload(row, include_picks=True) for row in self.list_rows(user["id"], season)]


def cookie_token(handler: BaseHTTPRequestHandler) -> str | None:
    raw = handler.headers.get("Cookie") or ""
    jar = SimpleCookie()
    try:
        jar.load(raw)
    except Exception:
        return None
    if COOKIE in jar:
        return jar[COOKIE].value
    auth = handler.headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    return None


def cookie_header(token: str | None, clear: bool = False) -> str:
    if clear or not token:
        return f"{COOKIE}=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0"
    age = SESSION_DAYS * 24 * 3600
    return f"{COOKIE}={token}; Path=/; HttpOnly; SameSite=Lax; Max-Age={age}"


class Accounts:
    def __init__(self, root: Path):
        self.store = Store(root)

    def dispatch(self, handler: BaseHTTPRequestHandler, method: str, body: dict | None) -> dict | None:
        parsed = urlparse(handler.path)
        path = parsed.path.rstrip("/") or "/"
        query = {k: (v[0] if v else "") for k, v in parse_qs(parsed.query).items()}
        if not path.startswith("/api/auth") and not path.startswith("/api/models"):
            return None
        token = cookie_token(handler)
        user = self.store.user_from_token(token)
        try:
            code, payload, cookie = self._route(method, path, query, body or {}, user, token)
        except ValueError as exc:
            return {"code": 400, "payload": {"error": str(exc)}}
        except KeyError as exc:
            return {"code": 404, "payload": {"error": str(exc)}}
        except PermissionError as exc:
            return {"code": 401, "payload": {"error": str(exc)}}
        out: dict[str, Any] = {"code": code, "payload": payload}
        if cookie is not None:
            out["cookie"] = cookie
        return out

    def _route(
        self,
        method: str,
        path: str,
        query: dict[str, str],
        body: dict,
        user: dict | None,
        token: str | None,
    ) -> tuple[int, dict, str | None]:
        if path == "/api/auth/me" and method == "GET":
            return 200, {"user": self.store.public_user(user) if user else None}, None
        if path in {"/api/auth/login", "/api/auth/register"} and method == "POST":
            claimed, session = self.store.claim(str(body.get("username") or ""))
            return 200, {"user": self.store.public_user(claimed)}, cookie_header(session)
        if path == "/api/auth/logout" and method == "POST":
            self.store.logout(token)
            return 200, {"ok": True}, cookie_header(None, clear=True)
        if path == "/api/models" and method == "GET":
            season = int(query.get("season") or 0)
            if not season:
                raise ValueError("season is required.")
            if query.get("mine"):
                if not user:
                    raise PermissionError("Log in first.")
                return 200, {"models": self.store.mine(user, season)}, None
            return 200, {"models": self.store.catalog(season, user)}, None
        if path == "/api/models/active" and method == "GET":
            season = int(query.get("season") or 0)
            if not season:
                raise ValueError("season is required.")
            if not user:
                return 200, {"model": None}, None
            return 200, {"model": self.store.active_model(user, season)}, None
        if path == "/api/models" and method == "POST":
            if not user:
                raise PermissionError("Log in to upload a model.")
            row = self.store.create_model(
                user,
                str(body.get("name") or ""),
                int(body.get("season") or 0),
                str(body.get("source") or ""),
                body.get("picks") if isinstance(body.get("picks"), dict) else {},
                unmatched=int(body.get("unmatched") or 0),
                published=bool(body.get("published", True)),
                active=bool(body.get("active", True)),
            )
            return 200, {"model": self.store.payload(row, include_picks=True)}, None
        parts = path.split("/")
        if len(parts) == 4 and parts[1] == "api" and parts[2] == "models":
            model_id = parts[3]
            if method == "GET":
                if model_id.startswith("admin:"):
                    _, engine, season_s = model_id.split(":", 2)
                    season = int(season_s)
                    games = self.store.load_games(season)
                    for admin in self.store.admin_models(season, games):
                        if admin["id"] == model_id:
                            return 200, {"model": admin}, None
                    raise KeyError("Model not found.")
                row = self.store.model_row(model_id)
                if not row:
                    raise KeyError("Model not found.")
                mine = bool(user and row["user_id"] == user["id"])
                if not row["published"] and not mine:
                    raise KeyError("Model not found.")
                return 200, {"model": self.store.payload(row, include_picks=True)}, None
            if not user:
                raise PermissionError("Log in first.")
            if method == "PATCH":
                row = self.store.patch_model(user, model_id, body)
                return 200, {"model": self.store.payload(row, include_picks=True)}, None
            if method == "DELETE":
                self.store.delete_model(user, model_id)
                return 200, {"ok": True}, None
        return 404, {"error": "not found"}, None
