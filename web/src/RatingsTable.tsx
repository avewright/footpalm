import { useMemo, useState } from "react";
import { signed } from "./format";
import { ConfLink, TeamLink } from "./TeamView";
import type { RatingsFile, SortKey, TeamRow } from "./types";

const COLUMNS: { key: SortKey; label: string; title: string; cls?: string }[] = [
  { key: "rank", label: "Rk", title: "Rank by Pom" },
  { key: "team", label: "Team", title: "Team", cls: "team" },
  { key: "conf", label: "Conf", title: "Conference", cls: "conf" },
  { key: "wins", label: "W-L", title: "Record", cls: "record" },
  { key: "pom", label: "Pom", title: "Points vs an average FBS team on a neutral field. Ranking key." },
  { key: "elo", label: "Elo", title: "538-style margin-of-victory Elo. 1500 is average. Not the ranking key." },
  { key: "adjo", label: "AdjO", title: "Opponent-adjusted offense, points per game vs an average defense" },
  { key: "adjd", label: "AdjD", title: "Opponent-adjusted defense, points allowed vs an average offense. Lower is better." },
  { key: "adjst", label: "AdjST", title: "Opponent-adjusted special teams, points per game" },
  { key: "tempo", label: "Tempo", title: "Scrimmage plays per team per game" },
  { key: "sos", label: "SoS", title: "Average opponent Pom" },
  { key: "luck", label: "Luck", title: "Actual win rate minus expected from score margins" },
  { key: "nil_roster", label: "Roster", title: "2026 football roster estimate (rev share + third-party NIL). Not all-sports Sideline. Not a model feature." },
];

function pomOf(row: TeamRow) {
  return Number(row.pom ?? row.palm ?? 0);
}

const LOWER_BETTER = new Set<SortKey>(["rank", "adjd"]);

function cell(row: TeamRow, key: SortKey) {
  switch (key) {
    case "wins":
      return `${row.wins}-${row.losses}`;
    case "pom":
      return signed(pomOf(row));
    case "elo":
      return row.elo == null ? "—" : String(Math.round(row.elo));
    case "adjo":
    case "adjd":
    case "adjst":
    case "sos":
      return signed(Number(row[key]));
    case "tempo":
      return row.tempo.toFixed(1);
    case "luck":
      return `${signed(row.luck * 100, 1)}%`;
    case "nil_roster":
      return row.nil_roster == null
        ? "—"
        : `$${(row.nil_roster / 1_000_000).toFixed(1)}M${row.nil_quality === "modeled" ? "*" : ""}`;
    default:
      return row[key] ?? "";
  }
}

export function RatingsTable({
  data,
  onOpenTeam,
  onOpenConference,
}: {
  data: RatingsFile;
  onOpenTeam: (team: string) => void;
  onOpenConference: (conf: string) => void;
}) {
  const [sortKey, setSortKey] = useState<SortKey>("pom");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const [q, setQ] = useState("");
  const [conf, setConf] = useState("all");

  const conferences = useMemo(
    () => ["all", ...[...new Set(data.teams.map((t) => t.conf).filter(Boolean))].sort()],
    [data],
  );

  const rows = useMemo(() => {
    const needle = q.trim().toLowerCase();
    const copy = data.teams.filter((row) => {
      if (conf !== "all" && row.conf !== conf) return false;
      if (needle && !row.team.toLowerCase().includes(needle)) return false;
      return true;
    });
    copy.sort((a, b) => {
      const av = sortKey === "pom" ? pomOf(a) : a[sortKey];
      const bv = sortKey === "pom" ? pomOf(b) : b[sortKey];
      if (av == null) return 1;
      if (bv == null) return -1;
      const cmp = typeof av === "string" ? av.localeCompare(String(bv)) : Number(av) - Number(bv);
      return sortDir === "asc" ? cmp : -cmp;
    });
    return copy;
  }, [data, sortKey, sortDir, q, conf]);

  function onSort(key: SortKey) {
    if (key === sortKey) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
      return;
    }
    setSortKey(key);
    setSortDir(LOWER_BETTER.has(key) || key === "team" || key === "conf" ? "asc" : "desc");
  }

  return (
    <div>
      <div className="toolbar">
        <input
          type="search"
          placeholder="Team"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          aria-label="Filter teams"
        />
        <select value={conf} onChange={(e) => setConf(e.target.value)} aria-label="Conference">
          {conferences.map((c) => (
            <option key={c} value={c}>
              {c === "all" ? "All conferences" : c}
            </option>
          ))}
        </select>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              {COLUMNS.map((col) => (
                <th key={col.key} className={col.cls} title={col.title}>
                  <button type="button" onClick={() => onSort(col.key)}>
                    {col.label}
                    {sortKey === col.key ? (sortDir === "asc" ? " ↑" : " ↓") : ""}
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.team}>
                {COLUMNS.map((col) => (
                  <td key={col.key} className={col.cls}>
                    {col.key === "team" ? (
                      <TeamLink team={row.team} onOpen={onOpenTeam} />
                    ) : col.key === "conf" ? (
                      <ConfLink conf={row.conf} onOpen={onOpenConference} />
                    ) : (
                      cell(row, col.key)
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
