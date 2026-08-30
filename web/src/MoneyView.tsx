import { useMemo, useState } from "react";
import { money } from "./format";
import type { MoneyFile } from "./types";

type Key = "team" | "conf" | "nil_roster" | "nil_all_sports" | "athletic_spend" | "staff_payroll" | "pom";

export function MoneyView({ data }: { data: MoneyFile }) {
  const [sortKey, setSortKey] = useState<Key>("nil_roster");
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
      const av = a[sortKey];
      const bv = b[sortKey];
      if (av == null) return 1;
      if (bv == null) return -1;
      const cmp = typeof av === "string" ? av.localeCompare(String(bv)) : Number(av) - Number(bv);
      return sortDir === "asc" ? cmp : -cmp;
    });
    return copy;
  }, [data, sortKey, sortDir, q, conf]);

  function onSort(key: Key) {
    if (key === sortKey) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else {
      setSortKey(key);
      setSortDir(key === "team" || key === "conf" ? "asc" : "desc");
    }
  }

  return (
    <div>
      <p className="lede-note">{data.note}</p>
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
              {(["team", "conf", "nil_roster", "nil_all_sports", "athletic_spend", "staff_payroll", "pom"] as Key[]).map((key) => (
                <th key={key} className={key === "team" || key === "conf" ? key : undefined}>
                  <button type="button" onClick={() => onSort(key)}>
                    {key === "nil_roster"
                      ? "FB roster"
                      : key === "nil_all_sports"
                        ? "All sports"
                        : key === "athletic_spend"
                          ? "Dept spend"
                          : key === "staff_payroll"
                            ? "Staff (conf)"
                            : key === "pom"
                              ? "Pom"
                              : key}
                    {sortKey === key ? (sortDir === "asc" ? " ↑" : " ↓") : ""}
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.team}>
                <td className="team">{row.team}</td>
                <td className="conf">{row.conf}</td>
                <td>
                  {money(row.nil_roster)}
                  {row.nil_quality === "modeled" ? " *" : ""}
                </td>
                <td>{money(row.nil_all_sports)}</td>
                <td>{money(row.athletic_spend)}</td>
                <td>{money(row.staff_payroll)}</td>
                <td>{row.pom?.toFixed(1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
