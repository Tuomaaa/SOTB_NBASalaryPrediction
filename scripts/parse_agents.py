"""Parse RealGM agent HTML and create agent features."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import unicodedata
import re
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
RAW = PROJ / "data" / "raw" / "raw_external"
PROC = PROJ / "data" / "processed"


def norm(name):
    name = str(name).strip().lower()
    nfkd = unicodedata.normalize("NFKD", name)
    out = "".join(c for c in nfkd if not unicodedata.combining(c))
    out = out.replace(".", "").replace("-", " ")
    out = re.sub(r"\s+", " ", out).strip()
    return out


def parse_agent_html():
    path = RAW / "NBA Player Agent Relationships - RealGM.html"
    with open(path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    table = soup.find("table")
    rows_html = table.find_all("tr")
    headers = [th.get_text(strip=True) for th in rows_html[0].find_all("th")]

    data = []
    for tr in rows_html[1:]:
        tds = tr.find_all("td")
        if len(tds) < len(headers):
            continue
        row = {headers[i]: tds[i].get_text(strip=True) for i in range(len(headers))}
        # Fix concatenated agent names: take primary agent only
        agent_td = tds[7]
        links = agent_td.find_all("a")
        if len(links) > 1:
            row["Agent"] = links[0].get_text(strip=True)
        data.append(row)

    df = pd.DataFrame(data)
    df["player_name_norm"] = df["Player"].apply(norm)
    df["agent"] = df["Agent"].str.strip()
    return df


def try_match(name, lookup):
    if name in lookup:
        return lookup[name]
    no_suffix = re.sub(r"\s+(jr|sr|iii|ii|iv)$", "", name)
    if no_suffix in lookup:
        return lookup[no_suffix]
    if name + " jr" in lookup:
        return lookup[name + " jr"]
    if name + " iii" in lookup:
        return lookup[name + " iii"]
    return None


def main():
    agents_df = parse_agent_html()
    print(f"Parsed {len(agents_df)} agent rows, {agents_df['agent'].nunique()} unique agents")

    # Save raw agent data
    agents_out = agents_df[["Player", "player_name_norm", "agent", "Team", "Contractual Status", "YOS"]].copy()
    agents_out.columns = ["player_name", "player_name_norm", "agent", "team", "contractual_status", "yos"]
    agents_out.to_csv(PROC / "agent_data.csv", index=False)
    print(f"Saved agent_data.csv: {len(agents_out)} rows")

    # Build lookup
    agent_lookup = agents_df.set_index("player_name_norm")["agent"].to_dict()
    agent_counts = agents_df["agent"].value_counts().to_dict()

    # Match against training data
    train = pd.read_csv(PROC / "training_data.csv")
    train["pn_clean"] = train["player_name_norm"].apply(norm)
    train["agent"] = train["pn_clean"].apply(lambda x: try_match(x, agent_lookup))

    has = train["agent"].notna().sum()
    total = len(train)
    players_matched = train[train["agent"].notna()]["player_name_norm"].nunique()
    players_total = train["player_name_norm"].nunique()
    print(f"\nTraining rows with agent: {has}/{total} ({has/total*100:.0f}%)")
    print(f"Players matched: {players_matched}/{players_total} ({players_matched/players_total*100:.0f}%)")

    # agent_client_count feature
    train["agent_client_count"] = train["agent"].map(agent_counts).fillna(0).astype(int)

    ha = train[train["agent"].notna()]
    corr = ha["agent_client_count"].corr(ha["cap_pct"])
    print(f"Correlation agent_client_count vs cap_pct: {corr:.3f}")

    # Top agents
    print(f"\nTop agents by avg cap_pct (min 5 training rows):")
    grp = ha.groupby("agent").agg(n=("cap_pct", "count"), avg_cap=("cap_pct", "mean"))
    grp = grp[grp["n"] >= 5].sort_values("avg_cap", ascending=False)
    for agent, row in grp.head(15).iterrows():
        sal_m = row["avg_cap"] * 153
        print(f"  {agent:30s} n={int(row['n']):>3d}  avg_cap={row['avg_cap']:.3f}  ~${sal_m:.1f}M")

    # Unmatched analysis
    unmatched = train[train["agent"].isna()]["player_name_norm"].unique()
    unmatched_seasons = train[train["player_name_norm"].isin(unmatched)].groupby("player_name_norm")["season"].max()
    print(f"\nUnmatched ({len(unmatched)} players) by latest season:")
    print(unmatched_seasons.value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
