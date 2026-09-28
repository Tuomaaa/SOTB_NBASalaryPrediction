"""Parse cached Spotrac injury tables into a dated event artifact."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from src.features.injury_history import INJURY_EVENTS, build_injury_events


def main() -> None:
    events, stats = build_injury_events()
    if events.empty:
        raise SystemExit("No Spotrac injury rows parsed")
    events.to_csv(INJURY_EVENTS, index=False)
    print("Spotrac injury extraction")
    for name, value in stats.items():
        print(f"  {name}: {value}")
    print(f"  categories: {events['injury_category'].value_counts().to_dict()}")
    print(f"  saved: {INJURY_EVENTS}")


if __name__ == "__main__":
    main()
