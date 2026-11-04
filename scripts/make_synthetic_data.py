"""Write the synthetic monthly and daily panels to CSV."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ddalloc import make_synthetic  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data")
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    panels = make_synthetic(seed=a.seed)
    panels.monthly.to_csv(out / "monthly.csv")
    panels.daily.to_csv(out / "daily.csv")
    print(f"wrote {len(panels.monthly)} months and {len(panels.daily)} days to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
