from __future__ import annotations

import pickle
import sys
from pathlib import Path

from .solver_process import solve_request


def main() -> int:
    if len(sys.argv) != 3:
        return 2
    request_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    try:
        with request_path.open("rb") as fh:
            request = pickle.load(fh)
        payload = ("ok", solve_request(request))
    except BaseException as exc:
        payload = ("error", f"{type(exc).__name__}: {exc}")
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
