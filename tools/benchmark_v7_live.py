from __future__ import annotations

import threading
import time

from _bootstrap import ROOT

from vision_engine.live.state import GameState
from vision_engine.solver.client import SolverClient
from vision_engine.solver.protocol import SolverRequest, SolverSettings
from vision_engine.solver.service import SolverEngine, SolverTCPServer

CELLS=("BL","CR","CH","CH","CR","CR","SO","BL","LI","CR","CH","HA",None,"SO","LI","SO","LI","HA","BL","LI","BL","CH","SO","HA","HA")
MASK=(1<<4)|(1<<13)|(1<<17)


def make_state(jewel=None, mask=MASK):
    return GameState("bench", 1, 0, "board", CELLS, mask, jewel, "score")


def main():
    engine=SolverEngine(workers=8, cache_entries=256)
    server=SolverTCPServer(("127.0.0.1",0),engine,socket_timeout=120)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    host,port=server.server_address
    client=SolverClient(host,port,timeout=120)
    settings=SolverSettings(rollouts=120)
    try:
        t=time.perf_counter(); cold=client.solve(SolverRequest(1,make_state("LI"),settings)); cold_ms=(time.perf_counter()-t)*1000
        t=time.perf_counter(); hot=client.solve(SolverRequest(2,make_state("LI"),settings)); hot_ms=(time.perf_counter()-t)*1000
        predicted=MASK | (1 << cold.recommendations[0].position)
        reqs=[SolverRequest(-(i+1),make_state(j,predicted),settings,purpose="bench-precompute") for i,j in enumerate(("BL","SO","LI","CR","HA","CH"))]
        t=time.perf_counter(); ack=client.precompute_requests(reqs); ack_ms=(time.perf_counter()-t)*1000
        deadline=time.time()+30
        while time.time()<deadline:
            ping=client.ping()
            if ping.get("cache_entries",0)>=7:
                break
            time.sleep(.05)
        warm_ms=(time.perf_counter()-t)*1000
        t=time.perf_counter(); next_hot=client.solve(SolverRequest(3,make_state("SO",predicted),settings,purpose="live-decision")); next_ms=(time.perf_counter()-t)*1000
        print(f"cold_live_ms={cold_ms:.1f} source={cold.source}")
        print(f"hot_cache_ms={hot_ms:.2f} source={hot.source}")
        print(f"precompute_ack_ms={ack_ms:.2f} scheduled={ack.get('scheduled')}")
        print(f"precompute_six_ready_ms={warm_ms:.1f}")
        print(f"next_jewel_cache_ms={next_ms:.2f} cache_hit={next_hot.cache_hit}")
    finally:
        client.close(); server.shutdown(); server.server_close(); engine.close()

if __name__ == "__main__":
    main()
