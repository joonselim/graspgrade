"""Worker for the video check page (web/validate.html). Runs on the team's computer.

It asks the site for work every few seconds, so nothing on this computer has to
accept connections from outside. A new video is downloaded, checked (60 s at
most) and priced; after someone presses start on the page, the pipeline runs
(web/worker/pipeline.py) with the Meta API key in ~/graspgrade/.env, and the
result goes back to the site. Downloaded files are deleted after each job.

Needs two lines in ~/graspgrade/.env next to MODEL_API_KEY:
    VALIDATE_PASSWORD=<the page password, also set in Vercel>
    VALIDATE_URL=https://graspgrade.vercel.app        (optional, this is the default)

Run in WSL from ~/graspgrade (web/worker/run_worker.sh restarts it if it stops):
    .venv/bin/python web/worker/worker.py
"""
import hashlib
import json
import os
import shutil
import socket
import time
import traceback
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "out" / "validate"
SALT = "graspgrade-validate-v1"
MAX_S = 60.5


def env():
    for line in (REPO / ".env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    if not os.environ.get("VALIDATE_PASSWORD"):
        raise SystemExit("VALIDATE_PASSWORD is missing from .env")
    url = os.environ.get("VALIDATE_URL", "https://graspgrade.vercel.app").rstrip("/")
    token = hashlib.sha256((SALT + os.environ["VALIDATE_PASSWORD"]).encode()).hexdigest()
    return url, token


def call(url, token, body):
    req = urllib.request.Request(url + "/api/validate", data=json.dumps(body).encode(), method="POST",
                                 headers={"content-type": "application/json", "x-validate-token": token})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def download(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        tmp = dst.with_suffix(".part")
        with urllib.request.urlopen(src, timeout=120) as r, open(tmp, "wb") as fh:
            shutil.copyfileobj(r, fh)
        tmp.rename(dst)
    return dst


def main():
    url, token = env()
    import pipeline  # after .env, so the Meta API client finds its key
    client = None
    host = socket.gethostname()
    print(f"worker for {url}, polling", flush=True)
    while True:
        try:
            nxt = call(url, token, {"action": "next", "host": host})
        except Exception as exc:  # the site or the network is down: wait and ask again
            print("poll failed:", repr(exc)[:200], flush=True)
            time.sleep(15)
            continue
        job = nxt.get("job")
        if not job:
            time.sleep(4)
            continue
        jid, d = job["id"], WORK / str(job["id"])
        update = lambda **f: call(url, token, {"action": "update", "id": jid, **f})  # noqa: E731
        try:
            src = download(job["url"], d / "input")
            if job["status"] == "probing":
                duration, w, h = pipeline.probe(src)
                if duration > MAX_S:
                    update(status="failed", duration_s=duration, error=f"영상이 {duration:.0f}초임. 60초 이하만 처리함.")
                    shutil.rmtree(d, ignore_errors=True)
                    continue
                update(status="estimated", duration_s=duration, est_cost=pipeline.estimate(duration),
                       progress=f"{w}x{h}, {duration:.1f}초")
                print(f"job {jid}: priced", flush=True)
                continue
            # status running: the page pressed start
            if client is None:
                client = pipeline.mc.make_client()
            t0 = time.time()
            result, cost = pipeline.run(src, d, lambda text: update(progress=text), client)
            result["seconds"] = round(time.time() - t0)
            update(status="done", result=result, cost=round(cost, 4), progress="완료")
            print(f"job {jid}: done, {len(result['clips'])} clips, ${cost:.2f}", flush=True)
            shutil.rmtree(d, ignore_errors=True)
        except Exception as exc:
            traceback.print_exc()
            try:
                update(status="failed", error=f"{type(exc).__name__}: {str(exc)[:300]}")
            except Exception:
                pass
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
