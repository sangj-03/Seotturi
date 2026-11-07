"""Build the static copy of the site for GitHub Pages, and optionally publish it.

    python3 scripts/build_static.py              # build into static_site/
    python3 scripts/build_static.py --publish    # build, then force-push static_site/ as one commit

The copy keeps the landing page and the benchmark page with their CSS, fonts, images and
scripts, so both stay reachable while the DGX is off. It rewrites absolute paths for the
Pages sub-path (--base), points the app at the live server (--live), and saves the live
/api/status, /api/bench and /api/accent as data/*.json. In the browser, scripts/static/shim.js uses the
live server when it answers and falls back to that snapshot otherwise.

Publishing replaces the single commit of the repository's gh-pages branch (the code stays on main),
so the branch does not grow with every snapshot. It only pushes when the site changed, ignoring
timestamps that move on every call. The live server's public address comes from --live or
SEOTTURI_LIVE_URL; without one, the copy shows the snapshot only.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "web")
STATIC = os.path.join(ROOT, "scripts", "static")
SNAPSHOT = os.path.join(ROOT, "state", "static_snapshot")
VOLATILE = {"server_time", "queue", "busy", "snapshot_time"}  # change on every call; not worth a deploy


def rewrite_html(html, base, live):
    """Absolute paths -> Pages sub-path; the app -> the live server; add the shim."""
    html = html.replace('<iframe src="/app?frame=1"', '<iframe data-live-src="@@LIVE@@/app?frame=1"')
    html = re.sub(r'href="/app"', 'href="@@LIVE@@/app" data-live-link', html)
    html = html.replace('href="/benchmark"', 'href="@@BASE@@/benchmark/"')
    html = re.sub(r'href="/(#[^"]*)?"', r'href="@@BASE@@/\1"', html)
    html = re.sub(r'(href|src)="/(?!/)', r'\1="@@BASE@@/', html)
    head = ('<link rel="stylesheet" href="@@BASE@@/css/static.css">\n'
            f'  <script>window.SEOTTURI_STATIC = {json.dumps({"base": base, "live": live})};</script>\n'
            '  <script src="@@BASE@@/js/shim.js"></script>\n</head>')
    html = html.replace("</head>", head, 1)
    return html.replace("@@BASE@@", base).replace("@@LIVE@@", live)


def rewrite_css(css, base):
    return re.sub(r'url\((["\']?)/(?!/)', rf'url(\1{base}/', css)


def fetch_snapshot(api):
    """Current /api/status and /api/bench, or the last saved ones if the server does not answer."""
    os.makedirs(SNAPSHOT, exist_ok=True)
    now = time.time()
    for name in ("status", "bench", "accent"):
        try:
            with urllib.request.urlopen(f"{api}/api/{name}", timeout=15) as r:
                body = json.load(r)
            if name == "status" and not body.get("model"):
                raise ValueError("engine not ready")
            if name in ("bench", "accent") and not body.get("items"):
                raise ValueError("no benchmark results")
            body["snapshot_time"] = now
            with open(os.path.join(SNAPSHOT, f"{name}.json"), "w", encoding="utf-8") as f:
                json.dump(body, f, ensure_ascii=False)
            print(f"snapshot {name}: fresh")
        except Exception as e:  # keep serving the previous snapshot
            print(f"snapshot {name}: kept previous ({e})")


def clean(out):
    os.makedirs(out, exist_ok=True)
    for name in os.listdir(out):
        if name == ".git":
            continue
        path = os.path.join(out, name)
        shutil.rmtree(path) if os.path.isdir(path) and not os.path.islink(path) else os.remove(path)


def build(out, base, live, api):
    fetch_snapshot(api)
    if not all(os.path.exists(os.path.join(SNAPSHOT, f"{n}.json")) for n in ("status", "bench")):
        raise SystemExit("no snapshot yet: start the server once (./run.sh start) before building")
    clean(out)
    for d in ("assets", "fonts"):
        shutil.copytree(os.path.join(WEB, d), os.path.join(out, d))
    os.makedirs(os.path.join(out, "css"))
    for name in ("landing.css", "benchmark.css", "alive.css"):
        with open(os.path.join(WEB, "css", name), encoding="utf-8") as f:
            css = rewrite_css(f.read(), base)
        with open(os.path.join(out, "css", name), "w", encoding="utf-8") as f:
            f.write(css)
    shutil.copy(os.path.join(STATIC, "static.css"), os.path.join(out, "css", "static.css"))
    os.makedirs(os.path.join(out, "js"))
    for name in ("landing.js", "benchmark.js", "alive.js"):
        with open(os.path.join(WEB, "js", name), encoding="utf-8") as f:
            js = f.read()
        if name == "landing.js":
            js = js.replace('"AI 엔진에 연결하지 못했어요"', '"AI 서버가 잠시 꺼져 있어요"')
        with open(os.path.join(out, "js", name), "w", encoding="utf-8") as f:
            f.write(js)
    shutil.copy(os.path.join(STATIC, "shim.js"), os.path.join(out, "js", "shim.js"))
    for src, dst in (("index.html", "index.html"), ("benchmark.html", "benchmark/index.html")):
        with open(os.path.join(WEB, src), encoding="utf-8") as f:
            html = rewrite_html(f.read(), base, live)
        os.makedirs(os.path.dirname(os.path.join(out, dst)) or out, exist_ok=True)
        with open(os.path.join(out, dst), "w", encoding="utf-8") as f:
            f.write(html)
    os.makedirs(os.path.join(out, "data"))
    for name in ("status", "bench", "accent"):
        if os.path.exists(os.path.join(SNAPSHOT, f"{name}.json")):  # accent is optional
            shutil.copy(os.path.join(SNAPSHOT, f"{name}.json"), os.path.join(out, "data", f"{name}.json"))
    open(os.path.join(out, ".nojekyll"), "w").close()  # serve files as they are


def fingerprint(out):
    """Hash of the built site, ignoring values that change on every snapshot."""
    paths = []
    for dirpath, dirs, files in os.walk(out):
        dirs[:] = [d for d in dirs if d != ".git"]
        paths += [os.path.join(dirpath, name) for name in files]
    h = hashlib.sha256()
    for path in sorted(paths):
        with open(path, "rb") as f:
            data = f.read()
        if path.endswith(".json") and os.path.basename(os.path.dirname(path)) == "data":
            body = json.loads(data)
            data = json.dumps({k: v for k, v in body.items() if k not in VOLATILE}, sort_keys=True).encode()
        h.update(os.path.relpath(path, out).encode() + b"\0" + data)
    return h.hexdigest()


def publish(out, repo):
    git = lambda *a: subprocess.run(["git", "-C", out, *a], check=True, capture_output=True, text=True)
    stamp = os.path.join(ROOT, "state", "static_published")
    fp = fingerprint(out)
    if os.path.exists(stamp) and open(stamp).read().strip() == fp:
        print("publish: unchanged, nothing pushed")
        return
    if not os.path.isdir(os.path.join(out, ".git")):
        git("init", "-q", "-b", "main")
        git("config", "user.name", "sangj-03")
        git("config", "user.email", "6yoops@kookmin.ac.kr")
        git("remote", "add", "origin", f"ssh://git@ssh.github.com:443/{repo}.git")
    git("checkout", "-q", "--orphan", "deploy")
    git("add", "-A")
    git("commit", "-q", "-m", f"Deploy static site ({time.strftime('%Y-%m-%d %H:%M')} snapshot)")
    git("branch", "-q", "-M", "gh-pages")
    git("push", "-q", "-f", "origin", "gh-pages")
    with open(stamp, "w") as f:
        f.write(fp)
    print(f"publish: pushed to {repo}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "static_site"))
    ap.add_argument("--base", default="/Seotturi", help="Pages sub-path (repo name), '' for a user site")
    ap.add_argument("--live", default=os.environ.get("SEOTTURI_LIVE_URL", ""),
                    help="public address of the running server (e.g. its Tailscale Funnel URL); '' = snapshot only")
    ap.add_argument("--api", default="http://127.0.0.1:8787", help="where to read the snapshot from")
    ap.add_argument("--repo", default="sangj-03/Seotturi")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()
    print(time.strftime("%F %T"), "build", args.out)
    build(args.out, args.base.rstrip("/"), args.live.rstrip("/"), args.api)
    if args.publish:
        publish(args.out, args.repo)


if __name__ == "__main__":
    main()
