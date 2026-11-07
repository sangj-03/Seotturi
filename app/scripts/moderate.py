"""Community moderation from the server's shell (not exposed on the web).

    ../.venv/bin/python -I scripts/moderate.py posts [N]        # newest posts
    ../.venv/bin/python -I scripts/moderate.py comments [N]     # newest comments
    ../.venv/bin/python -I scripts/moderate.py delete-post ID
    ../.venv/bin/python -I scripts/moderate.py delete-comment ID
"""

import os
import sqlite3
import sys
from datetime import datetime

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "app.db")


def when(ts):
    return datetime.fromtimestamp(ts).strftime("%m/%d %H:%M")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    con = sqlite3.connect(DB)
    cmd, arg = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
    if cmd == "posts":
        for pid, board, author, title, body, created, sample in con.execute(
                "SELECT id, board, author, title, body, created, sample FROM posts ORDER BY created DESC LIMIT ?",
                (int(arg or 20),)):
            print(f"{pid:5d} {when(created)} [{board}] {author}{' (예시)' if sample else ''}: {title} | {body[:60]!r}")
    elif cmd == "comments":
        for cid, pid, author, body, created in con.execute(
                "SELECT id, post_id, author, body, created FROM comments ORDER BY created DESC LIMIT ?", (int(arg or 20),)):
            print(f"{cid:5d} {when(created)} post {pid} {author}: {body[:80]!r}")
    elif cmd == "delete-post" and arg:
        with con:
            n = con.execute("DELETE FROM posts WHERE id=?", (int(arg),)).rowcount
            con.execute("DELETE FROM comments WHERE post_id=?", (int(arg),))
            con.execute("DELETE FROM likes WHERE post_id=?", (int(arg),))
            con.execute("DELETE FROM scraps WHERE post_id=?", (int(arg),))
        print(f"deleted {n} post")
    elif cmd == "delete-comment" and arg:
        with con:
            n = con.execute("DELETE FROM comments WHERE id=? OR parent=?", (int(arg), int(arg))).rowcount
        print(f"deleted {n} comment(s)")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
