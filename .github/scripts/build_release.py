#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""اسکریپت ساخت ریلیز WhisperSub — در ورکفلو ریلیز اجرا می‌شود.

کارها:
۱) بلوک «اطلاعات ریلیز» را در بالای سلول کد نوت‌بوک درج/به‌روزرسانی می‌کند (idempotent)
۲) فایل src/whispersub.py را از روی نوت‌بوک (منبع اصلی پروژه) تولید می‌کند

استفاده: python3 build_release.py <version> <commit_sha>
"""

import json
import re
import sys
from datetime import date

NOTEBOOK = "notebook/whispersub.ipynb"
SCRIPT = "src/whispersub.py"
REPO_URL = "https://github.com/Arsalani2/WhisperSub"

BEGIN = "# ==== WhisperSub Release Info (auto-generated) ===="
END = "# ==== /WhisperSub Release Info ===="


def build_header(version, commit):
    return "\n".join([
        BEGIN,
        "# WhisperSub — نسخه v" + version,
        "# تاریخ ساخت این ریلیز: " + date.today().isoformat() + " | کامیت: " + commit,
        "# ریلیز: " + REPO_URL + "/releases/tag/v" + version,
        "# این نوت‌بوک منبع اصلی پروژه است؛ فایل " + SCRIPT + " به‌صورت خودکار از روی همین فایل ساخته می‌شود.",
        END,
    ])


def main():
    if len(sys.argv) != 3:
        raise SystemExit("استفاده: build_release.py <version> <commit_sha>")
    version = sys.argv[1]
    commit = sys.argv[2][:7]

    with open(NOTEBOOK, encoding="utf-8") as f:
        nb = json.load(f)

    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    if not code_cells:
        raise SystemExit("هیچ سلول کدی در نوت‌بوک پیدا نشد.")

    src = "".join(code_cells[0]["source"])

    # حذف بلوک اطلاعات ریلیز قبلی تا درج مجدد idempotent بماند
    pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n\n?", re.DOTALL)
    src = pattern.sub("", src)

    header = build_header(version, commit)
    lines = src.split("\n")
    insert_at = 1 if lines and lines[0].startswith("# -*- coding") else 0
    lines = lines[:insert_at] + header.split("\n") + [""] + lines[insert_at:]
    src = "\n".join(lines)

    out_lines = src.split("\n")
    if out_lines and out_lines[-1] == "":
        out_lines.pop()
    code_cells[0]["source"] = [l + "\n" for l in out_lines]

    with open(NOTEBOOK, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")

    # ساخت src/whispersub.py از روی سلول‌های کد نوت‌بوک
    parts = []
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            parts.append("".join(cell["source"]).rstrip("\n"))
    with open(SCRIPT, "w", encoding="utf-8") as f:
        f.write("\n\n\n".join(parts) + "\n")

    print("[Info] اطلاعات ریلیز v" + version + " در " + NOTEBOOK + " درج شد و " + SCRIPT + " ساخته شد.")


if __name__ == "__main__":
    main()
