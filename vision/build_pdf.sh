#!/usr/bin/env bash
# Build vision.pdf from vision.md.
# pandoc (md -> styled HTML) + headless Chrome (HTML -> PDF).
# Chrome renders SVG / emoji / CJK with real system fonts; --virtual-time-budget
# advances the animated hero ~4 s so its scene-0 phrase is fully typed in the capture.
set -euo pipefail
cd "$(dirname "$0")"

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
HEAD="$(mktemp /tmp/vision-head.XXXX.html)"

cat > "$HEAD" <<'CSS'
<meta charset="utf-8">
<style>
  :root { color-scheme: light; }
  body{font-family:-apple-system,"Segoe UI",Inter,system-ui,"Helvetica Neue",Arial,"PingFang SC","Hiragino Sans",sans-serif;
       max-width:840px;margin:0 auto;padding:8px 28px 40px;color:#1f2433;line-height:1.6;font-size:15.5px;-webkit-print-color-adjust:exact;print-color-adjust:exact;}
  h1{font-size:30px;font-weight:800;letter-spacing:-.5px;margin:.4em 0 .3em;line-height:1.15;}
  h2{font-size:22px;font-weight:700;margin:1.5em 0 .4em;padding-bottom:.25em;border-bottom:1px solid #e5e8f0;}
  h3{font-size:17px;font-weight:700;margin:1.2em 0 .3em;color:#3b3f51;}
  p,li{font-size:15.5px;}
  a{color:#4f46e5;text-decoration:none;}
  img{display:block;width:100%;height:auto;margin:14px auto;border-radius:14px;break-inside:avoid;page-break-inside:avoid;}
  blockquote{margin:14px 0;padding:10px 18px;background:#f5f6fb;border-left:4px solid #6366f1;border-radius:0 10px 10px 0;color:#2b2f40;}
  blockquote p{margin:.3em 0;}
  table{border-collapse:collapse;width:100%;margin:14px 0;font-size:14px;break-inside:avoid;}
  th,td{border:1px solid #e2e5ef;padding:7px 11px;text-align:left;vertical-align:top;}
  th{background:#eef0f8;font-weight:700;}
  tr:nth-child(even) td{background:#fafbfe;}
  code{background:#eef0f8;padding:1px 6px;border-radius:5px;font-size:13px;font-family:"SF Mono",ui-monospace,Menlo,monospace;}
  hr{border:none;border-top:1px solid #e5e8f0;margin:26px 0;}
  em{color:#555a6e;}
  h1,h2,h3{break-after:avoid;page-break-after:avoid;}
  @page{margin:14mm 0;}
</style>
CSS

pandoc vision.md -f gfm -t html5 -s -H "$HEAD" --metadata pagetitle="Vision — Hibiki-Zero on device" -o vision.html

"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer \
  --run-all-compositor-stages-before-draw --virtual-time-budget=4000 \
  --user-data-dir="$(mktemp -d)" \
  --print-to-pdf="vision.pdf" \
  "file://$(pwd)/vision.html" 2>/dev/null

rm -f "$HEAD"
echo "Wrote $(pwd)/vision.pdf"
