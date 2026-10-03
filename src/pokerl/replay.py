"""Export a logged battle as a Showdown replay page.

The page is Showdown's standard downloadable-replay format: the battle log is embedded and the
viewer script is fetched from play.pokemonshowdown.com when the page is opened. Nothing is sent
to Showdown; the browser only downloads the static viewer.
"""

from __future__ import annotations

import html
from pathlib import Path

TEMPLATE = """<!DOCTYPE html>
<meta charset="utf-8" />
<!-- version 1 -->
<title>{title}</title>
<style>body {{font-family: Verdana, sans-serif; font-size: 10pt; margin: 0; padding: 12px 0;}}</style>
<div class="wrapper replay-wrapper" style="max-width:1180px;margin:0 auto">
<input type="hidden" name="replayid" value="{replay_id}" />
<div class="battle"></div><div class="battle-log"></div><div class="replay-controls"></div>
<div class="replay-controls-2"></div>
<h1 style="font-weight:normal;text-align:center"><strong>{format_name}</strong><br />{subtitle}</h1>
<script type="text/plain" class="battle-log-data">{log}</script>
</div>
<script>
let daily = Math.floor(Date.now()/1000/60/60/24);
document.write('<script src="https://play.pokemonshowdown.com/js/replay-embed.js?version'+daily+'"></'+'script>');
</script>
"""


def replay_html(record: dict, format_name: str = "[Gen 9] Multi Random Battle") -> str:
    policies = record["policies"]
    subtitle = html.escape(
        f"{policies['p1']} + {policies['p3']} vs. {policies['p2']} + {policies['p4']}"
        f" | winner: {record['winning_side']} | {record['turns']} turns"
    )
    log = "\n".join(record["omniscient_log"]).replace("</", "<\\/")
    return TEMPLATE.format(
        title=html.escape(record["battle_key"]),
        replay_id=html.escape(record["battle_key"]),
        format_name=html.escape(format_name),
        subtitle=subtitle,
        log=log,
    )


def write_replay(record: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(replay_html(record), encoding="utf-8")
    return path
