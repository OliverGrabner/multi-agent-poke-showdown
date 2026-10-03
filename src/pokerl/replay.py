"""Export a logged battle as a Showdown replay page.

The page is Showdown's standard downloadable-replay format: the battle log is embedded and the
viewer script is fetched from play.pokemonshowdown.com when the page is opened. Nothing is sent
to Showdown; the browser only downloads the static viewer.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

from pokerl import TEAM_COLOR

TEMPLATE = """<!DOCTYPE html>
<meta charset="utf-8" />
<!-- version 1 -->
<title>{title}</title>
<style>
body {{font-family: Verdana, sans-serif; font-size: 10pt; margin: 0; padding: 12px 0;}}
/* Team colors for speaker names in the team chat; !important beats Showdown's own chat styles. */
.battle-log strong.team-blue {{color: #3b82f6 !important;}}
.battle-log strong.team-red {{color: #ef4444 !important;}}
</style>
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


def chat_lines(record: dict) -> dict[int, list[str]]:
    """Each team's private talk and choices as log lines, keyed by where they go in the log.

    Only the speaker's name is colored by team. The talk for a decision happened just before the
    log reached that decision's `log_index`, so the viewer shows it right before the moves it led to.
    """
    log_index = {step: entry["log_index"] for step, entry in enumerate(record["steps"])}
    lines: dict[int, list[str]] = {}
    for side_name, side in record["sides"].items():
        if side["kind"] != "talking":
            continue
        # Showdown strips inline colors but keeps classes; the page's CSS colors these.
        team_class = f"team-{TEAM_COLOR[side_name].lower()}"
        for event in side["transcript"]:
            speaker = f'<strong class="{team_class}">{html.escape(record["players"][event["seat"]])}</strong>'
            if event["event"] == "say":
                body = f"{speaker}: {html.escape(event['text'])}"
            elif event["event"] == "choose":
                body = f"{speaker} <em>chose {html.escape(event['label'])}</em>"
            else:
                continue
            # One protocol line: no newlines, and a "|" would end the line early.
            body = " ".join(body.split()).replace("|", "/")
            lines.setdefault(log_index[event["step"]], []).append(f'|raw|<div class="chat">{body}</div>')
    return lines


def log_with_chat(record: dict) -> list[str]:
    chats = chat_lines(record)
    merged = []
    for index, line in enumerate(record["omniscient_log"]):
        merged.extend(chats.get(index, []))
        merged.append(line)
    merged.extend(chats.get(len(record["omniscient_log"]), []))
    return merged


def replay_html(record: dict, format_name: str = "[Gen 9] Multi Random Battle") -> str:
    players = record["players"]
    subtitle = html.escape(
        f"{players['p1']} + {players['p3']} vs. {players['p2']} + {players['p4']}"
        f" | winner: {record['winning_side']} | {record['turns']} turns"
    )
    # The log sits inside a <script> element, and only "</script" could end it early.
    log = re.sub(r"</(script)", r"<\\/\1", "\n".join(log_with_chat(record)), flags=re.IGNORECASE)
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
