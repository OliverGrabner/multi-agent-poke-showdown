"""Plain-English battle events, written from one player's point of view.

Each log line becomes one short sentence ("The opposing Ceruledge used Bitter Blade on your
Salazzle."). Lines with no meaning for a player are skipped; rare kinds fall back to the raw line.
"""

from __future__ import annotations

from pokerl import ALLY, SIDE_OF
from pokerl.tracker import Line, clean, parse, split_ident

SKIP = {
    "",
    "t:",
    "gametype",
    "gen",
    "tier",
    "rule",
    "teamsize",
    "player",
    "start",
    "turn",
    "upkeep",
    "split",
    "-anim",
    "request",
    "inactive",
    "inactiveoff",
    "teampreview",
    "clearpoke",
    "poke",
    "raw",
    "html",
    "uhtml",  # HTML announcements such as the custom-rule banner
}
STATUS = {
    "brn": "burned",
    "par": "paralyzed",
    "slp": "asleep",
    "psn": "poisoned",
    "tox": "badly poisoned",
    "frz": "frozen",
}
STATS = {
    "atk": "Attack",
    "def": "Defense",
    "spa": "Sp. Atk",
    "spd": "Sp. Def",
    "spe": "Speed",
    "accuracy": "accuracy",
    "evasion": "evasiveness",
}
WEATHER = {
    "SunnyDay": "harsh sunlight",
    "RainDance": "rain",
    "Sandstorm": "a sandstorm",
    "Snowscape": "snow",
    "DesolateLand": "extremely harsh sunlight",
    "PrimordialSea": "heavy rain",
    "DeltaStream": "strong winds",
}
CANT = {
    "par": "it is paralyzed",
    "slp": "it is asleep",
    "frz": "it is frozen",
    "flinch": "it flinched",
    "recharge": "it must recharge",
}


class Narrator:
    def __init__(self, seat: str, partner_name: str):
        self.seat = seat
        self.partner_name = partner_name

    def narrate(self, raw_lines: list[str]) -> list[str]:
        sentences = []
        for raw in raw_lines:
            line = parse(raw)
            if line is None or line.kind in SKIP:
                continue
            sentence = self.sentence(line)
            if sentence:
                sentences.append(sentence[0].upper() + sentence[1:])
        return sentences

    def who(self, ident: str) -> str:
        """'p1a: Salazzle' -> 'your Salazzle', "Sam's Hatterene" or 'the opposing Salazzle'."""
        seat, name = split_ident(ident)
        if seat == self.seat:
            return f"your {name}"
        if seat == ALLY[self.seat]:
            return f"{self.partner_name}'s {name}"
        return f"the opposing {name}"

    def owner(self, seat: str) -> str:
        if seat == self.seat:
            return "you"
        if seat == ALLY[self.seat]:
            return self.partner_name
        return f"opponent {seat}"

    def side(self, ident: str) -> str:
        seat, _ = split_ident(ident)
        return "your side" if SIDE_OF[seat] == SIDE_OF[self.seat] else "the opposing side"

    def hp(self, ident: str, condition: str) -> str:
        """Exact HP for the viewer's own Pokémon, a percentage for everyone else."""
        amount, _, status = condition.partition(" ")
        if status == "fnt" or amount == "0":
            return "fainted"
        current, _, maximum = amount.partition("/")
        if split_ident(ident)[0] == self.seat:
            return f"{current}/{maximum} HP"
        return f"{round(100 * int(current) / int(maximum))}% HP"

    def because(self, line: Line) -> str:
        source = line.tags.get("from")
        if not source:
            return ""
        source = STATUS.get(source, clean(source))
        if line.tags.get("of") and line.tags["of"] != (line.args[0] if line.args else ""):
            return f" (from {self.who(line.tags['of'])}'s {source})"
        return f" (from {source})"

    def sentence(self, line: Line) -> str | None:
        a, tags = line.args, line.tags
        match line.kind:
            case "move":
                target = f" on {self.who(a[2])}" if len(a) > 2 and a[2] and a[2] != a[0] else ""
                ending = " but missed" if "miss" in tags else ""
                return f"{self.who(a[0])} used {a[1]}{target}{ending}."
            case "switch" | "drag" | "replace":
                seat, name = split_ident(a[0])
                verb = "was dragged out" if line.kind == "drag" else "came in"
                return f"{self.owner(seat)}: {name} {verb} ({self.hp(a[0], a[2])})."
            case "-damage":
                return f"{self.who(a[0])} took damage{self.because(line)}, now at {self.hp(a[0], a[1])}."
            case "-heal":
                return f"{self.who(a[0])} recovered HP{self.because(line)}, now at {self.hp(a[0], a[1])}."
            case "-sethp":
                return f"{self.who(a[0])}'s HP is now {self.hp(a[0], a[1])}."
            case "faint":
                return f"{self.who(a[0])} fainted."
            case "-supereffective":
                return f"It's super effective on {self.who(a[0])}."
            case "-resisted":
                return f"It's not very effective on {self.who(a[0])}."
            case "-immune":
                return f"{self.who(a[0])} is unaffected{self.because(line)}."
            case "-crit":
                return f"A critical hit on {self.who(a[0])}!"
            case "-miss":
                return f"{self.who(a[0])}'s attack missed."
            case "-fail":
                return f"It failed ({self.who(a[0])})."
            case "-status":
                return f"{self.who(a[0])} is now {STATUS.get(a[1], a[1])}{self.because(line)}."
            case "-curestatus":
                return f"{self.who(a[0])} is no longer {STATUS.get(a[1], a[1])}."
            case "-boost" | "-unboost":
                change = "rose" if line.kind == "-boost" else "fell"
                return f"{self.who(a[0])}'s {STATS.get(a[1], a[1])} {change} by {a[2]}{self.because(line)}."
            case "-setboost":
                return f"{self.who(a[0])}'s {STATS.get(a[1], a[1])} was set to {a[2]}."
            case "-clearboost" | "-clearnegativeboost":
                return f"{self.who(a[0])}'s stat changes were removed."
            case "-clearallboost":
                return "All stat changes were removed."
            case "-weather":
                if "upkeep" in tags:
                    return None
                if a[0] == "none":
                    return "The weather cleared."
                return f"The weather became {WEATHER.get(a[0], a[0])}{self.because(line)}."
            case "-fieldstart":
                return f"{clean(a[0])} started{self.because(line)}."
            case "-fieldend":
                return f"{clean(a[0])} ended."
            case "-sidestart":
                return f"{clean(a[1])} was set up on {self.side(a[0])}."
            case "-sideend":
                return f"{clean(a[1])} ended on {self.side(a[0])}."
            case "-start":
                effect = f"its type changed to {a[2]}" if a[1] == "typechange" else f"{clean(a[1])} started"
                return f"{self.who(a[0])}: {effect}{self.because(line)}."
            case "-end":
                return f"{self.who(a[0])}: {clean(a[1])} ended."
            case "-singleturn" | "-activate":
                return f"{self.who(a[0])}: {clean(a[1])}{self.because(line)}." if len(a) > 1 else None
            case "-item":
                return f"{self.who(a[0])} has {a[1]}{self.because(line)}."
            case "-enditem":
                verb = "ate its" if "eat" in tags else "lost its"
                return f"{self.who(a[0])} {verb} {a[1]}{self.because(line)}."
            case "-ability":
                return f"{self.who(a[0])}'s ability is {a[1]}."
            case "cant":
                reason = CANT.get(a[1], clean(a[1]))
                return f"{self.who(a[0])} couldn't move because {reason}."
            case "-hitcount":
                return f"It hit {a[1]} times."
            case "-prepare":
                return f"{self.who(a[0])} is charging up {a[1]}."
            case "-transform":
                return f"{self.who(a[0])} transformed into {self.who(a[1])}."
            case "detailschange" | "-formechange":
                return f"{self.who(a[0])} changed form to {a[1].split(', ')[0]}."
            case "-hint" | "-message":
                return f"Note: {a[0]}"
            case "win":
                return f"{a[0]} won the battle."
            case _:
                return f"[{line.kind}] {' | '.join(a)}"
