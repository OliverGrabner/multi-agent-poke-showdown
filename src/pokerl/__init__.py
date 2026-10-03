"""LLM teammates in Pokemon Showdown 4-player Multi Battles."""

# Terastallization is switched off with Showdown's built-in Terastal Clause.
FORMAT = "gen9multirandombattle@@@Terastal Clause"
# Team generation needs the plain format id.
TEAM_FORMAT = "gen9multirandombattle"
SEATS = ("p1", "p2", "p3", "p4")
# Seats p1 and p3 share a side, as do p2 and p4.
SIDES = {"p1p3": ("p1", "p3"), "p2p4": ("p2", "p4")}
SIDE_OF = {seat: side for side, seats in SIDES.items() for seat in seats}
# Each side is a colored team; a player is named by team color and number, e.g. "Blue 2".
TEAM_COLOR = {"p1p3": "Blue", "p2p4": "Red"}
PLAYER = {"p1": "Blue 1", "p3": "Blue 2", "p2": "Red 1", "p4": "Red 2"}
ALLY = {"p1": "p3", "p3": "p1", "p2": "p4", "p4": "p2"}
FOES = {"p1": ("p2", "p4"), "p3": ("p2", "p4"), "p2": ("p1", "p3"), "p4": ("p1", "p3")}
