# Decisions log

Choices left to the building agent (handoff section 8), plus decisions John made in conversation.
Newest entries go at the bottom of each section. Dates are when the decision was made.

## Agreed with John (2026-10-02)

| Decision | Detail |
|---|---|
| Free talk always | A "ready" flag on each message; talk ends when both latest messages are ready. Then each agent privately and simultaneously picks a move and predicts its partner's move. If you said ready and your partner sends a new not-ready message, you get to reply. No-talk is a later experiment. |
| Talk before forced switches | Teammates also talk (same protocol) before mid-turn replacements after a faint, even if only one of them is switching. |
| Thinking | Use each model's native thinking as its private reasoning; no separate reasoning field. Run each battle turn as a tool loop (`say(message, ready)`, `choose(action, partner_prediction)`) so Qwen3.5's template keeps thinking within a battle turn and drops it across turns, which is the handoff's default. Cap thinking equally across models and log thinking-token counts. |
| Luck | Standard rules (Showdown has no no-crit or fixed-damage rule). Reduce variance with mirrored pairs: same seed, teams swapped between sides. |
| Compute and budget | Main experiments and the model benchmark run on open models on HPRC Grace A100s (starting with Qwen/Qwen3.5-27B, as in the pruning project), using the same allocation as that project. Paid APIs only for smoke tests and small reference points: **under $10 total**. |
| Commits | Replaces handoff section 10's "no commits" rule: the building agent may commit on John's behalf, using short conventional subjects (`feat:`, `fix:`, `docs:`...) and no AI co-author line. Agent instructions stay local (gitignored). |
| Benchmark | Interest in a cross-model rating (Elo-like). Plan: Full-History Bradley-Terry plus Glicko-1, as in the PokéAgent Challenge, over mirrored pairs. |

## Phase 1 build (2026-10-02)

| Decision | Reason |
|---|---|
| Python package (`src/pokerl`) plus a small Node bridge (`bridge/bridge.js`). | The simulator is JavaScript; agents, analysis and later RL live in Python. |
| The bridge drives Showdown's `Battle` class directly, not the `simulate-battle` CLI or `BattleStream`. | Synchronous steps, `toJSON`/`fromJSON` save/load, per-player log channels, and many battles in one process. |
| `pokemon-showdown` pinned to 0.11.11 by `bridge/package-lock.json`. | Matches the handoff's verified version; results stay reproducible. |
| A seat's view = its own Showdown log channel, its request, its ally's full team, and annotated legal actions. | This is exactly what a human player in that seat gets. Opponents' exact HP and hidden lines stay hidden. The ally team is attached on every step because Showdown omits it from switch requests. |
| Legal actions are canonical Showdown choices: targetable moves always name a target; a locked move is offered once; Terastallize variants are separate actions. | In Multi Battles Showdown accepts a move with no target and picks one at random, so we always specify a target to keep each action's meaning unambiguous. |
| Each legal move carries facts: type, category, power, accuracy, priority, STAB, which Pokémon it hits, and type effectiveness against each. | Bots need them. **Whether LLM agents see these facts is an open question for John** (strategy advice vs. game facts). |
| A switch rejected because of a hidden trapping ability (Shadow Tag, Arena Trap, Magnet Pull) leaves the seat pending; the simulator then marks it trapped. | This is real Showdown behavior and reveals the same information a human would get. Logged as a rejection, not as invalid output. |
| Bridge re-links allies' shared side conditions after `fromJSON`. | **Showdown 0.11.11 bug:** in Multi Battles allies share one `sideConditions` object, but `fromJSON` restores two copies, so hazards and screens stop working for p3/p4 after a reload. Before the fix, 15/20 resumed battles matched the uninterrupted battle; after it, 100/100 (saved at turns 2 to 11) and 200/200 (saved at turn 4). The handoff's earlier test compared two reloads with each other, which do match, so it did not catch this. |
| `|t:|` timestamp lines are dropped from logs. | They are the only nondeterministic lines; dropping them makes "same seed, same battle" exact. |
| Bot randomness comes from `(battle seed, seat, step)`, not a running generator. | Saved battles resume identically, and bots never share random state. |
| Team pool `data/teams/pool-v1.json`: 32 teams from Showdown's own Multi Random Battle generator, seeds derived from the label `v1`. Each battle draws 4 teams with no species repeated. | Small fixed pool, as the handoff asks; Showdown's generator gives realistic, legal sets with levels balanced by Showdown. |
| Battles run in mirrored pairs by default. | Cancels team and seed luck within a pair. |
| Safety cap of 300 turns (battle recorded as truncated). | Multi Random Battle has no Endless Battle Clause. Never triggered in 1,250 check battles (longest average 16.8 turns). |
| Max-power bot: highest power × accuracy × STAB × type effectiveness against a foe, minus damage to its ally; ignores status moves; never Terastallizes or switches voluntarily. Random bot: uniform over all legal actions. | Simple, standard baselines. |
| Logs: one JSON line per battle with seed, team ids, policies, every step's choices, rejections, the omniscient log, Showdown's input log, and timing. | Enough to recompute any metric and to replay the battle exactly. |
| Replays use Showdown's downloadable replay format. Opening one loads Showdown's viewer script from play.pokemonshowdown.com. | Nothing is sent to Showdown and no battle connects to it, but the browser does fetch a static file from their site. **Flag for John**; the alternative is vendoring the client. |
| Win rates use a 95% Wilson interval; draws and truncations count as half a win. | Small-sample-safe interval. |
| Laptop setup uses `python -m venv` (uv is not installed locally); HPRC will use uv as the pruning project does. | Works with the plain pyproject either way. |
