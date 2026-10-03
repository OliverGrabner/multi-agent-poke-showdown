# Pokémon 2v2 LLM Teammates: Project Handoff

> Original brief from John (October 2026), kept as written. Later decisions and changes are in
> [DECISIONS.md](DECISIONS.md), which takes precedence where they differ.

**Owner:** John
**Status:** Exploratory. No paper planned yet.
**Last updated:** October 2026

---

## 1. What this project is

Two LLM agents play as teammates in Pokémon battles against another team. Each agent controls its own Pokémon. Before every turn, the two agents talk freely to plan. Later, we will train them with reinforcement learning (RL) to see whether training improves both their reasoning and their teamwork, and what new behaviors emerge.

The game format is Pokémon Showdown's **Multi Battle**: two players per side, each controlling one active Pokémon, 2 vs 2. This is an official game mode, not something we invented.

This starts as an exploration. If results become interesting, it may become a paper later. Do not optimize for a paper now.

---

## 2. What success looks like

In order:

1. A stable 4-player battle environment that runs locally.
2. LLM teammates that play legal, sensible battles and talk to each other freely.
3. Baseline numbers showing how much talking helps, before any training.
4. Later: RL training that makes the team better at winning and at working together, plus documented emergent behaviors (for example role splits, new combos, or shorthand in messages).

---

## 3. Out of scope for now

- RL training. Do not start it until phases 1 and 2 are solid.
- Team building. Use given teams.
- Public Showdown servers or human opponents. Everything runs locally.
- Writing a paper.

---

## 4. Decisions already made

### Game setup

- Use Showdown's built-in Multi Battle format (`gen9multirandombattle`). Seats p1 and p3 are one team. Seats p2 and p4 are the other. Each player has 3 Pokémon, 1 active at a time.
- Run Showdown's simulator locally. No server is needed.
- Do not use poke-env. It assumes exactly 2 players.
- Start with a small fixed pool of teams to reduce randomness. Random teams come later.
- Each agent can see its partner's full team details, because Showdown gives this to each player. Hiding the partner's team is a later experiment.
- Opponents start as simple scripted bots (for example random, and highest-power move). LLM teams against LLM teams come later.

### How the agents talk

- Before each turn, teammates talk in free text, taking turns. Which agent speaks first alternates each turn.
- Messages have no required structure. Agents can share plans, information, questions, or multi-turn strategy.
- Each message carries a yes/no **"ready to pick"** flag. Talk ends when both agents' latest messages say ready.
- There is **no planning cap**. There is only a high safety ceiling to stop runaway loops. It should almost never trigger. Log every time it does, and set its final value from pilot data.
- Each message may include private reasoning that the partner never sees.
- Messages are private to the team. Opponents never see them.
- After talking, each agent **privately picks its own move**. Picks are not binding: an agent can do something different from what it said.
- With its pick, each agent also **predicts its partner's move**. This measures alignment without forcing agreement.

### Prompts

- Both teammates get the same prompt, except for names.
- The prompt explains the rules, the talk process, and the output format.
- **No strategy advice.** Do not tell agents to focus fire, use Protect, and so on. Strategy should come from the agents.

### History

- Each agent sees the full shared history of the battle: all events, all messages, all moves.
- Default: leave out an agent's own private reasoning from past turns (keep it within the current turn). Make this a setting.

### Comparison conditions (for baselines)

1. **No talk:** each agent acts alone.
2. **One message each,** then pick.
3. **Free talk:** the main design.
4. **One LLM controls both Pokémon:** a reference for perfect coordination.

---

## 5. Verified facts (tested October 2026)

Tested with the `pokemon-showdown` npm package, version 0.11.11:

- 4-player Multi Battles run in the simulator without a server.
- Each player's turn request includes its ally's full team: moves, items, stats, ability.
- Fixed teams can be passed into the Multi Battle format.
- 300 battles with random moves: 0 crashes, 16.1 turns on average, about 19 battles per second on one CPU thread. The simulator will not be the bottleneck. The LLM will.
- Same seed plus same choices gives an identical battle.
- A battle can be saved mid-game and reloaded (`toJSON` / `fromJSON`). Reloading the same save twice and playing the same choices gives identical results.
- Changing one agent's move also changes later dice rolls. Any "what if this agent had chosen differently" analysis must average over several replays with different seeds.
- Random teams change every run unless a team seed is passed. Pass seeds for reproducibility.
- The simulator has a command-line mode (`simulate-battle`) that reads commands from stdin and writes results to stdout.
- One turn of battle events averages about 85 tokens in raw form (about 150 at the 90th percentile), measured on random-move battles. A full battle history including conversation is estimated at around 15,000 tokens.

---

## 6. Phases

### Phase 1: Environment

Build a 4-seat environment. It needs to:

- start a battle from a seed and given teams
- give each seat its own observation
- accept one action per seat and advance the battle
- report the result
- save, load, and reseed battles
- log every battle and export viewable replays
- include simple scripted bots that can fill any seat

**Done when:**

- [ ] 1,000 battles run with 0 crashes
- [ ] a highest-power-move bot clearly beats a random bot
- [ ] the same seed reproduces the same battle

### Phase 2: LLM agents and baselines

Build the agents and the talk loop from Section 4, then measure.

1. Test the talk loop with scripted fake agents first, to check that ending, the safety ceiling, and move picking work.
2. Run a short smoke test with one strong API model against random bots.
3. Run the four comparison conditions against scripted bots, using one or two strong API models and one open model that could be trained later. Start around 100 battles per condition (margin of error about ±10 points).

**Done when:**

- [ ] 0 crashes, at least 95% of agent outputs valid, and talk ends naturally on most turns
- [ ] the agents clearly beat random bots
- [ ] every comparison condition has baseline numbers

### Phase 3: RL (later, outline only)

To be planned in detail after phase 2. Current direction:

- Add one change at a time: one trainable agent with a scripted partner, then both teammates sharing one model, then self-play against a pool of past versions.
- Reward: team win or loss. Any extra reward (for example per knockout) should fade out over training, because badly designed rewards can make doing nothing the best strategy.
- If one teammate starts free-riding, add per-agent credit (for example normalizing rewards per agent, or replay-based counterfactuals).
- Save checkpoints often and keep the best one, not just the last.
- Test trained agents with partners they never trained with, to check that teamwork is general and not a private habit.
- Measure reasoning before and after training, both in-game and on a few standard reasoning tests.

---

## 7. What to log and measure

Log everything per turn: the state each agent saw, all messages, private reasoning, picks, partner predictions, and the outcome. Keep battle replays.

Starting metrics (add more as needed):

- Win rate against each opponent type
- Messages per turn, and how often the safety ceiling triggers
- **Alignment:** how often each agent correctly predicts its partner's move
- **Said vs did:** how often an agent's move matches what it told its partner
- **Rubber-stamping:** how often an agent agrees right away without adding anything
- **First-speaker effect:** how often the opener's first idea becomes the plan
- **Coordination:** wasted double-targeting, hitting your own ally, combos
- **Message style over time:** length, shorthand, new conventions
- Invalid output rate

---

## 8. Left to the building agent

Decide these yourself. Record each choice and the reason in a decisions log.

- Language, libraries, and repo layout
- How to connect to the simulator
- How observations are written (wording, layout)
- Exact prompt wording, within the rules in Section 4
- Output format details, parsing, and retry rules
- The safety ceiling value for the pilot
- Which models to test first
- Team pool contents
- Test plan
- Log format and replay viewing
- Anything else not covered here

---

## 9. Risks to watch

- Agents looping or repeating themselves (documented in CAMEL and MAST)
- Rubber-stamping: one agent always agrees
- First-speaker bias: LLMs tend to favor the first offer they see
- Invalid moves or malformed output
- Cost growing with conversation length
- Later, in RL: agents learning private habits that only work with their own copy, free-riding, and reward hacking

---

## 10. Working rules

- **Do not make git commits.** John makes all commits himself. If unsure, ask.
- Run everything locally. Never connect to public Showdown servers.
- Agree on an API budget with John before running large batches. Rough estimate: about $1.30 per team per battle with free talk, before prompt caching, at an assumed $1 per million input tokens and $5 per million output tokens.
- Ask John before changing any decision in Section 4.
- Report progress in short, plain language with numbers.

---

## 11. References

- PokéAgent Challenge: https://arxiv.org/abs/2603.15563
- VGC-Bench: https://arxiv.org/abs/2506.10326
- SPIRAL (self-play on games improves reasoning): https://arxiv.org/abs/2506.24119
- MAGRPO (RL for LLM collaboration): https://arxiv.org/abs/2508.04652
- Dr. MAS (stable multi-agent RL for LLMs): https://arxiv.org/abs/2602.08847
- Lazy agents in multi-agent LLM RL: https://arxiv.org/abs/2511.02303
- GRPO and multi-agent coordination (dining philosophers): https://arxiv.org/abs/2606.07845
- PillagerBench (team vs team LLM agents): https://arxiv.org/abs/2509.06235
- Magentic Marketplace (first-offer bias): https://arxiv.org/abs/2510.25779
- CAMEL (agent conversation loops): https://arxiv.org/abs/2303.17760
- MAST (multi-agent failure types): https://arxiv.org/abs/2503.13657
- Pokémon Showdown simulator: https://github.com/smogon/pokemon-showdown
