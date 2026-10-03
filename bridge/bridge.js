'use strict';
// JSON-lines bridge to the Pokemon Showdown simulator.
// Each stdin line is {"id", "cmd", ...args}; each stdout line is {"id", "ok", "result"|"error"}.
// One process holds many battles. Steps are synchronous, so responses come back in request order.

const readline = require('readline');
const { Battle, Teams, Dex } = require('pokemon-showdown');
const { extractChannelMessages } = require('pokemon-showdown/dist/sim/battle');

const SEATS = ['p1', 'p2', 'p3', 'p4'];
const CHANNELS = { p1: 1, p2: 2, p3: 3, p4: 4 };
// Move targets the player must pick (mirrors the simulator's CHOOSABLE_TARGETS).
const CHOOSABLE = new Set(['normal', 'any', 'adjacentAlly', 'adjacentAllyOrSelf', 'adjacentFoe']);
const SPREAD_FOES = new Set(['allAdjacentFoes']);
const SPREAD_ALL = new Set(['allAdjacent']);

const battles = new Map();

function newBattle(args) {
  const battle = new Battle({ formatid: args.format, seed: args.seed, strictChoices: true });
  for (const seat of SEATS) {
    battle.setPlayer(seat, { name: args.names?.[seat] || seat, team: args.teams[seat] });
  }
  return battle;
}

// Timestamps are the only nondeterministic log lines; drop them so seeds reproduce logs exactly.
function cleanLines(lines) {
  return lines.filter(line => !line.startsWith('|t:|'));
}

function pokemonRef(pokemon) {
  if (!pokemon) return null;
  return {
    seat: pokemon.side.id,
    ident: `${pokemon.side.id}: ${pokemon.name}`,
    species: pokemon.species.name,
    fainted: pokemon.fainted,
  };
}

function effectiveness(battle, moveType, target) {
  if (!target || target.fainted) return null;
  if (!battle.dex.getImmunity(moveType, target)) return 0;
  return 2 ** battle.dex.getEffectiveness(moveType, target);
}

function describeMove(battle, pokemon, requestMove, index, targetLoc) {
  const move = battle.dex.moves.get(requestMove.id);
  const action = {
    choice: `move ${index}${targetLoc ? ` ${targetLoc}` : ''}`,
    kind: 'move',
    move: requestMove.move,
    move_id: move.id,
    type: move.type,
    category: move.category,
    base_power: move.basePower,
    accuracy: move.accuracy === true ? null : move.accuracy,
    priority: move.priority,
    target_type: requestMove.target || move.target,
    stab: pokemon.getTypes().includes(move.type),
    target: null,
    hits: [],
  };
  let hits = [];
  if (targetLoc) {
    const target = pokemon.getAtLoc(targetLoc);
    action.target = pokemonRef(target);
    hits = target ? [target] : [];
  } else if (SPREAD_FOES.has(action.target_type)) {
    hits = pokemon.foes();
  } else if (SPREAD_ALL.has(action.target_type)) {
    hits = [...pokemon.foes(), ...pokemon.allies()];
  }
  action.hits = hits.map(target => ({
    ...pokemonRef(target),
    is_ally: target.side === pokemon.side || target.side === pokemon.side.allySide,
    effectiveness: move.category === 'Status' ? null : effectiveness(battle, action.type, target),
  }));
  return action;
}

function switchActions(request, wantFainted) {
  const actions = [];
  request.side.pokemon.forEach((mon, i) => {
    const fainted = mon.condition.endsWith(' fnt');
    if (mon.active || fainted !== wantFainted) return;
    actions.push({
      choice: `switch ${i + 1}`,
      kind: 'switch',
      species: mon.details.split(',')[0],
      ident: mon.ident,
      condition: mon.condition,
    });
  });
  return actions;
}

// Every legal choice for a seat's current request, in canonical form.
// Targetable moves always carry an explicit target; the simulator would otherwise pick one at random.
function legalActions(battle, side) {
  const request = side.activeRequest;
  if (!request || request.wait || battle.ended || side.isChoiceDone()) return [];
  if (request.forceSwitch) {
    if (!request.forceSwitch[0]) return [{ choice: 'pass', kind: 'pass' }];
    const reviving = request.side.pokemon.some(mon => mon.active && mon.reviving);
    const actions = switchActions(request, reviving);
    return actions.length ? actions : [{ choice: 'pass', kind: 'pass' }];
  }
  const active = request.active[0];
  const pokemon = side.active[0];
  if (!active || !pokemon) return [{ choice: 'pass', kind: 'pass' }];
  const actions = [];
  // A locked move (Outrage, a charging Fly, recharge) keeps its earlier target, so offer it once.
  const locked = pokemon.getLockedMove() || pokemon.getSemiLockedMove();
  active.moves.forEach((requestMove, i) => {
    if (requestMove.disabled || requestMove.pp === 0) return;
    const targetType = requestMove.target || battle.dex.moves.get(requestMove.id).target;
    const locs = [];
    if (CHOOSABLE.has(targetType) && !locked) {
      for (const loc of [1, 2, -1, -2]) {
        if (battle.validTargetLoc(loc, pokemon, targetType)) locs.push(loc);
      }
    } else {
      locs.push(0);
    }
    for (const loc of locs) actions.push(describeMove(battle, pokemon, requestMove, i + 1, loc));
  });
  if (!active.trapped) actions.push(...switchActions(request, false));
  return actions;
}

function sideSummary(side) {
  return side ? side.getRequestData(true) : null;
}

// Advance the per-battle cursor and return everything new since the previous response.
function snapshot(id) {
  const entry = battles.get(id);
  const { battle } = entry;
  const fresh = battle.log.slice(entry.cursor).join('\n');
  entry.cursor = battle.log.length;
  const channels = extractChannelMessages(fresh, [-1, 1, 2, 3, 4]);
  const seats = {};
  for (const side of battle.sides) {
    const request = side.activeRequest;
    const legal = legalActions(battle, side);
    seats[side.id] = {
      request: request || null,
      ally: sideSummary(side.allySide),
      needs_action: legal.length > 0,
      legal,
      new_log: cleanLines(channels[CHANNELS[side.id]]),
    };
  }
  return {
    battle_id: id,
    turn: battle.turn,
    request_state: battle.requestState,
    ended: battle.ended,
    winner: battle.ended ? (battle.winner || null) : null,
    winning_side: battle.ended ? winningSide(battle) : null,
    omniscient_log: cleanLines(channels[-1]),
    seats,
  };
}

// In a multi battle the winner string is a player name; report the team (seat pair) instead.
function winningSide(battle) {
  if (!battle.winner) return null;
  const side = battle.sides.find(s => s.name === battle.winner);
  if (!side) return null;
  return ['p1', 'p3'].includes(side.id) ? 'p1p3' : 'p2p4';
}

const handlers = {
  ping() {
    return { pong: true, showdown: require('pokemon-showdown/package.json').version };
  },

  teams(args) {
    return args.seeds.map(seed => {
      const sets = Teams.generate(args.format, { seed });
      return { seed, packed: Teams.pack(sets), species: sets.map(set => set.species) };
    });
  },

  start(args) {
    if (battles.has(args.battle_id)) throw new Error(`Battle ${args.battle_id} already exists`);
    battles.set(args.battle_id, { battle: newBattle(args), cursor: 0 });
    return snapshot(args.battle_id);
  },

  // Apply each seat's choice independently. A rejected choice leaves that seat pending.
  choose(args) {
    const { battle } = getEntry(args.battle_id);
    const errors = {};
    for (const [seat, choice] of Object.entries(args.choices)) {
      try {
        battle.choose(seat, choice);
      } catch (err) {
        errors[seat] = err.message;
      }
    }
    return { ...snapshot(args.battle_id), errors };
  },

  save(args) {
    const entry = getEntry(args.battle_id);
    return { state: entry.battle.toJSON(), cursor: entry.cursor };
  },

  load(args) {
    const battle = Battle.fromJSON(args.state);
    battle.strictChoices = true;
    if (battle.gameType === 'multi') {
      // Showdown 0.11.11 bug: allies share one sideConditions object (Battle#start), but
      // fromJSON restores two copies, so hazards and screens silently stop working for p3/p4.
      battle.sides[2].sideConditions = battle.sides[0].sideConditions;
      battle.sides[3].sideConditions = battle.sides[1].sideConditions;
    }
    battles.set(args.battle_id, { battle, cursor: args.cursor });
    return snapshot(args.battle_id);
  },

  input_log(args) {
    return getEntry(args.battle_id).battle.inputLog;
  },

  close(args) {
    const entry = battles.get(args.battle_id);
    if (entry) entry.battle.destroy();
    battles.delete(args.battle_id);
    return { closed: true };
  },

  // Look up one move, ability, item or species by id in Showdown's data.
  describe(args) {
    const dex = Dex.forFormat(args.format);
    if (args.kind === 'move') {
      const move = dex.moves.get(args.id);
      return move.exists ? {
        name: move.name, type: move.type, category: move.category, base_power: move.basePower,
        accuracy: move.accuracy === true ? null : move.accuracy, priority: move.priority,
        desc: move.shortDesc || move.desc,
      } : null;
    }
    if (args.kind === 'species') {
      const species = dex.species.get(args.id);
      return species.exists ? { name: species.name, types: species.types } : null;
    }
    const table = { ability: dex.abilities, item: dex.items }[args.kind];
    if (!table) throw new Error(`Unknown kind ${args.kind}`);
    const entry = table.get(args.id);
    return entry.exists ? { name: entry.name, desc: entry.shortDesc || entry.desc } : null;
  },
};

function getEntry(id) {
  const entry = battles.get(id);
  if (!entry) throw new Error(`Unknown battle ${id}`);
  return entry;
}

const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
rl.on('line', line => {
  if (!line.trim()) return;
  let message;
  try {
    message = JSON.parse(line);
    const handler = handlers[message.cmd];
    if (!handler) throw new Error(`Unknown command ${message.cmd}`);
    const result = handler(message.args || {});
    process.stdout.write(JSON.stringify({ id: message.id, ok: true, result }) + '\n');
  } catch (err) {
    process.stdout.write(JSON.stringify({
      id: message?.id ?? null, ok: false, error: err.message, stack: err.stack,
    }) + '\n');
  }
});
