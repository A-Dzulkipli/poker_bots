"use strict";

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const SUITS = { s: "♠", h: "♥", d: "♦", c: "♣" };
const STREET_BOARD = { preflop: 0, flop: 3, turn: 4, river: 5 };
const CATEGORY_BASE = 13 ** 5;
const CATEGORIES = ["high card", "pair", "two pair", "three of a kind", "straight", "flush", "full house", "four of a kind", "straight flush"];

async function api(method, path, body) {
  const response = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || response.statusText);
  return data;
}

function shortName(name) {
  return name === "human" ? "You" : name.replace(/^(phase_[^/]+|builtin)\//, "");
}

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

// cards and table

function cardEl(card) {
  if (card === "back") return el("span", "c back");
  if (!card) return el("span", "c empty");
  const suit = card[1];
  const node = el("span", "c" + (suit === "h" || suit === "d" ? " red" : ""));
  node.append(el("span", "r", card[0] === "T" ? "10" : card[0]), el("span", "s", SUITS[suit]));
  return node;
}

function seatEl(seat, view) {
  const node = el("div", "seat");
  if (!seat.inHand) node.classList.add("folded");
  if (seat.toAct) node.classList.add("to-act");
  if (seat.winner) node.classList.add("winner");
  const badges = el("div", "badges");
  if (seat.seat === view.button) badges.append(el("span", "badge btn", "D"));
  if (seat.seat === view.sb) badges.append(el("span", "badge", "SB"));
  if (seat.seat === view.bb) badges.append(el("span", "badge", "BB"));
  node.append(badges, el("div", "name", `${shortName(seat.name)} · seat ${seat.seat}`));
  const cards = el("div", "cards");
  if (seat.cards) seat.cards.forEach((c) => cards.append(cardEl(c)));
  else if (seat.inHand) cards.append(cardEl("back"), cardEl("back"));
  node.append(cards, el("div", "stack", `${seat.stack} chips`));
  node.append(el("div", "rank", seat.rank || ""));
  node.append(el("div", "bet", seat.bet ? `bet ${seat.bet}` : seat.note || ""));
  return node;
}

function renderTable(container, view) {
  container.replaceChildren();
  const seats = view.seats.filter((s) => s.seated !== false);
  const hero = view.heroSeat;
  const others = hero === null || hero === undefined
    ? seats
    : seats.filter((s) => s.seat !== hero).sort((a, b) => ((a.seat - hero + 99) % 99) - ((b.seat - hero + 99) % 99));
  const top = el("div", "seats");
  others.forEach((s) => top.append(seatEl(s, view)));
  const center = el("div", "center");
  const board = el("div", "cards board");
  for (let i = 0; i < 5; i++) board.append(cardEl(view.board[i]));
  center.append(el("div", "street", view.street || ""), board, el("div", "pot", `Pot ${view.pot}`));
  container.append(top, center);
  if (hero !== null && hero !== undefined) {
    const bottom = el("div", "seats");
    seats.filter((s) => s.seat === hero).forEach((s) => bottom.append(seatEl(s, view)));
    container.append(bottom);
  }
}

// action log shared by play and replay

function describeActions(actions, names) {
  let street = null, contrib = {}, bet = 0;
  return actions.map((a) => {
    const lines = { street: a.street !== street ? a.street : null };
    if (a.street !== street) { street = a.street; contrib = {}; bet = 0; }
    const before = contrib[a.seat] || 0;
    const value = a.value_applied;
    let text;
    if (a.type === "small_blind") text = `posts small blind ${value}`;
    else if (a.type === "big_blind") text = `posts big blind ${value}`;
    else if (a.type === "fold") text = "folds";
    else if (a.type === "check_call") text = value === before ? "checks" : `calls ${value - before}`;
    else if (a.type === "raise") text = bet === 0 ? `bets ${value}` : `raises to ${value}`;
    else text = a.type;
    if (a.is_all_in) text += " (all-in)";
    if (a.type_requested && a.type_requested !== a.type && !a.type.endsWith("blind")) {
      text += ` [asked ${a.type_requested.replace("check_call", "check/call")}${a.type_requested === "raise" ? " to " + a.value_requested : ""}]`;
    }
    contrib[a.seat] = value;
    bet = Math.max(bet, value);
    lines.text = `${shortName(names[a.seat])}: ${text}`;
    return lines;
  });
}

function renderLog(list, actions, names, current) {
  list.replaceChildren();
  describeActions(actions, names).forEach((line, i) => {
    if (line.street) list.append(el("li", "street-head", line.street));
    const item = el("li", "", line.text);
    if (current !== undefined) {
      if (i === current - 1) item.classList.add("current");
      if (i >= current) item.classList.add("future");
    }
    list.append(item);
  });
  const mark = list.querySelector(".current");
  (mark || list.lastElementChild)?.scrollIntoView({ block: "nearest" });
}

function signed(n) { return (n > 0 ? "+" : "") + n; }

// tabs

document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === tab.dataset.tab));
  if (tab.dataset.tab === "replay") loadReplaySources();
}));

// play: setup

let botList = { finals: [], versions: [] };

function botOptions(select, selected) {
  select.replaceChildren();
  const group = (label, names) => {
    const g = el("optgroup");
    g.label = label;
    names.forEach((name) => { const o = el("option", "", name); o.value = name; g.append(o); });
    select.append(g);
  };
  group("Finals", botList.finals);
  if ($("show-versions").checked) group("Versions", botList.versions);
  if (selected && [...select.options].some((o) => o.value === selected)) select.value = selected;
}

function renderSeatBots() {
  const count = Number($("seat-count").value);
  const box = $("seat-bots");
  const previous = [...box.querySelectorAll("select")].map((s) => s.value);
  box.replaceChildren();
  if (count === 2) return;
  for (let seat = 1; seat < count; seat++) {
    const label = el("label", "", `Seat ${seat}`);
    const select = el("select");
    botOptions(select, previous[seat - 1] || $("fill-bot").value);
    label.append(select);
    box.append(label);
  }
}

function seatBots() {
  const count = Number($("seat-count").value);
  if (count === 2) return [$("fill-bot").value];
  return [...$("seat-bots").querySelectorAll("select")].map((s) => s.value);
}

async function initSetup() {
  for (let n = 2; n <= 9; n++) {
    const o = el("option", "", n === 2 ? "2 (heads-up)" : String(n));
    o.value = n;
    $("seat-count").append(o);
  }
  const [bots, config] = await Promise.all([api("GET", "/api/bots"), api("GET", "/api/config")]);
  botList = bots;
  botOptions($("fill-bot"));
  $("bot-delay").value = config.bot_delay_ms;
  $("seat-count").addEventListener("change", () => {
    $("fill-bot").parentElement.firstChild.textContent = Number($("seat-count").value) === 2 ? "Opponent " : "Fill with ";
    renderSeatBots();
  });
  $("fill-bot").addEventListener("change", () => {
    $("seat-bots").querySelectorAll("select").forEach((s) => botOptions(s, $("fill-bot").value));
  });
  $("show-versions").addEventListener("change", () => {
    botOptions($("fill-bot"), $("fill-bot").value);
    $("seat-bots").querySelectorAll("select").forEach((s) => botOptions(s, s.value));
  });
  try {
    const running = await api("GET", "/api/play/state");
    showGame(running.state);
    loop();
  } catch { /* no session running */ }
}

$("setup").addEventListener("submit", async (event) => {
  event.preventDefault();
  $("start").disabled = true;
  $("setup-msg").className = "msg";
  $("setup-msg").textContent = "Building the table (compiling the bots, a few seconds)…";
  try {
    const seed = $("seed").value === "" ? null : Number($("seed").value);
    const r = await api("POST", "/api/play/new" + cardsQuery(), {
      seat_bots: seatBots(), small_blind: Number($("sb").value), big_blind: Number($("bb").value),
      starting_bb: Number($("stack-bb").value), seed,
    });
    $("setup-msg").textContent = "";
    $("play-log").replaceChildren();
    showGame(r.state);
    loop();
  } catch (error) {
    $("setup-msg").className = "msg error";
    $("setup-msg").textContent = error.message;
  } finally {
    $("start").disabled = false;
  }
});

// play: table

let looping = false;
let dealNext = null;
let lastState = null;

const cardsQuery = () => ($("show-bot-cards").checked ? "?show_bot_cards=true" : "");
const delay = () => Math.max(0, Number($("bot-delay").value) || 0);

function showGame(state) {
  $("setup").hidden = true;
  $("game").hidden = false;
  render(state);
}

function render(state) {
  lastState = state;
  const net = state.net;
  $("session-info").textContent =
    `Hand ${state.hand + 1} · ${state.hands_played} played · you ${signed(net)} chips (${signed(+(net / state.big_blind).toFixed(1))} bb)`;
  renderTable($("play-table"), {
    heroSeat: state.human_seat,
    button: state.button, sb: state.sb_seat, bb: state.bb_seat,
    board: state.board, pot: state.pot, street: state.street,
    seats: state.seats.map((s) => ({
      seat: s.seat, name: s.name, stack: s.stack, bet: s.contributed, cards: s.cards,
      seated: s.seated, inHand: s.in_hand, toAct: s.seat === state.current_player,
    })),
  });
  renderLog($("play-log"), state.actions, state.seats.map((s) => s.name));
  renderControls(state);
}

function renderHandOver(state) {
  const hand = state.last_hand;
  renderTable($("play-table"), {
    heroSeat: state.human_seat, button: hand.button, board: hand.board, pot: hand.pot,
    street: hand.showdown ? "showdown" : "hand over",
    seats: hand.seats.map((s) => ({
      seat: s.seat, name: s.name, stack: s.stack, cards: s.cards, inHand: s.in_hand, rank: s.rank,
      winner: s.payout > 0, note: s.payout > 0 ? `wins ${s.payout}` : "",
    })),
  });
  const box = $("result");
  box.replaceChildren(el("div", "line", `Hand ${hand.hand + 1} finished.`));
  hand.seats.forEach((s) => {
    const line = el("div", "line");
    line.append(`${shortName(s.name)}: `, el("span", s.net > 0 ? "pos" : s.net < 0 ? "neg" : "", signed(s.net)));
    if (s.rank) line.append(` with ${s.rank}`);
    if (s.rebuy) line.append(` · rebuys ${s.rebuy}`);
    if (s.cash_out) line.append(` · cashes out ${s.cash_out}`);
    box.append(line);
  });
  box.hidden = false;
}

function renderControls(state) {
  const box = $("controls");
  box.replaceChildren();
  if (state.finished) {
    box.append(el("span", "waiting", "The table is finished (one player left)."));
    return;
  }
  const c = state.controls;
  if (!c) {
    if (state.current_player !== null && state.current_player !== state.human_seat) {
      box.append(el("span", "waiting", `${shortName(state.seats[state.current_player].name)} is thinking…`));
    }
    return;
  }
  const button = (text, cls, onClick, key) => {
    const b = el("button", cls, text);
    if (key) b.title = `key: ${key}`;
    b.addEventListener("click", onClick);
    box.append(b);
    return b;
  };
  button("Fold", "", () => act("fold"), "f");
  button(c.to_call === 0 ? "Check" : `Call ${c.to_call}`, "", () => act("check_call"), "c");
  if (!c.can_raise) return;
  if (c.raise_min === c.raise_max) {
    button(`All-in to ${c.raise_max}`, "primary", () => act("raise", c.raise_max), "r");
    return;
  }
  const slider = el("input");
  Object.assign(slider, { type: "range", min: c.raise_min, max: c.raise_max, step: 1, value: c.raise_min });
  const amount = el("input", "raise-to");
  Object.assign(amount, { type: "number", min: c.raise_min, max: c.raise_max, value: c.raise_min });
  const clamp = (v) => Math.min(c.raise_max, Math.max(c.raise_min, Math.round(v)));
  slider.addEventListener("input", () => { amount.value = slider.value; });
  amount.addEventListener("input", () => { slider.value = amount.value; });
  const verb = c.current_bet === 0 ? "Bet" : "Raise to";
  const go = button(verb, "primary", () => act("raise", clamp(Number(amount.value))), "r");
  box.insertBefore(slider, go);
  box.insertBefore(amount, go);
  const potRaise = (fraction) => clamp(c.current_bet + fraction * (state.pot + c.to_call));
  [["min", c.raise_min], ["½ pot", potRaise(0.5)], ["pot", potRaise(1)], ["all-in", c.raise_max]].forEach(([label, v]) =>
    button(label, "", () => { slider.value = v; amount.value = v; }));
}

async function act(type, value = 0) {
  $("controls").replaceChildren();
  try {
    const r = await api("POST", "/api/play/act" + cardsQuery(), { type, value });
    render(r.state);
  } catch (error) {
    $("controls").append(el("span", "msg error", error.message));
    return;
  }
  loop();
}

async function loop() {
  if (looping) return;
  looping = true;
  try {
    for (;;) {
      const r = await api("POST", "/api/play/advance" + cardsQuery());
      if (r.event === "hand_over") {
        renderHandOver(r.state);
        $("controls").replaceChildren();
        if ($("auto-deal").checked) {
          await sleep(Math.max(1500, 3 * delay()));
        } else {
          await new Promise((resolve) => {
            dealNext = resolve;
            const b = el("button", "primary", "Deal next hand");
            b.title = "key: space or n";
            b.addEventListener("click", resolve);
            $("controls").append(b);
          });
          dealNext = null;
        }
        $("result").hidden = true;
        render(r.state);
        continue;
      }
      render(r.state);
      if (r.event === "human" || r.event === "finished") break;
      await sleep(delay());
    }
  } catch (error) {
    $("controls").replaceChildren(el("span", "msg error", error.message));
  } finally {
    looping = false;
  }
}

$("show-bot-cards").addEventListener("change", async () => {
  if (looping && !dealNext) return;   // the next step redraws with the new setting
  const r = await api("GET", "/api/play/state" + cardsQuery());
  if (!dealNext) render(r.state);
});

$("end-session").addEventListener("click", async () => {
  if (dealNext) dealNext();
  const r = await api("POST", "/api/play/end");
  $("game").hidden = true;
  $("setup").hidden = false;
  $("result").hidden = true;
  $("setup-msg").className = "msg";
  $("setup-msg").textContent = `Saved ${r.hands} hands to ui/sessions/${r.session}/`;
});

document.addEventListener("keydown", (event) => {
  if (event.target.tagName === "INPUT" || event.target.tagName === "SELECT") return;
  if ($("play").classList.contains("active")) {
    if (dealNext && (event.key === " " || event.key === "n")) { event.preventDefault(); dealNext(); return; }
    const c = lastState?.controls;
    if (!c || looping) return;
    if (event.key === "f") act("fold");
    else if (event.key === "c") act("check_call");
    else if (event.key === "r") $("controls").querySelector("button.primary")?.click();
  } else if (replay.data) {
    const keys = { ArrowLeft: () => stepTo(replay.step - 1), ArrowRight: () => stepTo(replay.step + 1),
                   ArrowUp: () => handTo(replay.hand - 1), ArrowDown: () => handTo(replay.hand + 1) };
    if (keys[event.key]) { event.preventDefault(); keys[event.key](); }
  }
});

// replay

const replay = { data: null, hand: 0, step: 0, hero: null, sources: null };

async function loadReplaySources() {
  const source = $("replay-source").value;
  const pick = $("replay-pick");
  pick.replaceChildren();
  $("replay-msg").textContent = "";
  $("replay-table-label").hidden = source !== "tournaments";
  $("replay-pick-label").firstChild.textContent = source === "tournaments" ? "Block " : "Session ";
  if (source === "sessions") {
    const sessions = await api("GET", "/api/replay/sessions");
    replay.sources = sessions;
    sessions.forEach((s, i) => {
      const opponents = [...new Set(s.seats.slice(1).map(shortName))].join(", ");
      const net = s.human_net === null ? "" : ` · you ${signed(s.human_net)}`;
      const o = el("option", "", `${s.created_at.slice(0, 16).replace("T", " ")} · vs ${opponents} · ${s.hands} hands${net}${s.closed ? "" : " · open"}`);
      o.value = i;
      pick.append(o);
    });
    if (!sessions.length) $("replay-msg").textContent = "No saved sessions yet.";
  } else {
    const blocks = await api("GET", "/api/replay/tournaments");
    replay.sources = blocks;
    blocks.forEach((b, i) => {
      const o = el("option", "", `${b.experiment} · ${b.pair} · seed ${b.seed} (${b.tables} tables × ${b.hands} hands)${b.format_version < 2 ? " · from seed" : ""}${b.replayable ? "" : " · engine changed"}`);
      o.value = i;
      o.disabled = !b.replayable;
      pick.append(o);
    });
    const first = blocks.findIndex((b) => b.replayable);
    if (first >= 0) pick.value = first;
    if (blocks.length && first < 0) $("replay-msg").textContent = "Every block was run on a different engine version, so none can be regenerated.";
    updateTableLimit();
  }
}

function updateTableLimit() {
  const block = replay.sources?.[Number($("replay-pick").value)];
  if (block && $("replay-source").value === "tournaments") $("replay-table").max = block.tables - 1;
}

$("replay-source").addEventListener("change", loadReplaySources);
$("replay-pick").addEventListener("change", updateTableLimit);
$("replay-show-all").addEventListener("change", () => replay.data && drawReplay());

$("replay-load").addEventListener("click", async () => {
  const source = $("replay-source").value;
  const item = replay.sources?.[Number($("replay-pick").value)];
  if (!item) return;
  $("replay-load").disabled = true;
  $("replay-msg").className = "msg";
  $("replay-msg").textContent = source === "tournaments" && item.format_version < 2
    ? "This block stores outcomes only, so it is being replayed from its seed (can take half a minute)…" : "Loading…";
  try {
    let data;
    if (source === "sessions") {
      data = await api("GET", `/api/replay/session/${encodeURIComponent(item.session)}`);
      replay.hero = 0;
    } else {
      data = await api("GET", `/api/replay/tournament/${item.experiment}/${item.pair}/${item.seed}/${Number($("replay-table").value)}`);
      replay.hero = null;
    }
    replay.data = data;
    $("replay-warning").hidden = !data.warning;
    $("replay-warning").textContent = data.warning || "";
    const select = $("r-hand");
    select.replaceChildren();
    data.hands.forEach((h, i) => {
      const hero = replay.hero === null ? "" : ` · ${signed(h.ending_stacks[0] - h.starting_stacks[0])}`;
      const flag = data.mismatched_hands?.includes(h.hand_index) ? " ⚠" : "";
      const o = el("option", "", `#${h.hand_index + 1} pot ${h.payouts.reduce((a, b) => a + b, 0)}${hero}${flag}`);
      o.value = i;
      select.append(o);
    });
    $("replay-msg").textContent = data.hands.length ? "" : "This session has no finished hands.";
    $("replay-view").hidden = !data.hands.length;
    if (data.hands.length) handTo(0);
  } catch (error) {
    $("replay-msg").className = "msg error";
    $("replay-msg").textContent = error.message;
  } finally {
    $("replay-load").disabled = false;
  }
});

$("r-hand").addEventListener("change", () => handTo(Number($("r-hand").value)));
$("r-prev-hand").addEventListener("click", () => handTo(replay.hand - 1));
$("r-next-hand").addEventListener("click", () => handTo(replay.hand + 1));
$("r-back").addEventListener("click", () => stepTo(replay.step - 1));
$("r-fwd").addEventListener("click", () => stepTo(replay.step + 1));

function handTo(i) {
  if (!replay.data || i < 0 || i >= replay.data.hands.length) return;
  replay.hand = i;
  replay.step = 0;
  $("r-hand").value = i;
  drawReplay();
}

function stepTo(step) {
  const hand = replay.data.hands[replay.hand];
  if (step < 0) return handTo(replay.hand - 1);
  if (step > hand.actions.length) return handTo(replay.hand + 1);
  replay.step = step;
  drawReplay();
}

function drawReplay() {
  const { data, step, hero } = replay;
  const hand = data.hands[replay.hand];
  const names = data.seat_names;
  const actions = hand.actions;
  const done = step === actions.length;
  const showAll = $("replay-show-all").checked;

  const committed = Array(names.length).fill(0);
  const folded = new Set();
  let contrib = {}, street = null;
  actions.slice(0, step).forEach((a) => {
    if (a.street !== street) { street = a.street; contrib = {}; }
    committed[a.seat] += a.value_applied - (contrib[a.seat] || 0);
    contrib[a.seat] = a.value_applied;
    if (a.type === "fold") folded.add(a.seat);
  });
  const nextStreet = done ? null : actions[step].street;
  if (nextStreet !== street) contrib = {};   // the street closed: bets are in the pot
  const boardCount = done ? hand.board_dealt : STREET_BOARD[nextStreet];

  renderTable($("replay-table-view"), {
    heroSeat: hero, button: hand.button, sb: hand.sb_seat, bb: hand.bb_seat,
    board: hand.board.slice(0, boardCount),
    pot: committed.reduce((a, b) => a + b, 0),
    street: done ? (hand.went_to_showdown ? "showdown" : "hand over") : nextStreet,
    seats: names.map((name, seat) => {
      const shown = showAll || seat === hero || (done && hand.hand_ranks[seat] !== null);
      const rank = done && hand.hand_ranks[seat] !== null ? CATEGORIES[Math.floor(hand.hand_ranks[seat] / CATEGORY_BASE)] : "";
      return {
        seat, name, seated: hand.seated[seat], inHand: !folded.has(seat),
        cards: shown ? hand.hole_cards[seat] : null,
        stack: done ? hand.ending_stacks[seat] : hand.starting_stacks[seat] - committed[seat],
        bet: done ? 0 : contrib[seat] || 0, rank,
        toAct: !done && actions[step].seat === seat,
        winner: done && hand.payouts[seat] > 0,
        note: done && hand.payouts[seat] > 0 ? `wins ${hand.payouts[seat]}` : "",
      };
    }),
  });
  renderLog($("replay-log"), actions, names, step);
  $("r-position").textContent = `hand ${replay.hand + 1}/${data.hands.length} · step ${step}/${actions.length}`;

  const box = $("replay-result");
  box.hidden = !done;
  if (done) {
    box.replaceChildren();
    names.forEach((name, seat) => {
      if (!hand.seated[seat]) return;
      const net = hand.ending_stacks[seat] - hand.starting_stacks[seat];
      const line = el("div", "line");
      line.append(`${shortName(name)} (seat ${seat}): `, el("span", net > 0 ? "pos" : net < 0 ? "neg" : "", signed(net)));
      if (hand.rebuys[seat]) line.append(` · rebuys ${hand.rebuys[seat]}`);
      if (hand.cash_outs[seat]) line.append(` · cashes out ${hand.cash_outs[seat]}`);
      box.append(line);
    });
  }
}

initSetup().catch((error) => { $("setup-msg").className = "msg error"; $("setup-msg").textContent = error.message; });
