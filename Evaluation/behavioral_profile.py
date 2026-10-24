"""
Behavioral profiles for process models (Petri nets) and event logs, per:

  Kunze, Weidlich, Weske: "Behavioral Similarity - A Proper Metric",
  BPM 2011, LNCS 6896, pp. 166-181.

Implements the weak order relation and the derived behavioral profile
(strict order ~>, exclusiveness +, interleaving ||) for a Petri net and for
an event log. The similarity measures live in similarity_metric.py.

ASSUMPTIONS
--------------------------------------------------------------------
A1. Model -> weak order via the exact reachability graph (bounded nets).
    Cost is proportional to the number of reachable markings, so a hard
    cap (`max_markings`) aborts exploration with StateSpaceTooLarge instead
    of letting the process eat all RAM and get SIGKILLed.

A2. Log -> weak order via trace variants: x >_P y iff x occurs before y in
    at least one trace (occurrence counts don't matter).

A3. Activities correspond iff they have the same label.

A4. Self-pairs (x, x) are included, as Def. 4 implies.

WHAT CHANGED vs. THE PREVIOUS VERSION (results are identical)
--------------------------------------------------------------------
Petri net side
  * The old docstring promised a marking cap but none was enforced -> a
    state-space explosion ran until the OS killed the process. Now enforced.
  * Markings are compact `bytes` (one byte per place) instead of pm4py
    Marking objects + tuples of (place-name, count); firing uses
    precomputed integer deltas instead of pm4py's semantics (which rescans
    every transition and copies a Counter on each call).
  * The graph is stored as flat int arrays (CSR) instead of a dict of lists
    of tuples, and is NOT copied into networkx (which previously duplicated
    it twice: DiGraph + condensation).
  * SCCs are found with an iterative Tarjan pass that emits components in
    reverse topological order, so "labels reachable from here" is finished
    in the same pass. Label sets are int bitmasks (union = `|`).
  * The weak order is built by OR-ing one mask per label, not by looping
    over every edge x every reachable label.

Log side
  * Reads only concept:name with a streaming XML parser (no pandas frame of
    every attribute of every event). `use_pm4py=True` restores the old path.
  * Weak order per trace in O(L + A log A) with bitmasks (was O(L + A^2)).
"""

from __future__ import annotations

import gzip
import xml.etree.ElementTree as ET
from array import array
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Set, Tuple

import pm4py

Pair = Tuple[str, str]

DEFAULT_MAX_MARKINGS = 500_000_000


class StateSpaceTooLarge(RuntimeError):
    """Raised when the reachability graph exceeds the marking cap, or the
    net turns out to be unbounded (token count > 255 in one place, or a
    transition with no input places)."""


# --------------------------------------------------------------------------- #
# Behavioral profile container
# --------------------------------------------------------------------------- #

@dataclass
class BehavioralProfile:
    """Weak-order relation and the derived ~>, +, || relations for a single
    process model or log, over its full activity set."""

    activities: FrozenSet[str]
    weak_order: FrozenSet[Pair]  # x >_P y

    strict_order: FrozenSet[Pair] = field(init=False)   # x ~>_P y
    exclusive: FrozenSet[Pair] = field(init=False)      # x +_P y
    interleaving: FrozenSet[Pair] = field(init=False)   # x ||_P y

    def __post_init__(self):
        wo = self.weak_order
        strict, inter = set(), set()
        # strict / interleaving only need a pass over the (usually sparse)
        # weak order, not over all A x A pairs.
        for x, y in wo:
            if (y, x) in wo:
                inter.add((x, y))
            else:
                strict.add((x, y))
        acts = self.activities
        excl = {(x, y) for x in acts for y in acts
                if (x, y) not in wo and (y, x) not in wo}
        self.strict_order = frozenset(strict)
        self.exclusive = frozenset(excl)
        self.interleaving = frozenset(inter)

    def reverse_strict_order(self) -> FrozenSet[Pair]:
        return frozenset((y, x) for (x, y) in self.strict_order)

    def restricted_to(self, activities: Iterable[str]) -> "BehavioralProfile":
        acts = frozenset(activities) & self.activities
        wo = frozenset((x, y) for (x, y) in self.weak_order if x in acts and y in acts)
        return BehavioralProfile(activities=acts, weak_order=wo)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _pairs_from_masks(names: List[str], after: List[int]) -> FrozenSet[Pair]:
    """after[i] is a bitmask of label ids y with (names[i], y) in the relation."""
    pairs: Set[Pair] = set()
    for i, mask in enumerate(after):
        x = names[i]
        while mask:
            low = mask & -mask
            pairs.add((x, names[low.bit_length() - 1]))
            mask ^= low
    return frozenset(pairs)


# --------------------------------------------------------------------------- #
# A1: weak order from a Petri net, via exact (capped) reachability graph
# --------------------------------------------------------------------------- #

def weak_order_from_petri_net(net, im, fm=None,
                              max_markings: int = DEFAULT_MAX_MARKINGS,
                              progress_every: int = 100_000
                              ) -> Tuple[FrozenSet[str], FrozenSet[Pair]]:
    """Exact weak order of a bounded Petri net.

    x >_P y iff some reachable marking enables a transition labelled x whose
    firing leads to a marking from which some transition labelled y can still
    fire (after any number of further steps, silent ones included).
    """
    places = list(net.places)
    pidx = {p: i for i, p in enumerate(places)}
    n_places = len(places)

    labels = sorted({t.label for t in net.transitions if t.label is not None})
    label_id = {l: i for i, l in enumerate(labels)}
    bit = [1 << i for i in range(len(labels))]

    # transitions -> (pre: ((place, weight),...), delta: ((place, change),...), label id or -1)
    trans = []
    for t in net.transitions:
        need: Dict[int, int] = {}
        for a in t.in_arcs:
            i = pidx[a.source]
            need[i] = need.get(i, 0) + a.weight
        if not need:
            raise StateSpaceTooLarge(
                f"transition {t.name!r} has no input places (net is unbounded)")
        delta = {i: -w for i, w in need.items()}
        for a in t.out_arcs:
            i = pidx[a.target]
            delta[i] = delta.get(i, 0) + a.weight
        trans.append((tuple(need.items()),
                      tuple((i, d) for i, d in delta.items() if d),
                      label_id.get(t.label, -1)))

    init = bytearray(n_places)
    try:
        for p, c in im.items():
            if c > 0:
                init[pidx[p]] = c
    except ValueError:
        raise StateSpaceTooLarge("initial marking has more than 255 tokens in a place")

    # --- BFS. States get ids in discovery order and are expanded in id order,
    # so appending each state's out-edges as we go yields the CSR layout
    # (row_ptr / edge_tgt / edge_lbl) directly.
    start = bytes(init)
    state_id: Dict[bytes, int] = {start: 0}
    frontier = deque([start])
    row_ptr = array("i", [0])
    edge_tgt = array("i")
    edge_lbl = array("i")

    while frontier:
        m = frontier.popleft()
        for pre, delta, lid in trans:
            for i, w in pre:
                if m[i] < w:
                    break
            else:
                nm = bytearray(m)
                try:
                    for i, d in delta:
                        nm[i] += d
                except ValueError:
                    raise StateSpaceTooLarge(
                        "a place exceeded 255 tokens (net looks unbounded)")
                key = bytes(nm)
                sid = state_id.get(key)
                if sid is None:
                    sid = len(state_id)
                    if sid >= max_markings:
                        raise StateSpaceTooLarge(
                            f"more than {max_markings} reachable markings")
                    state_id[key] = sid
                    frontier.append(key)
                    if progress_every and sid % progress_every == 0:
                        print(f"  [behavioral_profile] explored {sid} markings so far...")
                edge_tgt.append(sid)
                edge_lbl.append(lid)
        row_ptr.append(len(edge_tgt))

    n = len(state_id)
    del state_id, frontier  # biggest structures; free before the closure pass

    # --- iterative Tarjan SCC (all states are reachable from state 0).
    # Components are completed in reverse topological order, so when a
    # component is closed every component it points to already has its mask.
    index = array("i", [-1]) * n
    low = array("i", [0]) * n
    comp = array("i", [-1]) * n
    ptr = array("i", row_ptr)          # next unread edge per node
    comp_mask: List[int] = []          # labels reachable from each component
    tstack: List[int] = [0]
    call: List[int] = [0]
    index[0] = low[0] = 0
    counter = 1

    while call:
        v = call[-1]
        p = ptr[v]
        if p < row_ptr[v + 1]:
            ptr[v] = p + 1
            w = edge_tgt[p]
            if index[w] == -1:
                index[w] = low[w] = counter
                counter += 1
                tstack.append(w)
                call.append(w)
            elif comp[w] == -1:                 # w is on the Tarjan stack
                if index[w] < low[v]:
                    low[v] = index[w]
        else:
            call.pop()
            if call:
                u = call[-1]
                if low[v] < low[u]:
                    low[u] = low[v]
            if low[v] == index[v]:              # v is the root of an SCC
                cid = len(comp_mask)
                members = []
                while True:
                    w = tstack.pop()
                    comp[w] = cid
                    members.append(w)
                    if w == v:
                        break
                mask = 0
                for w in members:
                    for e in range(row_ptr[w], row_ptr[w + 1]):
                        l = edge_lbl[e]
                        if l >= 0:
                            mask |= bit[l]
                        c = comp[edge_tgt[e]]
                        if c != cid:
                            mask |= comp_mask[c]
                comp_mask.append(mask)

    # x >_P y  <=>  y is reachable (as a label) after some edge labelled x
    after = [0] * len(labels)
    for e in range(len(edge_tgt)):
        l = edge_lbl[e]
        if l >= 0:
            after[l] |= comp_mask[comp[edge_tgt[e]]]

    return frozenset(labels), _pairs_from_masks(labels, after)


def behavioral_profile_from_pnml(path: str,
                                 max_markings: int = DEFAULT_MAX_MARKINGS
                                 ) -> BehavioralProfile:
    net, im, fm = pm4py.read_pnml(path)
    activities, wo = weak_order_from_petri_net(net, im, fm, max_markings=max_markings)
    return BehavioralProfile(activities=activities, weak_order=wo)


# --------------------------------------------------------------------------- #
# A2: weak order from an event log, via trace-variant scanning
# --------------------------------------------------------------------------- #

def weak_order_from_traces(traces: Iterable[Tuple[str, ...]]
                           ) -> Tuple[FrozenSet[str], FrozenSet[Pair]]:
    """x >_P y holds iff first_occurrence(x) < last_occurrence(y) in some
    trace. Per trace, sort activities by last occurrence, keep suffix-OR
    bitmasks, and binary-search each activity's first occurrence: the
    activities whose last occurrence lies after it are exactly a suffix.
    O(L + A log A) per trace; the result is exactly the same relation."""
    aid: Dict[str, int] = {}
    names: List[str] = []
    after: List[int] = []

    for trace in traces:
        first: Dict[str, int] = {}
        last: Dict[str, int] = {}
        for i, a in enumerate(trace):
            if a not in first:
                first[a] = i
                if a not in aid:
                    aid[a] = len(names)
                    names.append(a)
                    after.append(0)
            last[a] = i
        if not first:
            continue

        by_last = sorted(last.items(), key=lambda kv: kv[1])
        lasts = [li for _, li in by_last]
        suffix = [0] * (len(by_last) + 1)
        for k in range(len(by_last) - 1, -1, -1):
            suffix[k] = suffix[k + 1] | (1 << aid[by_last[k][0]])
        for a, f in first.items():
            after[aid[a]] |= suffix[bisect_right(lasts, f)]

    return frozenset(names), _pairs_from_masks(names, after)


def _variants_from_xes_stream(path: str, activity_key: str = "concept:name"
                              ) -> Set[Tuple[str, ...]]:
    """Distinct activity sequences, parsed as a stream so memory stays
    proportional to the number of *variants*, not events/attributes."""
    variants: Set[Tuple[str, ...]] = set()
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as f:
        context = ET.iterparse(f, events=("start", "end"))
        _, root = next(context)
        current: List[str] = []
        for ev, elem in context:
            if ev != "end":
                continue
            tag = elem.tag.rsplit("}", 1)[-1]
            if tag == "event":
                for child in elem:                  # direct attributes only
                    if child.get("key") == activity_key:
                        current.append(child.get("value"))
                        break
                elem.clear()
            elif tag == "trace":
                variants.add(tuple(current))
                current = []
                root.clear()                        # drop finished traces
    return variants


def behavioral_profile_from_xes(path: str, activity_key: str = "concept:name",
                                use_pm4py: bool = False) -> BehavioralProfile:
    if use_pm4py:
        log = pm4py.read_xes(path)
        traces = [tuple(v) for v in pm4py.get_variants(log).keys()]
    else:
        traces = _variants_from_xes_stream(path, activity_key)
    activities, wo = weak_order_from_traces(traces)
    return BehavioralProfile(activities=activities, weak_order=wo)
