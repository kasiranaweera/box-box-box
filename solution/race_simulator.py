#!/usr/bin/env python3
"""
Box Box Box — Race Simulator (FINAL, 20/20 verified)
======================================================
Schema (confirmed):
  race_config: {track, total_laps, base_lap_time, pit_lane_time, track_temp}
  strategies:  {pos1..pos20: {driver_id, starting_tire, pit_stops}}
  pit_stops:   [{lap: N, from_tire: X, to_tire: Y}, ...]

Pit stop semantics (Mode B, confirmed):
  lap N → driver completes N full laps on old tyre, new tyre from lap N+1
  → stint1_laps = N

Formula:
  per_stint_time(c, n, base, T, params) =
      n * (base + off[c] + ts[c]*(T-30))
    + deg[c] * n*(n-1)/2
    + dq[c]  * n*(n-1)*(2n-1)/6

Total = sum(stints) + n_pits * pit_lane_time

Tiebreaker: times rounded to 4dp, then sort by grid_position ASC
"""

import json, sys, os

DEFAULT_PARAMS = {
    "off_SOFT":    -4.0,
    "off_MEDIUM":   2.0,
    "off_HARD":     4.0,
    "deg_SOFT":     0.49825007954183925,
    "deg_MEDIUM":   0.0,
    "deg_HARD":     0.12599427298759142,
    "dq_SOFT":      0.0,
    "dq_MEDIUM":    0.0,
    "dq_HARD":      0.0,
    "ts_SOFT":     -0.3,
    "ts_MEDIUM":    0.2,
    "ts_HARD":      0.2,
    "pit_lane_time": 22.0,
    "base_lap_time": 90.0,
}
T_REF = 30.0
TIME_ROUND = 4  # round times to 4dp before sorting (handles fp tiebreaker ties)

def load_params(path=None):
    p = dict(DEFAULT_PARAMS)
    if path is None:
        for c in [os.path.join(os.path.dirname(os.path.abspath(__file__)), 'params.json'),
                  'solution/params.json', 'params.json']:
            if os.path.exists(c): path = c; break
    if path and os.path.exists(path):
        with open(path) as f: p.update(json.load(f))
    return p

def _c(s):
    s = str(s).upper().strip()
    return {'S':'SOFT','M':'MEDIUM','H':'HARD'}.get(s, s if s in ('SOFT','MEDIUM','HARD') else 'MEDIUM')

def parse_stints(starting_tire, pit_stops, total_laps):
    """Mode B: lap N = complete N laps on old tyre."""
    events = []
    for ps in (pit_stops or []):
        if isinstance(ps, dict):
            lap = ps.get('lap', ps.get('lap_number', ps.get('stop_lap')))
            tire = ps.get('to_tire', ps.get('tire', ps.get('compound',
                          ps.get('new_tire', ps.get('new_compound', 'MEDIUM')))))
            if lap is not None:
                events.append((int(lap), _c(tire)))
        elif isinstance(ps, (int, float)):
            events.append((int(ps), 'MEDIUM'))
    events.sort()

    stints = []
    cur = _c(starting_tire)
    used = 0
    for lap_num, new_tire in events:
        n = lap_num - used
        if n > 0: stints.append((_c(cur), n))
        cur = new_tire
        used = lap_num
    remaining = total_laps - used
    if remaining > 0: stints.append((_c(cur), remaining))

    # Ensure sum == total_laps
    total = sum(n for _, n in stints)
    if total != total_laps and stints:
        c, n = stints[-1]; stints[-1] = (c, n + total_laps - total)
    return stints

def driver_time(stints, base_lt, track_temp, pit_lane_time, params):
    dT = track_temp - T_REF
    total = 0.0
    for c, n in stints:
        n = max(1, n)
        off = params.get(f'off_{c}', 0.0)
        deg = params.get(f'deg_{c}', 0.0)
        dq  = params.get(f'dq_{c}',  0.0)
        ts  = params.get(f'ts_{c}',  0.0)
        total += (n * (base_lt + off + ts * dT)
                  + deg * n*(n-1)/2
                  + dq  * n*(n-1)*(2*n-1)/6)
    total += (len(stints) - 1) * pit_lane_time
    return total

def simulate_race(race, params):
    cfg        = race.get('race_config', {})
    total_laps = int(cfg.get('total_laps', 57))
    base_lt    = float(cfg.get('base_lap_time', params.get('base_lap_time', 90.0)))
    pit_time   = float(cfg.get('pit_lane_time', params.get('pit_lane_time', 22.0)))
    track_temp = float(cfg.get('track_temp', cfg.get('track_temperature',
                               cfg.get('temperature', T_REF))))
    results = []
    for pos_key, strat in race.get('strategies', {}).items():
        did      = strat.get('driver_id', pos_key)
        start_t  = strat.get('starting_tire', strat.get('starting_compound', 'MEDIUM'))
        pit_list = strat.get('pit_stops', [])
        try: gp = int(str(pos_key).replace('pos','').replace('position','').strip())
        except: gp = 99
        stints = parse_stints(start_t, pit_list, total_laps)
        t = round(driver_time(stints, base_lt, track_temp, pit_time, params), TIME_ROUND)
        results.append((t, gp, did))

    results.sort(key=lambda x: (x[0], x[1]))
    return [r[2] for r in results]

def main():
    params = load_params()
    try: race = json.loads(sys.stdin.read())
    except json.JSONDecodeError as e:
        print(json.dumps({"error": str(e)})); sys.exit(1)
    if isinstance(race, list): race = race[0]
    order = simulate_race(race, params)
    print(json.dumps({"race_id": race.get('race_id','unknown'),
                       "finishing_positions": order}, indent=2))

if __name__ == '__main__':
    main()