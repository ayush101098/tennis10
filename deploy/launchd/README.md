# Keeping the production feeds alive

Both sports are fed from this machine. Nothing runs in the cloud, so if these
processes are not running, tennisalpha.in serves an empty board (tennis) and a
stale or missing feed (table tennis) — to everyone, paying or not.

Four processes:

| Process | Feeds |
|---|---|
| `sofa_proxy.py` | :3001 — the TLS-impersonating SofaScore proxy everything else reads |
| `push_sofa.py` | tennis schedules, odds and per-event live data → `/api/sofa` |
| `tabletennis.live` | 8s in-play poller → `live_predictions.json` |
| `tabletennis.push` | uploads TT artifacts → `/api/tt` |
| `tabletennis.refresh` | re-ingests and re-predicts the TT slate every 3h |

Until now these were started by hand and did not survive a reboot.

## Install

```bash
bash deploy/launchd/install.sh
```

That writes one LaunchAgent per process to `~/Library/LaunchAgents/`, loads
them, and starts them immediately. They restart on crash (`KeepAlive`) and come
back after a reboot at login (`RunAtLoad`).

## Check

```bash
launchctl list | grep tennisalpha
tail -f ~/Library/Logs/tennisalpha/push_sofa.log
```

A quicker end-to-end check — does production actually have data?

```bash
curl -s https://tennisalpha.in/api/sofa/category/3/scheduled-events/$(date +%F) \
  | python3 -c 'import sys,json; print(len(json.load(sys.stdin).get("events",[])), "ATP events")'
```

## Stop / remove

```bash
bash deploy/launchd/install.sh --uninstall
```

## Notes

- The agents run with the working directory set to the repo, so `.env` is
  picked up exactly as it is when you run the commands by hand.
- `sofa_proxy.py` must be up before the pushers are useful; the pushers simply
  log failures and retry until it is, so start order does not matter.
- Logs rotate nowhere — they are plain append files under
  `~/Library/Logs/tennisalpha/`. Truncate them if they grow.
