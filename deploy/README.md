# Scheduling the bot (Phase 6b)

The strategy decides on a 5-trading-day horizon. If nobody runs it for two
months, the account is not running that strategy any more; it is holding one
stale snapshot. Automation is what closes that gap.

Two layers do the work:

- `tradersjoy autorun` - one **guarded** decision cycle, safe to run unwatched.
- a **scheduler** that fires it on weekdays after the US close.

There are two supported schedulers. **GitHub Actions is the recommended one.**

| | GitHub Actions | systemd timer |
|---|---|---|
| Cost | free, unlimited (public repo) | free |
| Needs your machine awake | no | yes |
| Setup | add two secrets | copy two files |
| Timing | best-effort, often late | exact |
| State | journal committed to the repo | local SQLite |

Timing precision looks like the one place systemd wins, and for most trading
automation it would be decisive. Here it is not: we decide after the close and
the orders queue for the next open, so a run at 18:00 ET and a run at 23:00 ET
place the identical trade. The session is resolved from Alpaca's calendar rather
than from the wall clock, so even a badly delayed run knows which day it is
trading.

## What the guards do

`autorun` refuses to trade rather than trading on bad state. In order:

| Guard | Refuses when | Why |
|---|---|---|
| Lock file | another run is in flight | two traders on one account is never right |
| Halt file | `data/HALT` exists | pause trading without touching timers or keys |
| Market clock | a session is in progress | today's bar is not final until the close |
| Session resolve | Alpaca's calendar returns nothing | no session means nothing to decide on |
| Duplicate | this session already traded | a retry must not re-decide and churn |
| Freshness | the store does not reach that session | stale data would re-trade a stale opinion |

Every path, including every refusal, writes a heartbeat row. That is the point:
a scheduler that quietly dies leaves a visible gap instead of looking like a
strategy that had nothing to do.

Check it any time:

```bash
uv run tradersjoy status
```

Exit codes: `0` when it traded or legitimately skipped, `1` when something was
actually wrong. So `systemctl status` and the Actions UI only light up for real
problems.

---

# Option 1: GitHub Actions (recommended)

`.github/workflows/trade.yml` runs the decision at 23:00 UTC on weekdays
(18:00 EST / 19:00 EDT, after the close in both halves of the year, so no
daylight-saving arithmetic).

## Setup

Add the paper credentials as repository secrets:

```bash
gh secret set ALPACA_API_KEY
gh secret set ALPACA_API_SECRET
```

That is the whole install. The schedule activates once the workflow is on the
default branch.

Trigger a run by hand to check it:

```bash
gh workflow run "Daily decision"
gh run watch
```

The manual trigger takes `execute` and `force` inputs, so you can test a real
placement once without editing the workflow.

## Two things this design has to handle

**The runner is ephemeral.** The bar database is deliberately *not* carried
between runs: a cold ingest of ~500 calendar days yields ~340 sessions per
ticker, comfortably above the 201 the model's features need, and takes about
eight seconds. Bars are disposable. The journal is not, so it lives in its own
small file (`state/journal.sqlite`, via `JOURNAL_DATABASE_URL`) that the workflow
commits back after every run.

**GitHub disables scheduled workflows after 60 idle days.** Only new commits
reset that timer; workflow runs, releases, and issues do not count. A bot repo
nobody commits to gets its schedule silently killed at day 60, which is exactly
the failure the heartbeat exists to catch. The daily journal commit doubles as
the keepalive, so persistence and liveness are solved by the same mechanism.

## Credentials

Alpaca issues separate keys for paper and live accounts; paper keys do not
authenticate against the live endpoint, and `AlpacaBroker` pins `paper=True`
regardless. The blast radius of these secrets is fake money. The workflow file is
public, which is fine: secrets are not exposed to it in fork pull requests.

---

# Option 2: systemd user timer

Runs on this machine. Useful as a local fallback, or if you would rather not put
credentials in CI.

```bash
mkdir -p ~/.config/systemd/user
cp deploy/tradersjoy.service deploy/tradersjoy.timer ~/.config/systemd/user/

systemctl --user daemon-reload
systemctl --user enable --now tradersjoy.timer

# let user services run without an active login session
sudo loginctl enable-linger "$USER"
```

Verify:

```bash
systemctl --user list-timers tradersjoy.timer   # next scheduled firing
systemctl --user start tradersjoy.service       # fire once, right now
journalctl --user -u tradersjoy.service -n 50   # what it did
tail -f logs/autorun.log                        # same, from the file
```

## WSL2 caveat

A systemd **user** timer only fires while the WSL instance is running. Windows
shuts WSL down when the last terminal closes, so on a laptop that sleeps or a
machine you close at night, 14:00 can pass with nothing running.

`Persistent=true` covers the common case: the run fires late, on the next boot,
and the guards decide whether that late firing is still safe to act on. But
"late" can mean the next morning.

Ways around it, in increasing order of effort:

1. **Keep WSL alive.** A Windows Task Scheduler entry at logon running
   `wsl.exe -d Ubuntu -u ghosh -e sleep infinity`.
2. **Drive it from Windows.** Task Scheduler at 14:00 weekdays running
   `wsl.exe -d Ubuntu -u ghosh -e bash -lc "cd ~/projects/tradersjoy && ~/.local/bin/uv run tradersjoy autorun ..."`.
3. **Move it off this machine.** Which is what Option 1 already does.

---

# Going live

Both schedulers ship as a **dry run**. They walk every guard, refresh data,
record the decision, and place nothing.

Leave it that way for at least a week. What you are checking:

1. It fires every weekday (`tradersjoy status`, or the repo's commit history,
   shows no gaps).
2. It skips holidays on its own.
3. **Turnover looks sane.** This is the one that matters. A top-5 ranker can ask
   for a full rotation daily; at 5 bps a side, rotating the whole book every day
   is a guaranteed loss no model this weak can outrun. If the dry runs show
   near-total daily turnover, fix that before `--execute`, not after.

Then:

- **Actions:** set `SCHEDULED_EXECUTE: "--execute"` in `.github/workflows/trade.yml`.
- **systemd:** swap the commented `ExecStart` in
  `~/.config/systemd/user/tradersjoy.service`, then `daemon-reload` and
  `restart tradersjoy.timer`.

# Stopping it

```bash
touch data/HALT                                   # local pause
rm data/HALT                                      # resume
systemctl --user disable --now tradersjoy.timer   # stop the local timer
gh workflow disable "Daily decision"              # stop the scheduled Actions run
```

Note that `data/HALT` is a *local* file, so it pauses the systemd timer but not
the Actions run. Use `gh workflow disable` for that one.
