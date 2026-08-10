# Scheduling the bot (Phase 6b)

The strategy decides on a 5-trading-day horizon. If nobody runs it for two
months, the account is not running that strategy any more, it is holding one
stale snapshot. Automation is what closes that gap.

Two layers do the work:

- `tradersjoy autorun` - one **guarded** decision cycle, safe to run unwatched.
- a **systemd user timer** - fires it on weekdays after the US close.

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
actually wrong. So `systemctl status` only lights up for real problems.

## Install

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

## Going live

The shipped unit runs a **dry run**. It walks every guard, refreshes data,
records the decision, and places nothing.

Leave it that way for at least a week. What you are checking:

1. It fires every weekday (`tradersjoy status` shows no gaps).
2. It skips holidays on its own.
3. **Turnover looks sane.** This is the one that matters. A top-5 ranker can ask
   for a full rotation daily; at 5 bps a side, rotating the whole book every day
   is a guaranteed loss no model this weak can outrun. If the dry runs show
   near-total daily turnover, fix that before `--execute`, not after.

Then edit `~/.config/systemd/user/tradersjoy.service`, swap the commented
`ExecStart` for the one with `--execute`, and:

```bash
systemctl --user daemon-reload
systemctl --user restart tradersjoy.timer
```

## Stopping it

```bash
touch data/HALT                          # pause; the timer still fires and logs
rm data/HALT                             # resume
systemctl --user disable --now tradersjoy.timer   # stop scheduling entirely
```

## WSL2 caveat (read this)

A systemd **user** timer only fires while the WSL instance is running. Windows
shuts WSL down when the last terminal closes, so on a laptop that sleeps or a
machine you close at night, 14:00 can pass with nothing running.

`Persistent=true` in the timer covers the common case: the run fires late, on the
next boot, and the guards decide whether that late firing is still safe to act
on. But "late" can mean the next morning, and a decision made at 09:00 with the
market about to open is not the same as one made calmly at 17:00 ET.

If you want this genuinely reliable, the options are, in increasing order of
effort:

1. **Keep WSL alive.** A Windows Task Scheduler entry at logon running
   `wsl.exe -d Ubuntu -u ghosh -e sleep infinity` keeps the instance up.
2. **Drive it from Windows.** Task Scheduler at 14:00 weekdays running
   `wsl.exe -d Ubuntu -u ghosh -e bash -lc "cd ~/projects/tradersjoy && ~/.local/bin/uv run tradersjoy autorun ..."`.
   This wakes WSL on demand, so nothing needs to stay up.
3. **Move it off this machine.** A small always-on box or a cheap VM. This is
   the only option that survives the laptop being shut.

Option 2 is the pragmatic one for a laptop. The guards make any of them safe;
they differ only in how often the run actually happens.
