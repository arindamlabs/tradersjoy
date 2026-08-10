# Operations runbook

Day-to-day operation of the deployed bot: how to tell it is alive, how to stop
it, how to rotate credentials, and what to do when something looks wrong.

For *what the system is and why*, see the [README](../README.md). For scheduler
setup and the trade-offs between GitHub Actions and systemd, see
[deploy/README.md](../deploy/README.md).

## Where the bot actually runs

**GitHub Actions**, weekdays at 23:00 UTC (18:00 EST / 19:00 EDT), defined in
`.github/workflows/trade.yml`. It is **live**: `SCHEDULED_EXECUTE: "--execute"`,
so the scheduled run places real orders on the Alpaca **paper** account.

Your laptop is not involved. Nothing here depends on WSL being awake. The
systemd units in `deploy/` are an unused alternative, not installed.

Consequence worth internalising: **`data/HALT` does not stop the deployed bot.**
That file is a local switch for a local runner. To stop the cloud bot see
[Emergency stop](#emergency-stop).

---

## Quick reference

```bash
git pull && uv run tradersjoy status      # is the bot alive and what did it do?
gh run list --workflow="Daily decision"   # did GitHub actually fire it?
gh workflow disable "Daily decision"      # STOP the bot
gh workflow enable  "Daily decision"      # resume
gh workflow run     "Daily decision"      # run once now (dry run by default)
```

---

## Is it running?

Two independent questions, and they fail differently.

**1. Did the scheduler fire?** This is the one that fails silently.

```bash
gh run list --workflow="Daily decision" --limit 10
```

Look at the trigger column. `schedule` means the cron fired on its own.
`workflow_dispatch` means a human pressed the button. A healthy bot produces one
`schedule` row per weekday.

**2. What did it decide?** The heartbeat, which the run commits back to the repo.

```bash
git pull
uv run tradersjoy status
```

`status` reads `state/journal.sqlite`, the deployment's journal, and prints which
source it used. **It is only as current as your last `git pull`.** Add `--local`
to inspect this machine's own runs instead.

You can also just read the git log; the bot commits after every run:

```bash
git log --oneline --grep="chore(journal)" -10
```

### Expected states

| Status | Meaning |
|---|---|
| `traded` | Placed orders. Normal on a rebalance day. |
| `no_orders` | Ran fine, strategy wanted nothing. Normal mid-week: the ml strategy only rebalances on the first session of each week. |
| `dry_run` | Decided but placed nothing. Expected only from a manual dispatch. |
| `skipped` | A guard declined: weekend, holiday, market open, already traded, halted. Not a problem. |
| `failed` | Something was wrong: stale data, broken feed, overlapping run. Exit code 1, so the Actions run shows red. |

A `failed` run means the bot **refused to trade**, not that it traded badly. The
guards are designed to prefer doing nothing.

---

## Emergency stop

**Stop the bot from trading, now:**

```bash
gh workflow disable "Daily decision"
```

Takes effect immediately; no further scheduled runs. Resume with
`gh workflow enable "Daily decision"`.

**Softer option, keep it running but stop it trading:** set
`SCHEDULED_EXECUTE: ""` in `.github/workflows/trade.yml`, commit, push. It keeps
firing, deciding, and journalling, but places nothing.

**Positions are not touched by either.** Stopping the bot freezes the book as-is;
it does not sell. To flatten, either sell manually in the
[Alpaca dashboard](https://app.alpaca.markets/paper/dashboard/overview) or reset
the account (below).

**Reset the paper account** (wipes positions and history, restores the starting
balance): the Reset button in the Alpaca paper dashboard. Free and unlimited.
Deliberately **not** wired into the code, so nothing can wipe the account
automatically. Two follow-ups if you do it:

- If you reset to a balance other than $100,000, update `starting_equity` in
  `LiveTrader` and `STARTING_EQUITY` in `dashboard/app.py`, or P/L reporting will
  be wrong.
- The run journal survives, because it is our file rather than Alpaca's. The
  equity curve will show the discontinuity.

---

## Credentials and logins

### Alpaca

Keys come from the [paper dashboard](https://app.alpaca.markets/paper/dashboard/overview),
under "API Keys". Paper and live accounts issue **separate** keys; paper keys do
not authenticate against the live endpoint, and `AlpacaBroker` pins `paper=True`
regardless.

Used in two places, which must be updated together:

```bash
# 1. locally, in .env (gitignored)
nano .env            # ALPACA_API_KEY, ALPACA_API_SECRET

# 2. in GitHub Actions secrets, which is what the bot uses
gh secret set ALPACA_API_KEY
gh secret set ALPACA_API_SECRET
gh secret list       # confirm both, and their update times
```

To rotate without echoing the value into your shell history:

```bash
grep '^ALPACA_API_KEY=' .env | cut -d= -f2- | tr -d '\n' | gh secret set ALPACA_API_KEY
```

Verify the new keys work before relying on them:

```bash
uv run tradersjoy trade --strategy buyhold --no-journal   # dry run, reads the account
```

### GitHub

```bash
gh auth status       # needs scopes: repo, workflow
gh auth login        # if expired
```

The `workflow` scope is required to push changes to anything under
`.github/workflows/`. Without it the push is rejected with a confusing error.

---

## Routine maintenance

### Every weekday (10 seconds)

Nothing required. The bot commits its own record. Glance at the repo's commit
history if you want reassurance.

### Weekly

```bash
git pull && uv run tradersjoy status
```

Confirm one run per weekday and no `failed` rows.

### Quarterly: retrain

The model rots. It is trained on history up to a point and has no idea time has
passed since.

```bash
uv run tradersjoy ingest        # refresh bars
uv run tradersjoy train         # walk-forward + save data/models/ml.joblib
uv run tradersjoy evaluate --horizon 5   # honest out-of-sample scorecard
git add data/models/ml.joblib && git commit -m "chore: retrain model" && git push
```

The model artifact is **committed on purpose**: the Actions runner is ephemeral
and has to get it from somewhere. Pushing it is what deploys it.

Read the `train` output honestly. If AUC drifts below ~0.50 the model has
stopped ranking better than chance and you should consider disabling execution
rather than shipping it.

### Every ~60 days: the GitHub inactivity trap

GitHub **disables scheduled workflows after 60 days with no new commits**.
Workflow runs do not count; only commits reset the timer.

The bot's own daily journal commit normally handles this. But if it stops for any
reason, the 60-day clock starts, and it will silently switch itself off. If
`gh run list` shows no `schedule` rows and the workflow looks idle:

```bash
gh workflow enable "Daily decision"
```

### Occasionally: dependencies

```bash
uv sync --upgrade
uv run pytest && uv run ruff check .
git add uv.lock && git commit -m "chore: update dependencies" && git push
```

CI runs the same checks on push, so a green CI badge means the runner agrees.

---

## Troubleshooting by symptom

### "No `schedule` runs, only `workflow_dispatch`"

The cron is not firing. In order of likelihood:

1. The workflow was auto-disabled after 60 idle days -> `gh workflow enable "Daily decision"`.
2. GitHub is delayed. Scheduled triggers are best-effort and routinely late, occasionally dropped. One missing day is not a fault; three in a row is.
3. The workflow file is not on the default branch. Schedules only run from the default branch.

### Run status `failed`, reason "stale data"

The store did not reach the session being traded, so the bot refused rather than
re-trading a stale opinion. Almost always yfinance being down or lagging. It
self-corrects on the next run. If it persists for days, check yfinance manually:

```bash
uv run tradersjoy ingest
```

### Run status `failed`, reason "another run holds autorun.lock"

Two runs overlapped. The `concurrency` group in the workflow should prevent this;
if it recurs, check for a hung run in the Actions tab and cancel it.

### Run status `skipped`, reason "market is still open"

The run fired before the close. Harmless, it declines to decide on an unfinished
bar. If it happens every day, the cron time is wrong for the season.

### Run status `skipped`, reason "already traded session X"

A second run for a session already traded. The duplicate guard working correctly.
Override deliberately with a dispatch and `force: true`.

### "The equity curve is flat / it never trades"

Expected mid-week. The ml strategy rebalances only on the **first trading session
of each week**, so Tuesday through Friday legitimately produce `no_orders`.

### "`status` says nothing has fired in days" but Actions looks fine

Your local copy is stale. `git pull` first. `status` reads a committed file.

### Push rejected: "remote contains work you do not have"

The bot committed a journal row while you were working:

```bash
git pull --rebase && git push
```

Do not force-push; you would drop the bot's record.

---

## Local development

The local journal is separate from the deployment's, deliberately, so local
experiments never conflict with the bot's committed `state/journal.sqlite`.

```bash
uv run pytest                        # 86 tests, no network
uv run ruff check .

# safe experiments: dry run, and do not touch the journal
uv run tradersjoy trade --strategy ml --model data/models/ml.joblib \
    --risk --risk-profile caps --no-journal

# read-only dashboard
uv sync --extra dashboard
uv run tradersjoy dashboard          # http://localhost:8501
```

`--no-journal` on throwaway runs keeps the local heartbeat clean, so
`status --local` stays meaningful.

---

## Disaster recovery

If the laptop dies, **the bot keeps running.** It lives on GitHub Actions and
depends on nothing local. To rebuild a working checkout:

```bash
git clone https://github.com/arindamlabs/tradersjoy.git
cd tradersjoy
uv sync
cp .env.example .env && nano .env    # Alpaca paper keys
uv run tradersjoy ingest             # rebuild bars, ~20 years, a few minutes
```

What is and is not in the repo:

| Item | In git? | If lost |
|---|---|---|
| Code, workflows, deploy units | yes | - |
| Trained model (`data/models/ml.joblib`) | yes | - |
| Run journal (`state/journal.sqlite`) | yes | - |
| Bars (`data/tradersjoy.sqlite`) | no | rebuild with `ingest`, it is derived data |
| Alpaca keys (`.env`) | no | re-issue from the Alpaca dashboard |
| Logs (`logs/`) | no | local-only; the Actions log is the real record |

The only genuinely irreplaceable thing is the Alpaca account itself, and it is a
resettable paper account.
