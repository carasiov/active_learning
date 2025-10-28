## Agent Addendum: Tracking User Interactions via Logs

As an agent, you cannot drive the browser UI, but you can reliably infer user interactions from server-side logs. Use this guide to fetch and interpret those logs when investigating behavior, reproducing steps, or summarizing user activity.

### Log Destinations
- Primary structured log (DEBUG to file): `/tmp/ssvae_dashboard.log`
- Server stdout/stderr (when backgrounded): `artifacts/dashboard_server.log`

### Logger Names and Coverage
- `dashboard.app` – app lifecycle and routing (page navigation)
- `dashboard.callbacks` – callback calls, successes, prevented updates, exceptions
- `dashboard.state` – state-change summaries (when helpers are used)
- `dashboard.commands` – command dispatcher history and messages
- `dashboard.model_manager` – persistence events (metadata/history/labels)

### What You Can Observe
- Navigation events from `display_page(...)` include `pathname=/...` and the chosen route.
- UI events via `@logged_callback(...)` emit entries such as:
  - `... <name> CALLED | args=(...), kwargs={}`
  - `... <name> SUCCESS | result=(...)`
  - `... PREVENTED UPDATE` when no state change is intended
  - `... FAILED | ...` with traceback when exceptions occur
- Command executions are recorded by the dispatcher with success/failure and messages.

### Tail and Filter Examples
- All activity:
  - `tail -f /tmp/ssvae_dashboard.log`
- Callbacks only:
  - `tail -f /tmp/ssvae_dashboard.log | rg "dashboard\.callbacks|CALLED|SUCCESS|FAILED"`
- Navigation/routing:
  - `tail -f /tmp/ssvae_dashboard.log | rg "DISPLAY_PAGE|pathname="`
- Server console (HTTP access, static assets):
  - `tail -f artifacts/dashboard_server.log`

### Increase Console Verbosity (Optional)
- File logging is always at DEBUG, but console verbosity is configured at app startup.
- Edit `use_cases/dashboard/app.py` and set:
  - `DashboardLogger.setup(console_level=logging.DEBUG)`
- Restart the server for changes to take effect.

### Quick Server Operations
- Start (foreground): `poetry run python use_cases/dashboard/app.py`
- Start (background): `nohup poetry run python use_cases/dashboard/app.py > artifacts/dashboard_server.log 2>&1 & echo $!`
- Check port: `ss -ltn | rg ':8050'`
- Stop: `kill <PID>`

### Implementation References
- Logging setup: `use_cases/dashboard/core/logging_config.py`
- Callback logging decorator: `use_cases/dashboard/utils/callback_utils.py`
- Router/page logging (navigation): `use_cases/dashboard/app.py`

With the above, you can correlate user clicks to specific callbacks, follow navigation flows (home → model → pages), and inspect command effects without direct UI access.

