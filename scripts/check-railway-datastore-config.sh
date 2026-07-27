#!/usr/bin/env bash
#
# Compare the live Redis broker settings against the values recorded in
# infra/railway-services.md.
#
# Not part of the local test gate: it needs the Railway CLI, an authenticated
# session, and a running service. Run it after changing a datastore service, or
# when a service has been recreated, to confirm the deployed configuration still
# matches the documented one.
#
# Usage:  scripts/check-railway-datastore-config.sh
# Exit:   0 all settings match, 1 a mismatch, 2 could not check.

set -uo pipefail

REDIS_SERVICE="${REDIS_SERVICE:-Redis}"

# Keep in sync with infra/railway-services.md. 268435456 = 256mb.
EXPECT_MAXMEMORY="268435456"
EXPECT_POLICY="noeviction"

if ! command -v railway >/dev/null 2>&1; then
  echo "railway CLI not found. Install it or run the redis-cli check by hand;" >&2
  echo "infra/railway-services.md has the command." >&2
  exit 2
fi

raw=$(railway ssh -s "$REDIS_SERVICE" \
  'redis-cli -a "$REDIS_PASSWORD" --no-auth-warning config get maxmemory maxmemory-policy' 2>&1)
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "railway ssh failed (exit $rc):" >&2
  echo "$raw" >&2
  exit 2
fi

# CONFIG GET returns alternating key/value lines, in unspecified order, after
# whatever the CLI prints about the SSH key. Pick the value out by its key.
value_of() {
  echo "$raw" | awk -v want="$1" '
    $0 == want { getline v; print v; found = 1; exit }
    END { if (!found) exit 1 }
  '
}

fail=0
check() {
  local key="$1" expected="$2" actual
  if ! actual=$(value_of "$key"); then
    echo "FAIL  $key: not present in CONFIG GET output" >&2
    fail=1
    return
  fi
  if [ "$actual" != "$expected" ]; then
    echo "FAIL  $key: expected '$expected', got '$actual'" >&2
    fail=1
    return
  fi
  echo "ok    $key = $actual"
}

check maxmemory "$EXPECT_MAXMEMORY"
check maxmemory-policy "$EXPECT_POLICY"

if [ "$fail" -ne 0 ]; then
  cat >&2 <<'MSG'

The deployed Redis configuration does not match infra/railway-services.md.
Either the service drifted (start command edited, or the service was recreated
from the Railway template, which restores the default command and drops both
flags), or the documented values are stale.

maxmemory-policy is the one to look at first: an eviction policy on a Celery
broker silently deletes queued tasks under memory pressure. See the Redis
section of infra/railway-services.md.
MSG
  exit 1
fi

echo
echo "Redis matches infra/railway-services.md."
