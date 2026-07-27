# Railway service configuration

The app services carry their own configuration in this repo: `railway.toml`,
`railway.worker.toml` and `railway.beat.toml` hold the build and start commands,
with the reasoning behind each non-obvious flag in a comment next to it.

Redis and Postgres cannot work that way. They are image services with no repo
source, so there is no file for Railway to read and their entire runtime
configuration lives in the dashboard. This file is the reviewable record for
those two: what is set, and why it must stay that way.

Read it before changing a datastore service, and update it in the same change.

Project `e2426fe6-a773-43e3-8bb4-ef96921e8546`, environment `production`
(`460d0b79-8991-4cad-a4e2-39c0b0532fca`).

## Redis (service `8e5c610a-a078-4be5-a7ce-211c203e5d04`)

Celery broker and result backend. Also backs the Django caches.

- Image: `redis:8.2.1`
- Volume `/data`, exposed to the start command as `$RAILWAY_VOLUME_MOUNT_PATH`
- Private endpoint `redis`, TCP proxy on 6379

Start command:

```
/bin/sh -c "rm -rf $RAILWAY_VOLUME_MOUNT_PATH/lost+found/ && exec docker-entrypoint.sh redis-server --requirepass $REDIS_PASSWORD --save 60 1 --dir $RAILWAY_VOLUME_MOUNT_PATH --maxmemory 256mb --maxmemory-policy noeviction"
```

Every flag past `redis-server`, and why:

| Flag | Why |
| --- | --- |
| `--requirepass $REDIS_PASSWORD` | The TCP proxy makes the port publicly reachable. |
| `--save 60 1` | RDB snapshot after 60s if at least one key changed. Queued tasks survive a restart. |
| `--dir $RAILWAY_VOLUME_MOUNT_PATH` | Without it the snapshot lands on the ephemeral container filesystem and the volume is pointless. |
| `--maxmemory 256mb` | Ceiling. See below. |
| `--maxmemory-policy noeviction` | **Do not change this.** See below. |

The `rm -rf .../lost+found/` prefix comes from the Railway Redis template. The
volume is ext4, so it has a `lost+found` directory at its root, and Redis
refuses to treat a non-empty unexpected directory as its data dir.

### The two flags whose failure modes are silent

**`--maxmemory 256mb`** (issue #129). Redis has no memory ceiling by default. It
grows until the container is OOM-killed, which on Railway is billed the whole
way up. 256mb is roughly 40x the working set: the broker holds a handful of
queued tasks plus cache entries, and measured usage sits in the 5-7 MB range.
The cap is there to bound a runaway, not to be approached in normal operation.

**`--maxmemory-policy noeviction`** is stated explicitly rather than left to the
default *because* it is the default. An `--maxmemory` value with no policy
alongside it reads like an oversight and invites someone to "fix" it by adding
`allkeys-lru`, which is the correct choice for a pure cache and the wrong one
here. This Redis is a Celery broker. Under an eviction policy, hitting the
ceiling makes Redis delete queued task payloads to make room: jobs disappear
with no error at either end, and nothing surfaces the loss. Under `noeviction`
the producer gets a loud `OOM command not allowed` write error instead, which
is recoverable and shows up in logs.

The tradeoff is deliberate. Cache writes fail at the ceiling too, since the
caches share this instance. A failed cache write is a slow request; a silently
dropped Celery task is missing work nobody knows about.

### Verify

```
railway ssh -s Redis 'redis-cli -a "$REDIS_PASSWORD" --no-auth-warning config get maxmemory maxmemory-policy'
```

Expect `maxmemory 268435456` and `maxmemory-policy noeviction`.

Two traps when checking this after a change:

- A runtime `CONFIG SET` and a start-command flag read back identically. If you
  have been testing with `CONFIG SET`, revert it first, and confirm
  `redis-cli info server | grep uptime_in_seconds` is low enough to prove the
  process actually restarted. Otherwise a passing read proves nothing.
- Railway's `redeploy` restarts the container but does not pick up a changed
  start command, and reports success either way. Use a fresh deploy
  (`serviceInstanceDeployV2`, or the dashboard's Deploy) after editing it.

`scripts/check-railway-datastore-config.sh` runs the check and compares against
the values documented here.

## Postgres (service `ce0d937f-1a9b-4a84-86bf-767c4e866eba`)

Primary application database.

- Image: `ghcr.io/railwayapp-templates/postgres-ssl:18`
- Volume `/var/lib/postgresql/data`
- Private endpoint `postgres`, TCP proxy on 5432
- **No start command override.** The template image's entrypoint is used as-is.

Recorded here so that "nothing is set" is a documented state rather than an
unknown one. The service runs on template defaults, including
`shared_buffers` and `max_connections`. If a tuning flag is ever added, it
belongs in this file with its reasoning before it is applied.

`SSL_CERT_DAYS` and `RAILWAY_DEPLOYMENT_DRAINING_SECONDS` are template variables,
not application configuration.

## Why this is a document and not a config file

The alternative is a checked-in `redis.conf` mounted into the service, which
would make these settings diffable in the normal way. It does not fit: the
service's source is a public image, so it has no access to this repo's files.
Getting a config file in would mean building Redis from a Dockerfile here,
which adds a build to every Redis change and takes on maintenance of an image
that currently updates itself.

The honest limitation is that this file cannot enforce anything. Someone can
still change the start command in the dashboard and never open this file. What
it does buy is that the setting and its reasoning are reviewable, survive in
git history, and are recoverable if a service is recreated from the template.
`scripts/check-railway-datastore-config.sh` closes part of the gap by checking
the live values on demand.
