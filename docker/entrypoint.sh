#!/bin/sh
set -eu

alembic upgrade head

if [ "${1:-serve}" = "serve" ]; then
  exec python -m deuce.api.server
fi

exec "$@"
