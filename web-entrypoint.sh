#!/bin/sh
set -e

echo "🔐 Generating Basic Auth credentials..."

# Require BASIC_AUTH_USERNAME and BASIC_AUTH_PASSWORD to be set
if [ -z "$BASIC_AUTH_USERNAME" ] || [ -z "$BASIC_AUTH_PASSWORD" ]; then
  echo "ERROR: BASIC_AUTH_USERNAME and BASIC_AUTH_PASSWORD must be set in .env"
  echo "See .env.example for reference."
  exit 1
fi

# Use htpasswd (much simpler than Python + bcrypt)
apk add --no-cache apache2-utils > /dev/null 2>&1

# Generate htpasswd file from environment variables
htpasswd -nbB "$BASIC_AUTH_USERNAME" "$BASIC_AUTH_PASSWORD" > /shared/.htpasswd

echo "✓ Basic Auth configured"
echo "  User: $BASIC_AUTH_USERNAME"
echo "🚀 Starting web service..."

# Start the original command
exec "$@"
