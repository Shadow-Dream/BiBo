#!/bin/bash

CONTAINERS=$(docker ps -aq)

if [ -z "$CONTAINERS" ]; then
    echo "⚠️ No container to remove"
else
    echo "🗑️ Removing..."
    docker rm -f $CONTAINERS
    echo "✅ Done"
fi
