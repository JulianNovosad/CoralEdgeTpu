#!/bin/bash
HOST="pi@192.168.178.48"
DEST="/home/pi/CoralEdgeTpu"

echo "Syncing files to $HOST:$DEST..."
rsync -avz --exclude 'build' --exclude '.git' --exclude '.gemini' --exclude '.agent' --progress ./ $HOST:$DEST/

echo "Executing remote build and test..."
ssh -t $HOST "$DEST/remote_test.sh"
