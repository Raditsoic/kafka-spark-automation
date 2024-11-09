#!/bin/bash
inotifywait -m ./dataset/batch -e create | while read path action file; do
    echo "The file '$file' appeared in directory '$path' via '$action'"
    python3 ./src/trainer.py -f "$path$file"
done