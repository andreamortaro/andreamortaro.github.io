#!/usr/bin/env bash

# first install nb2hugo
# build converts .ipynb into .md
FILES="$(find notebooks/todo -type f -name '*.ipynb')"
for f in $FILES
do
    nb2hugo $f --site-dir . --section posts
    mv $f notebooks/done
done
