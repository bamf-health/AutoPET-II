#!/usr/bin/env bash

./build.sh

docker save autopet | gzip -c > autopet.tar.gz
