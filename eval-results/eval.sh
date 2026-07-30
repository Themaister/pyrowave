#!/bin/bash

python pyrowave-eval-gen.py \
	--input /storage/pyrowave-reference/sf6-1440-444-1.nut \
	--input /storage/pyrowave-reference/sf6-1440-444-2.nut \
	--input /storage/pyrowave-reference/cp77-1440-444-1.nut \
	--input /storage/pyrowave-reference/cp77-1440-444-2.nut \
	--input /storage/pyrowave-reference/e33-1440-444-1.nut \
	--input /storage/pyrowave-reference/e33-1440-444-2.nut \
	--input /storage/pyrowave-reference/fate-1440-444-1.nut \
	--input /storage/pyrowave-reference/got-1440-444-1.nut \
	--input /storage/pyrowave-reference/got-1440-444-2.nut \
	--input /storage/pyrowave-reference/hzd-1440-1.nut \
	--input /storage/pyrowave-reference/hzd-1440-2.nut \
	--input /storage/pyrowave-reference/witcher-3-1440-444-1.nut \
	--input /storage/pyrowave-reference/witcher-3-1440-444-2.nut \
	--output /tmp/test.json \
	--width 2560 --height 1440 \
	--pyrowave \
	--pyroenc-offline-path /home/maister/git/pyroenc/cmake-build-release/offline/pyroenc-offline
