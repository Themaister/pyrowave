#!/bin/bash

python pyrowave-eval-gen.py \
	--input ~/pyrowave-reference/witcher-3-4k-420.nut \
	--input ~/pyrowave-reference/witcher-3-4k-420-2.nut \
	--input ~/pyrowave-reference/sf6-4k-1.nut \
	--input ~/pyrowave-reference/sf6-4k-2.nut \
	--input ~/pyrowave-reference/cp77-4k-1.nut \
	--input ~/pyrowave-reference/cp77-4k-2.nut \
	--input ~/pyrowave-reference/e33-4k-1.nut \
	--input ~/pyrowave-reference/e33-4k-2.nut \
	--input ~/pyrowave-reference/fate-4k-1.nut \
	--input ~/pyrowave-reference/got-4k-1.nut \
	--input ~/pyrowave-reference/got-4k-2.nut \
	--input ~/pyrowave-reference/hzd-4k-1.nut \
	--input ~/pyrowave-reference/hzd-4k-2.nut \
	--output test.json \
	--width 1920 --height 1080 \
	--pyrowave
