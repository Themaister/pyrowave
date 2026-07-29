#!/usr/bin/env python3

import sys
import argparse
import os
import json
import subprocess

def input_to_basename(inp):
    basename = os.path.basename(inp)
    return os.path.splitext(basename)[0]

def main():
    parser = argparse.ArgumentParser(description = 'Script for generating evaluator JSON files.')
    parser.add_argument('--input', type = str, action = 'append', help = 'Path to reference (lossless) input file.')
    parser.add_argument('--output', type = str, help = 'Path to output .json file.')
    parser.add_argument('--width', type = int, default = 1920, help = 'Width of outputs.')
    parser.add_argument('--height', type = int, default = 1080, help = 'Height of outputs.')
    parser.add_argument('--chroma444', action = 'store_true', help = 'Use 4:4:4.')
    parser.add_argument('--pyrowave', action = 'store_true', help = 'Add pyrowave tests.')
    parser.add_argument('--h264', action = 'store_true', help = 'Use H.264 tests.')
    parser.add_argument('--h265', action = 'store_true', help = 'Use H.265 tests.')
    parser.add_argument('--fps', type = int, default = 60, help = 'Use specific FPS.')
    parser.add_argument('--pyroenc-offline-path', type = str, help = 'Path to pyroenc offline tool.')

    args = parser.parse_args()
    if not args.input:
        raise ArgumentError('Need --input')

    # TODO: HDR and 10-bit
    pix_fmt = 'yuv444p' if args.chroma444 else 'yuv420p'
    out_dir = os.path.dirname(args.output)

    tests = {}
    num_pixels = args.width * args.height
    pyrowave_bitrate_candidates_baseline = [ 10, 20, 35, 50, 75, 100, 125, 150, 175, 200, 250 ]
    pyroenc_bitrate_candidates_baseline = [ 1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 80, 100 ]

    if (args.h264 or args.h265) and args.chroma444:
        raise ArgumentError('h264/h265 testing is currently not compatible with 4:4:4')

    for inp in args.input:
        test_name = input_to_basename(inp)
        scaled_name = f'{test_name}.{args.width}x{args.height}x{444 if args.chroma444 else 420}.y4m'
        out_path = os.path.join(out_dir, scaled_name)
        subprocess.run(['ffmpeg', '-y', '-i', inp, '-vf', f'scale={args.width}:{args.height}', '-pix_fmt', pix_fmt, '-t', '10', out_path], check = True)
        clips = []
        clips.append({ 'path' : scaled_name, 'name' : 'reference', 'desc' : 'reference' })

        if args.pyrowave:
            for rate in pyrowave_bitrate_candidates_baseline:
                if num_pixels >= 3840 * 2160:
                    rate = rate * 3
                elif num_pixels >= 2560 * 1440:
                    rate = rate * 2
                elif num_pixels >= 1920 * 1080:
                    rate = rate * 3 // 2
                rate = int(rate * args.fps / 60)
                clips.append({ 'codec' : 'pyrowave', 'mbits' : rate, 'name' : f'pyrowave_{rate}', 'desc' : f'PyroWave ({rate} mbits)' })

        active_pyroenc_types = []
        if args.h264:
            active_pyroenc_types.append('h264')
        if args.h265:
            active_pyroenc_types.append('h265')

        for pyro in active_pyroenc_types:
            for rate in pyroenc_bitrate_candidates_baseline:
                encoded_name = f'{test_name}.{rate}.{pyro}'
                nut_name = f'{test_name}.{rate}.{pyro}.nut'
                encoded_out_path = os.path.join(out_dir, encoded_name)
                encoded_nut_path = os.path.join(out_dir, nut_name)
                rate = int(rate * args.fps / 60)
                subprocess.run([args.pyroenc_offline_path, '--output', encoded_out_path, '--intra-refresh', '32', '--bitrate-kbits', str(rate * 1000), out_path], check = True)
                subprocess.run(['ffmpeg', '-fflags', '+genpts', '-y', '-r', str(args.fps), '-i', encoded_out_path, '-c:v', 'copy', encoded_nut_path], check = True)
                clips.append({ 'codec' : 'ffmpeg', 'path' : nut_name, 'name' : f'h264_{rate}', 'desc' : f'{pyro} ({rate} mbits)' })
                os.remove(encoded_out_path)

        tests[test_name] = clips

    doc = { 'tests' : tests }
    with open(args.output, 'w') as f:
        f.write(json.dumps(doc, indent = 4))

if __name__ == '__main__':
    main()
