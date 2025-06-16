#!/usr/bin/env python3

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sys
import csv
import os

def main():
    codecs = ['h264_nvenc', 'hevc_nvenc', 'av1_nvenc', 'pyrowave']
    cols = ['xpsnr', 'ssim', 'ssimulacra2', 'vmaf', 'vmafneg']

    codec_name = {
            'h264_nvenc' : 'H.264 NVENC',
            'hevc_nvenc' : 'H.265 NVENC',
            'av1_nvenc' : 'AV1 NVENC',
            'pyrowave' : 'PyroWave' }

    metric_name = {
            'xpsnr' : 'W-XPSNR',
            'ssim' : 'SSIM',
            'ssimulacra2' : 'SSIMULACRA2',
            'vmaf' : 'VMAF',
            'vmafneg' : 'VMAF NEG' }

    data_sets = {}

    for codec in codecs:
        with open(os.path.join(sys.argv[1], codec + '.csv')) as f:
            reader = csv.DictReader(f)
            bpps = []
            results = dict()
            for col in cols:
                results[col] = []
            for row in reader:
                bpp = row['bpp']
                bpps.append(bpp)
                for col in cols:
                    results[col].append(float(row[col]))
            data_sets[codec] = ([float(x) for x in bpps], results)

    print(data_sets)

    for col in cols:
        fig = go.Figure()
        for codec in codecs:
            codec_results = data_sets[codec]
            fig.add_trace(go.Scatter(x = codec_results[0], y = codec_results[1][col], name = codec_name[codec], showlegend = True))
        fig.update_traces(mode = 'markers+lines')
        fig.update_xaxes(title_text = 'bits per pixel')
        fig.update_yaxes(title_text = metric_name[col])
        fig.update_layout(legend = dict(title = dict(text = 'codec')))
        fig.update_layout(title = os.path.basename(sys.argv[1]))
        fig.write_image(os.path.join(sys.argv[1], col + '.png'))

if __name__ == '__main__':
    main()
