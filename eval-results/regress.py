#!/usr/bin/env python3

import numpy as np
import csv
import argparse
import math

def estimate_psnr(kb, mpixels, chroma444, h, coefficients):
    estimate = coefficients[0]

    estimate += kb * coefficients[1]
    estimate += kb ** 0.5 * coefficients[2]
    estimate += kb ** 2.0 * coefficients[3]
    estimate += kb ** 1.5 * coefficients[4]

    estimate += mpixels * coefficients[5]
    estimate += mpixels**0.5 * coefficients[6]
    estimate += mpixels**0.75 * coefficients[7]
    estimate += mpixels**0.25 * coefficients[8]

    estimate += h * coefficients[9]
    estimate += h**0.5 * coefficients[10]
    estimate += h**2 * coefficients[11]
    estimate += h**1.5 * coefficients[12]

    estimate += kb * mpixels * coefficients[13]
    estimate += kb * h * coefficients[14]
    estimate += mpixels * h * coefficients[15]

    estimate += chroma444 * coefficients[16]

    estimate += chroma444 * kb * coefficients[17]
    estimate += chroma444 * mpixels * coefficients[18]
    estimate += chroma444 * h * coefficients[19]

    estimate += chroma444 * kb ** 2 * coefficients[20]
    estimate += chroma444 * mpixels**0.5 * coefficients[21]
    estimate += chroma444 * h**2 * coefficients[22]

    estimate += chroma444 * kb * mpixels * coefficients[23]
    estimate += chroma444 * kb * h * coefficients[24]
    estimate += chroma444 * mpixels * h * coefficients[25]

    estimate += kb * mpixels * h * coefficients[26]
    estimate += chroma444 * kb * mpixels * h * coefficients[27]
    return estimate

def main():
    parser = argparse.ArgumentParser(description = 'Script for generating a PSNR estimator JSON files.')
    parser.add_argument('--csv', type = str, action = 'append', help = 'Path to result CSV')

    args = parser.parse_args()
    if not args.csv:
        raise ArgumentError('Need --csv')

    kbits = []
    mpixels = []
    chroma444 = []
    h = []
    psnr = []

    for csv_path in args.csv:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            _ = next(reader)
            for row in reader:
                for i in range(16):
                    # Ignore values that are way outside the usable range
                    if float(row[4 + i]) < 30 or float(row[4 + i]) > 50:
                        continue

                    kbits.append(float(row[0]))
                    mpixels.append(float(row[1]) * float(row[2]) / 1000000)
                    chroma444.append(float(row[3]))
                    h.append(1 + i / 8)
                    psnr.append(float(row[4 + i]))

    kbits = np.array(kbits)
    mpixels = np.array(mpixels)
    chroma444 = np.array(chroma444)
    h = np.array(h)
    psnr = np.array(psnr)

    # Estimate PSNR as a linear set of equations.
    # More parameters seem to yield better results somehow.
    # Constant bias, various moments of the inputs, and cross-moments to extract correlation.
    A = np.column_stack([
        np.ones_like(psnr),
        kbits,
        kbits**0.5,
        kbits**2,
        kbits**1.5,
        mpixels,
        mpixels**0.5,
        mpixels**0.75,
        mpixels**0.25,
        h,
        h**0.5,
        h**2,
        h**1.5,
        kbits * mpixels,
        kbits * h,
        mpixels * h,
        chroma444,
        chroma444 * kbits,
        chroma444 * mpixels,
        chroma444 * h,
        chroma444 * kbits * kbits,
        chroma444 * mpixels**0.5,
        chroma444 * h * h,
        chroma444 * kbits * mpixels,
        chroma444 * kbits * h,
        chroma444 * mpixels * h,
        kbits * mpixels * h,
        chroma444 * kbits * mpixels * h])

    coefficients, _, _, _ = np.linalg.lstsq(A, psnr, rcond = None)
    print(coefficients)

    print(f'PSNR = {coefficients[0]} +')
    print(f'\t{coefficients[1]} * kB + {coefficients[2]} * sqrt(kB) + {coefficients[3]} * kB^2 + {coefficients[4]} * kB^1.5 +')
    print(f'\t{coefficients[5]} * MPixels + {coefficients[6]} * sqrt(MPixels) + {coefficients[7]} * MPixels^2 + {coefficients[8]} * MPixels^0.25 +')
    print(f'\t{coefficients[9]} * H + {coefficients[10]} * sqrt(H) + {coefficients[11]} * H^2 + {coefficients[12]} * H^1.5')
    print(f'\t{coefficients[13]} * kB * MPixels + {coefficients[14]} * kB * H + {coefficients[15]} * MPixels * H')

    total_sqr_errors = 0.0
    for i in range(len(psnr)):
        estimate = estimate_psnr(kbits[i], mpixels[i], chroma444[i], h[i], coefficients)
        total_sqr_errors += (estimate - psnr[i]) * (estimate - psnr[i])

    print('StdDev:', math.sqrt(total_sqr_errors / len(psnr)))

    b = 260.4
    print(estimate_psnr(b, 1280 * 720 * 1e-6, 0, 1.5, coefficients))
    print(estimate_psnr(b * math.pow((1920 * 1080) / (1280 * 720), 0.5), 1920 * 1080 * 1e-6, 0, 1.5, coefficients))
    print(estimate_psnr(b * math.pow((2560 * 1440) / (1280 * 720), 0.5), 2560 * 1440 * 1e-6, 0, 1.5, coefficients))
    print(estimate_psnr(b * math.pow((3840 * 2160) / (1280 * 720), 0.5), 3840 * 2160 * 1e-6, 0, 1.5, coefficients))

    for i in range(50):
        mbits = 100 + i * 10
        kb = 1000 * mbits / (8 * 60)
        #print(f'4:2:0 1080p mbits {mbits}, psnr = {estimate_psnr(kb, 1920 * 1080 * 1e-6, 0, 2, coefficients):.4}')
        #print(f' 4:4:4 1080p mbits {mbits}, psnr = {estimate_psnr(kb, 1920 * 1080 * 1e-6, 1, 2, coefficients):.4}')
        print(f'  4:2:0 1440p mbits {mbits}, psnr = {estimate_psnr(kb, 2560 * 1440 * 1e-6, 0, 2, coefficients):.4}')
        print(f'   4:4:4 1440p mbits {mbits}, psnr = {estimate_psnr(kb, 2560 * 1440 * 1e-6, 1, 2, coefficients):.4}')

if __name__ == '__main__':
    main()