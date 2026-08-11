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

class BucketPixelsToSize:
    def __init__(self, h, c444, psnr):
        self.h = h
        self.c444 = c444
        self.psnr = psnr
        self.results = []

    def add_pix_to_size_result(self, pix, size):
        self.results.append((pix, size))

    def finish(self):
        self.results.sort(key = lambda a: a[0])

# Maps out a curve for size -> psnr
class BucketSizeToPSNR:
    def __init__(self, h, pix, c444):
        self.h = h
        self.pix = pix
        self.c444 = c444
        self.results = []

    def add_size_to_psnr_result(self, size, psnr):
        self.results.append((size, psnr))

    def finish(self):
        self.results.sort(key = lambda a: a[1])

    def get_size_requirement_for_psnr(self, psnr):
        for i in range(len(self.results) - 1):
            first = self.results[i + 0]
            second = self.results[i + 1]
            if psnr >= first[1] and psnr <= second[1]:
                l = (psnr - first[1]) / (second[1] - first[1])
                return first[0] + l * (second[0] - first[0])
        return 0

def find_or_create_bucket_size_to_psnr(buckets, h, pix, c444):
    for bucket in buckets:
        if bucket.h == h and bucket.pix == pix and bucket.c444 == c444:
            return bucket
    bucket = BucketSizeToPSNR(h, pix, c444)
    buckets.append(bucket)
    return bucket

def find_or_create_bucket_pix_to_size(buckets, h, c444, psnr):
    for bucket in buckets:
        if bucket.h == h and bucket.c444 == c444 and bucket.psnr == psnr:
            return bucket
    bucket = BucketPixelsToSize(h, c444, psnr)
    buckets.append(bucket)
    return bucket

def regress(values):
    pix = [x[0] for x in values]
    sizes = np.array([x[1] for x in values])

    pix = np.array(pix)
    pix = np.sqrt(pix / 1e6) - 2
    pix2 = pix**2
    pix3 = pix**3
    pix4 = pix**4
    pix5 = pix**5
    pix6 = pix**6
    pix7 = pix**7

    A = np.column_stack([np.ones_like(pix), pix, pix2, pix3, pix4, pix5, pix6, pix7])
    coefficients, residuals, _, _ = np.linalg.lstsq(A, sizes, rcond = None)
    residual_manual = math.sqrt(residuals[0] / len(pix))
    return coefficients, residual_manual

def main():
    parser = argparse.ArgumentParser(description = 'Script for generating a PSNR estimator JSON files.')
    parser.add_argument('--csv', type = str, action = 'append', help = 'Path to result CSV')

    args = parser.parse_args()
    if not args.csv:
        raise ArgumentError('Need --csv')

    buckets_size_to_psnr = []
    buckets_pix_to_size = []

    for csv_path in args.csv:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            _ = next(reader)
            for row in reader:
                kb = float(row[0])
                pix = float(row[1]) * float(row[2])
                c444 = float(row[3])
                for i in range(16):
                    #h = 1 + i / 8
                    h = i
                    bucket = find_or_create_bucket_size_to_psnr(buckets_size_to_psnr, h, pix, c444)
                    psnr = float(row[4 + i])
                    bucket.add_size_to_psnr_result(kb, psnr)

    for size_to_psnr_bucket in buckets_size_to_psnr:
        size_to_psnr_bucket.finish()

        for psnr in range(30, 51):
            size_req = size_to_psnr_bucket.get_size_requirement_for_psnr(psnr)
            if size_req > 0:
                pix_to_size_bucket = find_or_create_bucket_pix_to_size(buckets_pix_to_size, size_to_psnr_bucket.h, size_to_psnr_bucket.c444, psnr)
                pix_to_size_bucket.add_pix_to_size_result(size_to_psnr_bucket.pix, size_req)

    lo_stddev = 1e30
    hi_stddev = 0

    for bucket in buckets_pix_to_size:
        bucket.finish()
        #print(f'Regressing for h = {bucket.h}, c444 = {bucket.c444}, psnr = {bucket.psnr}')
        coefficients, stddev = regress(bucket.results)
        #print(f'PSNR = {bucket.psnr} -> stddev {stddev}')

        size = bucket.results[len(bucket.results) // 2][1]
        stddev /= size

        if stddev > hi_stddev:
            print(f'New record {stddev} for PSNR {bucket.psnr}, h = {bucket.h}, size = {size}')
        hi_stddev = max(hi_stddev, stddev)
        lo_stddev = min(lo_stddev, stddev)

    print(f'LO {lo_stddev}, HI {hi_stddev}\n\n')

    print('#ifndef PYROWAVE_REGRESSION_RESULTS_H_')
    print('#define PYROWAVE_REGRESSION_RESULTS_H_')
    print('#include <math.h>')
    print('#include <assert.h>')
    print('#include <stddef.h>')
    print('/* Autogenerated by regress.py */')
    print('enum pyrowave_height_factor {')
    print('\tPYROWAVE_HEIGHT_FACTOR_1_00 = 0,')
    print('\tPYROWAVE_HEIGHT_FACTOR_1_12 = 1,')
    print('\tPYROWAVE_HEIGHT_FACTOR_1_25 = 2,')
    print('\tPYROWAVE_HEIGHT_FACTOR_1_37 = 3,')
    print('\tPYROWAVE_HEIGHT_FACTOR_1_50 = 4,')
    print('\tPYROWAVE_HEIGHT_FACTOR_1_62 = 5,')
    print('\tPYROWAVE_HEIGHT_FACTOR_1_75 = 6,')
    print('\tPYROWAVE_HEIGHT_FACTOR_1_87 = 7,')
    print('\tPYROWAVE_HEIGHT_FACTOR_2_00 = 8,')
    print('\tPYROWAVE_HEIGHT_FACTOR_2_12 = 9,')
    print('\tPYROWAVE_HEIGHT_FACTOR_2_25 = 10,')
    print('\tPYROWAVE_HEIGHT_FACTOR_2_37 = 11,')
    print('\tPYROWAVE_HEIGHT_FACTOR_2_50 = 12,')
    print('\tPYROWAVE_HEIGHT_FACTOR_2_62 = 13,')
    print('\tPYROWAVE_HEIGHT_FACTOR_2_75 = 14,')
    print('\tPYROWAVE_HEIGHT_FACTOR_2_87 = 15')
    print('};')
    print('#define PYROWAVE_REGRESSION_MIN_PSNR_HVS_M_H 30')
    print('#define PYROWAVE_REGRESSION_MAX_PSNR_HVS_M_H 50')
    print('#define PYROWAVE_REGRESSION_MIN_PIXELS (1280 * 720)')
    print('#define PYROWAVE_REGRESSION_MAX_PIXELS (3840 * 2160)')
    print('static const struct pyrowave_rate_regression_result {')
    print('\tint psnr_hvs_m_h;')
    print('\tint height_factor;')
    print('\tint chroma444;')
    print('\tdouble poly_coefficients[8];')
    print('} pyrowave_buckets[] = {')

    for bucket in buckets_pix_to_size:
        coefficients, _ = regress(bucket.results)
        coeffs_str = [str(x) for x in coefficients]
        coeffs_str = ', '.join(coeffs_str)
        print('\t{', f'{int(bucket.psnr)}, {int(bucket.h)}, {int(bucket.c444)},', '{', coeffs_str, '}', '},')
    print('};')
    print('''
static double pyrowave_psnr_hvs_m_h_estimate_mbits(
        int psnr, int width, int height,
        enum pyrowave_height_factor height_factor, int chroma444, double fps)
{
    int num_pixels = width * height;
    size_t i;
    assert(psnr >= PYROWAVE_REGRESSION_MIN_PSNR_HVS_M_H && psnr <= PYROWAVE_REGRESSION_MAX_PSNR_HVS_M_H);
    assert(num_pixels >= PYROWAVE_REGRESSION_MIN_PIXELS && num_pixels <= PYROWAVE_REGRESSION_MAX_PIXELS);
    
    for (i = 0; i < sizeof(pyrowave_buckets) / sizeof(pyrowave_buckets[0]); i++)
    {
        const struct pyrowave_rate_regression_result *result = &pyrowave_buckets[i];
        if (result->psnr_hvs_m_h == psnr && result->height_factor == height_factor && result->chroma444 == chroma444)
        {
            double pixel_non_linear = (double)num_pixels;
            double power_chain = 1.0;
            double estimate = 0.0;
            int coeff;
            
            /* Non-linear warp to make the polynomial estimate more accurate. */
            pixel_non_linear = sqrt(pixel_non_linear * 1e-6) - 2.0;
            for (coeff = 0; coeff < 8; coeff++)
            {
                estimate += power_chain * result->poly_coefficients[coeff];
                power_chain *= pixel_non_linear;
            }
            
            return estimate * 8e-3 * fps;
        }
    }
    
    return 0.0;
}''')
    print('#endif /* PYROWAVE_REGRESSION_RESULTS_H_ */')

if __name__ == '__main__':
    main()