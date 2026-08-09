#include "pyrowave_regression_results.h"
#include <stdio.h>
#include <stdlib.h>

int main()
{
	for (int psnr = 32; psnr <= 38; psnr++)
	{
		fprintf(stderr, "Computing targets for PSNR %d\n", psnr);
		for (int height = 720; height <= 2160; height += 9 * 4)
		{
			double mbits = pyrowave_psnr_hvs_m_h_estimate_mbits(psnr, (height / 9) * 16, height, PYROWAVE_HEIGHT_FACTOR_2_00, 0, 60.0);
			fprintf(stderr, "Height = %u -> %.1f mbit\n", height, mbits);

			double mbits_444 = pyrowave_psnr_hvs_m_h_estimate_mbits(psnr, (height / 9) * 16, height, PYROWAVE_HEIGHT_FACTOR_2_00, 1, 60.0);
			fprintf(stderr, "Height = %u -> %.1f mbit / %.1f mbit (+ %.3f %% change)\n", height, mbits, mbits_444, 100.0 * (mbits_444 / mbits - 1.0));
		}
		fprintf(stderr, "\n");
	}

	FILE *file = fopen("rates.m", "w");
	if (!file)
		return EXIT_FAILURE;

	for (int height_factor = 0; height_factor <= PYROWAVE_HEIGHT_FACTOR_2_75; height_factor += 2)
	{
		for (int psnr = 34; psnr <= 36; psnr++)
		{
			fprintf(file, "heights_%d = [", psnr);
			for (int height = 720; height <= 2160; height += 9)
				fprintf(file, "%d%s", height, height != 2160 ? ", " : "");
			fprintf(file, "];\n");

			for (int c444 = 0; c444 < 2; c444++)
			{
				fprintf(file, "rates_c%s_%d = [", (c444 ? "444" : "420"), psnr);

				for (int height = 720; height <= 2160; height += 9)
				{
					double mbits = pyrowave_psnr_hvs_m_h_estimate_mbits(psnr, (height / 9) * 16, height, height_factor,
																			c444, 60.0);
					fprintf(file, "%f%s", mbits, height != 2160 ? ", " : "");
				}
				fprintf(file, "];\n");
			}
		}

		fprintf(file, "figure;\n");
		fprintf(file, "plot(");
		for (int psnr = 34; psnr <= 36; psnr++)
		{
			fprintf(file, "heights_%d, rates_c420_%d, ", psnr, psnr);
			fprintf(file, "heights_%d, rates_c444_%d", psnr, psnr);
			if (psnr != 36)
				fprintf(file, ", ");
		}
		fprintf(file, ");\n");
		fprintf(file, "legend(");
		for (int psnr = 34; psnr <= 36; psnr++)
		{
			fprintf(file, "\"%d dB (Y) PSNR-HVS-M-H 4:2:0\", ", psnr);
			fprintf(file, "\"%d dB (Y) PSNR-HVS-M-H 4:4:4\"", psnr);
			if (psnr != 36)
				fprintf(file, ", ");
		}
		fprintf(file, ");\n");

		static const char *height_names[] = {
			"H = 1.00",
			"H = 1.25",
			"H = 1.50",
			"H = 1.75",
			"H = 2.00",
			"H = 2.25",
			"H = 2.50",
			"H = 2.75",
		};
		fprintf(file, "title(\"%s\");\n", height_names[height_factor / 2]);
		fprintf(file, "xlabel(\"Height in pixels\");\n");
		fprintf(file, "ylabel(\"Mbit/s (@ 60 fps)\");\n");
	}
}