#!/bin/bash

set -e

input_file="$1"
output_dir="$2"

if [ -z $input_file ]; then
	echo "Need to specify input."
	exit 1
fi

if [ -z $output_dir ]; then
	echo "Need to specify output."
	exit 1
fi

mkdir -p "$output_dir"
rm -f "$output_dir"/*.csv

csv_header="bpp,xpsnr,ssim,ssimulacra2,vmaf,vmafneg"
echo $csv_header > "$output_dir"/h264_nvenc.csv
echo $csv_header > "$output_dir"/hevc_nvenc.csv
echo $csv_header > "$output_dir"/av1_nvenc.csv
#echo $csv_header > "$output_dir"/h264_vaapi.csv
#echo $csv_header > "$output_dir"/hevc_vaapi.csv
#echo $csv_header > "$output_dir"/av1_vaapi.csv
echo $csv_header > "$output_dir"/pyrowave.csv

append_stats_to_csv()
{
	stats=$1
	csv=$2

	bpp=$(grep "BPP =" $stats | grep -oE '[^ ]+$')
	xpsnr=$(grep "W-XPSNR:" $stats | grep -oE '[^ ]+$')
	ssim=$(grep "SSIM Score:" $stats | grep -oE '[^ ]+$')
	ssimulacra2=$(grep "Average:" $stats | grep -oE '[^ ]+$')
	vmaf=$(grep "VMAF:" $stats | grep -oE '[^ ]+$')
	vmafneg=$(grep "VMAF NEG:" $stats | grep -oE '[^ ]+$')

	echo "$bpp,$xpsnr,$ssim,$ssimulacra2,$vmaf,$vmafneg" >> $csv
}

encode_nvenc()
{
	bit_rate=$1k
	# Round up to not give codec an impossible situation
	buf_size=$(($1 / 60 + 1))k
	codec=$2
	encoded_name="$output_dir"/${codec}_nvenc_${bit_rate}.mkv
	stats=""$output_dir"/${codec}_nvenc_stats_${bit_rate}.txt"

	echo "============="
	echo "Encoding with NVENC, bitrate ${bit_rate}."
	echo ffmpeg -y -i $input_file -b:v $bit_rate -c:v ${codec}_nvenc \
		-preset p1 -tune ull -g 1 -rc cbr -bufsize $buf_size $encoded_name
	ffmpeg -y -i $input_file -b:v $bit_rate -c:v ${codec}_nvenc \
		-preset p1 -tune ull -g 1 -rc cbr -bufsize $buf_size $encoded_name >/dev/null 2>/dev/null
	ffmpeg -y -i $encoded_name $encoded_name.y4m >/dev/null 2>/dev/null
	psy-ex-scores.py $input_file $encoded_name.y4m -t 16 -e 4 | ansi2txt > $stats
	rm $encoded_name.y4m
	echo "Stats in ${stats}, reference output in ${encoded_name}."
	du -b $encoded_name
	echo "============="

	actual_size=$(stat -c %s $encoded_name)
	input_size=$(stat -c %s $input_file)
	bpp=$(echo "scale=3; 12 * $actual_size / $input_size" | bc)
	echo "BPP = $bpp" >> $stats

	append_stats_to_csv $stats "$output_dir"/${codec}_nvenc.csv

	rm $encoded_name
}

encode_vaapi()
{
	bit_rate=${1}k
	# Round up to not give codec an impossible situation
	buf_size=$(($1 / 60 + 1))k
	codec=$2
	encoded_name="$output_dir"/${codec}_vaapi_${bit_rate}.mkv
	stats=""$output_dir"/${codec}_vaapi_stats_${bit_rate}.txt"
	echo "============="
	echo "Encoding with VAAPI, bitrate ${bit_rate}."
	ffmpeg -y -vaapi_device /dev/dri/renderD128 -i $input_file -c:v ${codec}_vaapi \
		-idr_interval 1 -g 1 -rc_mode CBR -b:v $bit_rate -bufsize $buf_size \
		-vf format=nv12,hwupload $encoded_name >/dev/null 2>/dev/null
	ffmpeg -y -i $encoded_name $encoded_name.y4m >/dev/null 2>/dev/null
	psy-ex-scores.py $input_file $encoded_name.y4m -t 16 -e 4 | ansi2txt > $stats
	rm $encoded_name.y4m
	echo "Stats in ${stats}, reference output in ${encoded_name}."
	du -b $encoded_name
	echo "============="

	actual_size=$(stat -c %s $encoded_name)
	input_size=$(stat -c %s $input_file)
	bpp=$(echo "scale=3; 12 * $actual_size / $input_size" | bc)
	echo "BPP = $bpp" >> $stats

	append_stats_to_csv $stats "$output_dir"/${codec}_vaapi.csv

	rm $encoded_name
}

encode_pyrowave()
{
	bit_rate=${1}k
	bytes_per_image=$((($1 * 1000) / (60 * 8)))
	encoded_name="$output_dir"/pyrowave_${bit_rate}.y4m
	stats=""$output_dir"/pyrowave_stats_${bit_rate}.txt"
	echo "============="
	echo "Encoding with PyroWave, bitrate ${bit_rate}, $bytes_per_image bytes per image."
	pyrowave-sandbox $input_file $encoded_name $bytes_per_image 2>/dev/null
	psy-ex-scores.py $input_file $encoded_name -t 16 -e 4 | ansi2txt > $stats
	echo "Stats in ${stats}, reference output in ${encoded_name}."
	echo "============="

	bpp=$(echo "scale=3; 8 * $bytes_per_image / (1920 * 1080)" | bc)
	echo "BPP = $bpp" >> $stats
	append_stats_to_csv $stats "$output_dir"/pyrowave.csv

	rm $encoded_name
}

encode_nvenc_group()
{
	encode_nvenc $1 h264
	encode_nvenc $1 hevc
	encode_nvenc $1 av1
}

encode_vaapi_group()
{
	encode_vaapi $1 h264
	encode_vaapi $1 hevc
	encode_vaapi $1 av1
}

encode_accel_group()
{
	encode_nvenc_group $1
	# VAAPI seems to not understand what CBR rate control means :') Ignore it.
	#encode_vaapi_group $1
}

encode_pyrowave 50000
encode_pyrowave 75000
encode_pyrowave 100000
encode_pyrowave 150000
encode_pyrowave 200000
encode_pyrowave 250000
encode_pyrowave 300000
encode_pyrowave 400000
encode_pyrowave 500000
encode_accel_group 50000
encode_accel_group 75000
encode_accel_group 100000
encode_accel_group 150000
encode_accel_group 200000
encode_accel_group 250000
encode_accel_group 300000
encode_accel_group 400000
encode_accel_group 500000

python plot.py $output_dir

