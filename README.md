# PyroWave

PyroWave is an intra-only video codec (practially speaking a still-image codec)
that is optimized for extremely fast GPU encode (< ~0.1 ms encode and decode at 1080p, < ~0.2 ms at 4K).
It is fully implemented in Vulkan compute shaders.

The targeted bit-rates are quite high (~200+ mbit/s) and the intended use case is
local network game streaming over ethernet with absolute minimum latency where bandwidth is less of a concern.

It is integrated as part of my [pyrofling](https://github.com/Themaister/pyrofling) project.

It is similar in scope to my [master thesis from 2014](https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2400689),
except the use case now is game streaming instead of adaptation of raw video to ethernet links.

I'm mostly doing this for my own amusement, and I don't expect this codec to be useful for anyone else at the moment.

## Overview

### Colorspace

Currently implemented YCbCr 4:2:0. 4:4:4 can also be supported trivially.

### Wavelets

The images are transformed with the Discrete Wavelet Transform using CDF 9/7 filter.
This is basically the exact same as JPEG2000.

### Exact rate control

The encoder can target exact maximum bitrate for an encoded image. This process takes ~20 us.

### Trivial "entropy" coding

The encoding of coefficients is trivial compared to normal image codecs,
and is responsible for a large increase in bit-rate, especially at higher compression ratios.
This simplicity massively improves encode/decode performance, since it's extremely parallelizable.
It also makes it possible to do a single-pass exact rate control for an entire image.

It should be possible in theory to add a more proper entropy coder to the encoded bit-planes to achieve somewhat
competent compression, but that has not been explored and is out of scope.
High Throughput JPEG2000 is likely a good place to look for that anyway.

### Robustness against packet loss

Being intra-only and encoding 64x64 blocks of coefficients in isolation ensures great recovery against packet loss.
It's plausible pyrowave could be useful over fiber links.