# Subjective bitrate evaluation

Over a few days, I did some evaluation. Doing white-box evaluation like this is **not** the correct thing
but I did not have access to test subjects, so this should be seen as a crude draft, but based on my own experience
over the last year using this, the numbers make sense to me.

# Test methodology

The methodology is vaguely based on BT.500 Annex 1 to Part 2, Double Stimulus Impairment Scale, Variant II.
Mostly because I'm familiar with it from my master thesis.
The basic idea is a double-blind test where the subject is shown a sequence of clips:

- 10 seconds of a lossless reference
- 3 seconds of mid-gray
- 10 seconds of a test clip
  - Subject does not know the bitrate of this clip, it's random
  - The lossless reference may be shown.
    This is used to account for that the test subject may be overly critical to the clips shown and allows us to verify that there is meaningful statistical difference
    between the lossy and lossless clips.
- 3 seconds of mid-gray
- 10 seconds of a lossless reference
- 3 seconds of mid-gray
- 10 seconds of the same test clip
- 8 second mid-gray voting period

During the final period, the subject can vote:

- 5: Imperceptible
- 4: Perceptible, but not annoying
- 3: Slightly annoying
- 2: Annoying
- 1: Very annoying

I think the right target for streaming is somewhere between 4 and 5.

My testing is not compliant with BT.500 since even "expert test subjects" as defined by the spec should not know the
intricate details of the system under test (e.g. I know exactly what kind of artifacts to look for),
but it's the best I can do on my own. A proper test also requires way more test subjects and data.
The ideal test would involve a large number of non-expert subjects.

There is a stronger (i.e. difficult) test, which is the flicker test, but I don't think this is meaningful
for game streaming. The flicker test is mostly relevant for professional production use cases,
where visually lossless is critical since the content in question will be re-encoded later.
That's not the case for game streaming. When playing, you cannot see the original reference,
so as long as it's not obvious the stream is compressed, I think that's a good target to aim for.

It's also possible the better solution would be to test subjects response while actually playing
rather than passively watching gameplay footage, but I'm not aware how such a test should be constructed.
If actually playing a game, it's likely that subjects would focus their attention to the important parts of a screen
rather than looking at "irrelevant" background detail which noone would focus on during gameplay unless
they were trying hard to spot compression artefacts.

## Viewing distance

I chose 2H, meaning two times the height of the screen.
I believe this is a compromise between sitting in front of a PC monitor,
and playing on a TV.
Sitting closer to the screen, it's expected that more bitrate is needed since
fine details is easier to detect.
Sitting further away, human eyes are quite terrible at seeing detail,
and PyroWave being a wavelet codec is very good at exploiting that fact.

A better test would account for different viewing sizes too,
but I don't have that much time to burn on this problem.

# Test clips

I captured some difficult clips from various games. The raw footage is over 70 GB in total,
so not something that can be easily shared. This is not authoritative at all as to what good footage is,
it was just something that was easily available to capture.

- HZD remastered
  - Dense and sharp foliage.
  - Used the intro sequence where child Aloy is going through the tutorial section.
- Expedition 33
  - Inside Flying Waters
  - Running around on the world map outside Flying Waters
  - Very noisy presentation
- Fate: Samurai Remnant
  - Lots of chroma detail while spelunking in laylines
- Ghost of Tsushima
  - During prologue where you invade the mongol stronghold and get the first standoff 
  - Lots of fine speckled details
  - Dark scene
  - High contrast with fire
- Street Fighter 6
  - The HUD during fights has a lot of really difficult diagonal chroma detail that is very hard for wavelet codecs
- Witcher 3
  - In the intro, go outside and look over the forest. A ton of fine detail that easily compresses into mush
- Cyberpunk 2077
    - Some clips from the benchmark 
  
Clips were captured in 4K 4:2:0 and 1440p 4:4:4 with pyrofling.
My PC could not keep up with 4K 4:4:4 capture without tons of dropped frames.

```
pyrofling --encoder rawvideo \
    --width 3840 --height 2160 --fps 60 --no-audio --offline /tmp/clip.y4m
# Can add --444
```

From these clips, I used `pyrowave-eval-gen.py` to generate tests for:

- 720p 4:2:0, downsampled from 1440p 4:4:4
- 720p 4:4:4, downsampled from 1440p 4:4:4
- 1080p 4:2:0, downsampled from 4K 4:2:0
- 1080p 4:4:4, downsampled from 4K 4:4:4
- 1440p 4:2:0, chroma downsampled from 1440p 4:4:4
- 4K 4:2:0, using 4K 4:2:0 directly

The 4:4:4 tests were done to get a feeling for the bandwidth requirements for 4:4:4 vs 4:2:0.
4:4:4 needs extra bits for chroma to be meaningful.
I skipped 1440p 4:4:4 and 4K 4:4:4 testing, since 4:4:4 is more meaningful to test for
lower resolutions. Especially on Deck, 4:2:0 chroma is noticeable to me.
Full chroma at these resolutions is quite niche in my experience.

My expectation while game streaming is that the streaming host is powerful and renders at least 1440p.
Targeting smaller 720p and 1080p would be done through good quality downsampling (at least that's what I do ...).
It's possible this affects the numbers since the 720p and 1080p test clips are very high quality.
Game rendering at 1440p and especially 4K in modern games usually involves some kind of upscale anyway to reach
stable frame rates.

4 different screens were used to evaluate. To avoid scaling issues, I needed 4 different
screens that were all native resolution to the test content:

- 720p: Steam Deck OLED at max brightness
- 1080p: My vivobook S laptop with OLED screen
- 1440p: My gaming monitor. Not a very high-end monitor by any means.
- 4K: LG G5 2025 model.

Then I ran `pyrowave-evaluator` for an hour each setup.
The program implements the BT.500 scheme above and during voting
the subject (me) can vote 1 to 5 during the 8 second mid-gray period.
The suggested length is about 30 minutes for a session, but I needed all the data I could get.
I don't think I got quite enough data to reach high confidence.

The main goal was to get an idea how resolution would scale the bitrate requirements to get an equivalent subjective score curve.
After collecting the results, I think I've generated a (very) rough estimate:

```python
#!/usr/bin/env python3

import math

"""
Based on empirical results
W = width in pixels
H = height in pixels
fps = frames per second
chroma444 = if true, 4:4:4, else 4:2:0
hdr10 = if true, 10-bit PQ, else SDR (8-bit vs 10-bit doesn't really matter since pyrowave is a floating point codec).
"""
def rate_mbits(W, H, fps, chroma444, hdr10):
    # 0.38 matched my test data well enough.
    rate = 125 * math.pow(((W * H) / (1280 * 720)), 0.38)
    rate *= 1.15 if chroma444 else 1.0
    rate *= 1.2 if hdr10 else 1.0
    # Intra-codec so bitrate requirements scale linearly with frame rate.
    rate *= fps / 60
    return rate

def main():
    for i in range(2):
        for c in range(2):
            HDR = i == 1
            chroma444 = c == 1
            print(f'{'SDR' if i == 0 else 'HDR10'} {'4:4:4' if chroma444 else '4:2:0'}:')
            print(f' 720p -> {rate_mbits(1280, 720, 60, chroma444, HDR):.6} mbit/s')
            print(f' 1080p -> {rate_mbits(1920, 1080, 60, chroma444, HDR):.6} mbit/s')
            print(f' 1440p -> {rate_mbits(2560, 1440, 60, chroma444, HDR):.6} mbit/s')
            print(f' 4K -> {rate_mbits(3840, 2160, 60, chroma444, HDR):.6} mbit/s')

    # Upper end of viability for gigabit ethernet:
    print(f'4K 4:4:4 HDR10 120 FPS -> {rate_mbits(3860, 2560, 120, True, True):.6} mbit/s')

if __name__ == '__main__':
    main()
```

```
SDR 4:2:0:
 720p -> 125.0 mbit/s
 1080p -> 170.114 mbit/s
 1440p -> 211.686 mbit/s
 4K -> 288.086 mbit/s
SDR 4:4:4:
 720p -> 143.75 mbit/s
 1080p -> 195.631 mbit/s
 1440p -> 243.439 mbit/s
 4K -> 331.299 mbit/s
HDR10 4:2:0:
 720p -> 150.0 mbit/s
 1080p -> 204.136 mbit/s
 1440p -> 254.024 mbit/s
 4K -> 345.703 mbit/s
HDR10 4:4:4:
 720p -> 172.5 mbit/s
 1080p -> 234.757 mbit/s
 1440p -> 292.127 mbit/s
 4K -> 397.559 mbit/s
4K 4:4:4 HDR10 120 FPS -> 849.821 mbit/s (uncompressed is ~29860 mbit/s).
```

The scaling factor for pixel counts is very weak.
At 720p, the detail remaining in the image is quite critical and is quite hard to compress for PyroWave.

NOTE: The testing I did was for SDR. The HDR scaling factor is a vague factor I came up with while
testing some HDR10 earlier.

These bitrates serve as a baseline for bitrates for good quality in my experience.
The bitrates can be scaled with a quality factor as desired.