
This is an AI assisted port of Pyrowave to Metal on Apple platforms.

This has no Granite code, using native Metal and converting the shaders to MSL.

This is covered by the standard MIT license available in ../LICENSE.

Port notes:
* The API has been simplified, removing Vulkanisms and using Metal objects for GPU decode.
* You can set the environment variable PYROWAVE_PRECISION to 0, 1, or 2, to make the same precision/speed tradeoffs as the main library.
* Using FP32 math and FP16 storage in the shaders ended up being the fastest and most accurate combination on Apple hardware.
