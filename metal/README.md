
This is an AI assisted port of Pyrowave to Metal on Apple platforms.

This has no Granite code, using native Metal and converting the shaders to MSL.

This is covered by the standard MIT license available in ../LICENSE.

Upstream note: This port may be removed entirely at some point if some effort is spend on a more direct implementation.
This code duplication and AI use is a pragmatic compromise since a quick and dirty port was needed by clients and upstream did not have a setup to do a "proper" port. Upstream is not able to maintain or support this port beyond the absolute minimum and is provided here for convenience for said clients.

Port notes:
* The API has been simplified, removing Vulkanisms and using Metal objects for GPU decode.
* You can set the environment variable PYROWAVE_PRECISION to 0, 1, or 2, to make the same precision/speed tradeoffs as the main library.
* Using FP32 math and FP16 storage in the shaders ended up being the fastest and most accurate combination on Apple hardware.
