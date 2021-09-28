# Birdsong stuff for Alberto

`aup2wav` converts an aup file into 7 wave files using the audacity python
code by davidavdav. If you put a folder contining `.aup` files in e.g.
2019-02-21, then

    aup2wav 2019-02-21/e.aup

will create a directory

    2019-02-21/wav/e/

and put files `channel-1.wav` through to `channel-7.wav` in it

## GCC-PHAT code
The GCC-PHAT code is in `gcc-phat.py`. It takes as a compulsory argument
a directory containing the 7 wav files entitled `channel-1.wav` through to
`channel-7.wav`. You can also pass in as options:

`--start` : a start time (in seconds) (default 0.0)
`--length` : the length of the window you want to perform GCC-PHAT over
(default 2.0)
`--speedofsound : the speed of sound (default 343)

It will produce two plots. One of the 7 waveforms, and one of the estimated
strength in each direction. By default the resolution is $1%\circ$.
