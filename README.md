# Birdsong stuff for Alberto

`aup2wav` converts an aup file into 7 wave files using the audacity python
code by davidavdav. If you put a folder contining `.aup` files in e.g.
2019-02-21, then

    aup2wav 2019-02-21/e.aup

will create a directory

    2019-02-21/wav/e/

and put files `channel-1.wav` through to `channel-7.wav` in it
