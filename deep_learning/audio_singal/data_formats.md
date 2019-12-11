# Data Formats in Audio



### Reference

*   https://magiclen.org/acoustics/



## Quality

-   Sampling Rate : Sample the audio wave in a fixed period.
    -   Unit: `Hz`, for example 44.1KHz standing for sampling audio states for 44100 time points in a second.
    -   奈奎斯特定理 (Nyquist Theorem) : 取樣頻率兩倍大於被取樣訊號的最大頻率，即可重構出原始的被取樣訊號。
    -   Singal Distortion: 如果取樣頻率不足於被取樣訊號的最大頻率加上其低通濾波器的過渡頻寬的兩倍，最大頻率附近的聲音訊號依然會有混疊 (aliasing) 現象。
    -   Transition Band : typical size of transition band is 2050 Hz and that's why CD using 44100 Hz as the sampling rate. $(20000 Hz + 2050 Hz) * 2 = 44100 Hz$ The CD sampling rate, 44100Hz, is the min requirement.
-   Sampling Depth:
    -   Unit: `bits`, e.g. 16 bits standing for $2^{16} = 65536$, representing the sampling types of audio strength.
-   Bitrate (bps, bit per seconds), e.g. `128 kbps`, etc.
    -   CBR (Constant Bitrate)
    -   VBR (Variable Bitrate) : Encoder-dependent



## Space

### Uncompressed

$$Storage\ Space = Sampling\ rate\ *\ Sampling\ depth\ *\ the\ length\ of\ audio\ *\ the\ number\ of\ channels$$

For example, a 4-minute song in WAV format with sampleing rate 44.1KHz, 16 bit sampling depth and two channels.

$$240 (secs) * 44100 * 16 (bits) * 2 (channels) / 8 (to bytes) / 1024 (to Kbytes) / 1024 (to Mbytes) \approx 40.37 MB$$



### Compressed

$$Storage\ Space\ =\ Average\ Bitrate\ *\ Length\ of\
 Audio$$

For example, a 4-minute song with average bitrate 128kps.

$$128 (kbps) * 240 (=60*4 seconds) / 8 (to bytes) / 1024 (to Mbytes) = 3.75 MB$$



## Compressed / Lossy



### MP3 (MPEG-1 Audio Layer 3)

Reference: <https://magiclen.org/linux-lame/>

*   Famous encoder to compress the raw audio to MP3 format is `LAME`.

*   Sampling depth: 16 bits
*   Sampling rate: 48 KHz



### OGG (Vorbis)

Reference: <https://magiclen.org/linux-ogg/>

*   Vorbis tools include Ogg encoder (`oggenc`) and Ogg decoder (`oggdec`).



### AAC (Advanced Audio Coding, based on MPEG-2)

Reference: <https://magiclen.org/linux-faac-2/>

*   Famous encoder : FAAC

*   Sampling depth up to 32 bits.
*   Sampling rate up to 96 KHz.



### HE-AAC (High-Efficiency AAC, based on MPEG-4)

Reference: <https://magiclen.org/linux-fdk-aac-2/>

*   Famous Encoder: FDK AAC
*   HE-AAC = AAC + SBR (Spectral Band Replication) is used in low bits encoding.



### HE-AACv2 (High-Efficiency AAC Version 2, based on MPEG-4)

Reference: <https://magiclen.org/linux-fdk-aac-2/>

*   Famous Encoder: FDK AAC

*   HE-AAC = AAC + SBR (Spectral Band Replication) + PS (Parameteric Stereo) is used in much lower bits encoding.



### Opus

Reference: <https://magiclen.org/linux-opus/>

*   Encoder: `opusenc`; Decoder: `opusdec`.

*   A new tech merges SILK (Skype develoing encoding format) and low latency CELT (Constrained Energy Lapped Transform).
*   Espically good for `real-time` audio transmission.
*   Sampling rate: 6 kps ~ 512 kpbs.



## Lossly Compressed



### FLAC (Free Lossless Audio Codec)

Reference: <https://magiclen.org/linux-flac/>



### ALAC (Apple Lossless)

### APE (Monkey's Audio)

### WMA Lossless (Windows Media Audio 9 Lossless)



## Compare



### Decoder

FFmpeg

*   Basic Transfrom Command ```$ ffmpeg -i file.wav -ar 44100 -ab 16 file_enc.wav```

LAME

Vorbis

FAAC

FDK-AAC

Opus Tools



### Spectrum Analyzing

Audacity




## Transformation



### MFCC

