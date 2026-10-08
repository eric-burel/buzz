---
title: Installation
sidebar_position: 2
---

To install Buzz, download the latest version for your operating
system. Buzz is available on **Mac**, **Windows**, and **Linux**.

### macOS

Download the `.dmg` from the [SourceForge](https://sourceforge.net/projects/buzz-captions/files/).

> **Intel Macs:** Buzz now requires Apple silicon. The last version to support
> Intel Macs is **1.4.5**.

### Windows

Get the installation files from the [SourceForge](https://sourceforge.net/projects/buzz-captions/files/).

App is not signed, you will get a warning when you install it. Select `More info` -> `Run anyway`.

## Linux

Buzz is available as a [Flatpak](https://flathub.org/apps/io.github.chidiwilliams.Buzz) or a [Snap](https://snapcraft.io/buzz). 

To install flatpak, run:
```shell
flatpak install flathub io.github.chidiwilliams.Buzz
```

[![Download on Flathub](https://flathub.org/api/badge?svg&locale=en)](https://flathub.org/en/apps/io.github.chidiwilliams.Buzz)

To install snap, run:
```shell
sudo apt-get install libportaudio2 libcanberra-gtk-module libcanberra-gtk3-module
sudo snap install buzz
sudo snap connect buzz:password-manager-service
```

[![Get it from the Snap Store](https://snapcraft.io/static/images/badges/en/snap-store-black.svg)](https://snapcraft.io/buzz)

## PyPI

```shell
pip install buzz-captions
python -m buzz
```

On Linux install system dependencies you may be missing
```
sudo apt-get install --no-install-recommends libyaml-dev libtbb-dev libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-xinerama0 libxcb-shape0 libxcb-cursor0 libportaudio2 gettext libpulse0 ffmpeg
```
On versions prior to Ubuntu 24.04 install `sudo apt-get install --no-install-recommends libegl1-mesa`

## Kyutai STT (source installation)

Kyutai support is an optional source-install extra. From the Buzz repository,
install it with:

```shell
uv sync --extra kyutai
```

Restart Buzz; Kyutai STT will then be available for file transcription under
Preferences > Models. Buzz supports the English/French `kyutai/stt-1b-en_fr`
model and the English-only `kyutai/stt-2.6b-en` model. The models are several
gigabytes and are downloaded on demand. The PyTorch implementation uses CUDA
when available and falls back to CPU; CPU transcription can be very slow.

The initial integration supports file transcription only. Kyutai does not
support Buzz's translation task. Live microphone streaming, semantic VAD, and
prebuilt Buzz installers are not included yet.


## CUDA GPU Acceleration

Since version `1.4.6` Nvidia CUDA GPU acceleration is no longer included in base package of the Buzz and has to be installed separately on Linux and Windows. You will get a prompt to do so if Nvidia GPU is detected, but you can also install CUDA support manually by going to `Help -> About Buzz -> Install CUDA Acceleration`. CUDA is not needed for `whisper.cpp` and if you do not have Nvidia GPU.
