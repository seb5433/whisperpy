========
Overview
========

Python speaker diarization with OpenAI Whisper ASR & pyannote.audio for accurate multi‑speaker transcription &
labeling.

**Now includes a CLI tool for easy audio transcription!**

* Free software: MIT license

Installation
============

**System Requirements:**

WhisperPy requires cmake and build tools to compile some dependencies. Install them first:

**Manjaro/Arch Linux:**
::

    sudo pacman -S cmake base-devel

**Ubuntu/Debian:**
::

    sudo apt update
    sudo apt install cmake build-essential

**CentOS/RHEL/Fedora:**
::

    # Fedora
    sudo dnf install cmake gcc-c++ make
    
    # CentOS/RHEL
    sudo yum install cmake gcc-c++ make

**macOS:**
::

    # Using Homebrew
    brew install cmake
    
    # Or install Xcode Command Line Tools
    xcode-select --install

**Windows:**
::

    # Install Visual Studio Build Tools and CMake
    # Or use conda: conda install cmake

**Package Installation:**

After installing system dependencies:

::

    pip install -e .

**Alternative: Using Conda (Recommended for complex environments):**

::

    # This handles cmake and other dependencies automatically
    conda install -c conda-forge cmake
    pip install -e .

**Note:** This package includes both audio transcription (OpenAI Whisper) AND speaker diarization (pyannote.audio) as core features.

Documentation
=============

Command Line Interface
----------------------

After installation, you can use the `whipy` command to transcribe audio files with automatic speaker identification:

**Basic Usage:**

::

    whipy transcribe audio_file.wav

This will:
1. Transcribe the audio using OpenAI Whisper
2. Identify different speakers using pyannote.audio
3. Save results with speaker labels as `audio_file.json`

**HuggingFace Token Setup:**

For speaker diarization, set your HuggingFace token:

::

    whipy set-token YOUR_HUGGINGFACE_TOKEN

Get your token from: https://huggingface.co/settings/tokens

**Command Options:**

* ``-m, --model``: Choose Whisper model (tiny, base, small, medium, large, large-v2, large-v3). Default: base
* ``-o, --output``: Specify output file path. Default: same as input with .json extension  
* ``-l, --language``: Specify language code for better accuracy. Auto-detect if not specified
* ``-f, --format``: Output format - 'json' for structured data or 'txt' for readable text. Default: json
* ``-v, --verbose``: Enable verbose output to see processing details

**Examples:**

::

    # Basic transcription with speaker identification (default behavior)
    whipy transcribe my_audio.mp3
    
    # Use a larger model with JSON output (default)
    whipy transcribe -m large my_audio.wav
    
    # Save as text format instead of JSON
    whipy transcribe -f txt my_audio.mp3
    
    # Specify output file and language
    whipy transcribe -o transcript.json -l en my_audio.m4a
    
    # Verbose output to see what's happening
    whipy transcribe -v -m medium my_audio.flac

**Output Formats:**

* **JSON format** (default): Structured output perfect for LLMs with metadata, speaker segments, conversation timeline, and full transcript
* **TXT format**: Human-readable conversation with timestamps and speaker labels

**Supported audio formats:** WAV, MP3, M4A, FLAC, OGG, and many others supported by FFmpeg.

Troubleshooting
===============

**Installation Issues:**

If you get cmake-related errors during installation:

::

    # Make sure cmake is installed (see System Requirements above)
    cmake --version
    
    # If sentencepiece fails to compile, try installing via conda:
    conda install -c conda-forge sentencepiece
    pip install -e .

**Runtime Issues:**

* **"HUGGINGFACE_TOKEN not found"**: Set your token using ``whipy set-token YOUR_TOKEN``
* **CUDA/GPU issues**: WhisperPy works on CPU by default. For GPU acceleration, ensure PyTorch CUDA is properly installed
* **Audio format issues**: Convert your audio to a common format like WAV or MP3 if you encounter format-related errors

Library Usage
-------------

For programmatic use:

.. code-block:: python

    import whisperpy_diarizer
    # Use the CLI functions programmatically
    from whisperpy_diarizer.cli import perform_diarization, match_segments_with_speakers



Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
