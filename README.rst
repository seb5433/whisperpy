========
Overview
========

Python speaker diarization with OpenAI Whisper ASR & pyannote.audio for accurate multi‑speaker transcription &
labeling.

* Free software: MIT license

Installation
============

::

    pip install whisperpy-diarizer

You can also install the in-development version with::

    pip install git+ssh://git@github.com:seb5433/whisperpy.git

Documentation
=============


To use the project:

.. code-block:: python

    import whisperpy_diarizer
    whisperpy_diarizer.diar(...)



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
