# -*- coding: utf-8 -*-
#
# This file is part of SIDEKIT.
#
# SIDEKIT is a python package for speaker verification.
# Home page: http://www-lium.univ-lemans.fr/sidekit/
#
# SIDEKIT is a python package for speaker verification.
# Home page: http://www-lium.univ-lemans.fr/sidekit/
#    
# SIDEKIT is free software: you can redistribute it and/or modify
# it under the terms of the GNU LLesser General Public License as 
# published by the Free Software Foundation, either version 3 of the License, 
# or (at your option) any later version.
#
# SIDEKIT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with SIDEKIT.  If not, see <http://www.gnu.org/licenses/>.

"""
Copyright 2014-2016 Anthony Larcher and Sylvain Meignier

:mod:`frontend` provides methods to process an audio signal in order to extract
useful parameters for speaker verification.
"""

import numpy
import scipy
import sidekit
from scipy.fftpack.realtransforms import dct
from ringBuffer import Flow

PARAM_TYPE = numpy.float32

__author__ = "Anthony Larcher and Sylvain Meignier"
__copyright__ = "Copyright 2014-2016 Anthony Larcher and Sylvain Meignier"
__license__ = "LGPL"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'


def coroutine(func):
    """
    Decorator that allows to forget about the first call of a coroutine .next()
    method or .send(None)
    This call is done inside the decorator
    :param func: the coroutine to decorate
    """
    def start(*args,**kwargs):
        cr = func(*args,**kwargs)
        next(cr)
        return cr
    return start


def hz2mel(f, htk=True):
    """Convert an array of frequency in Hz into mel.

    :param f: frequency to convert

    :return: the equivalence on the mel scale.
    """
    if htk:
        return 2595 * numpy.log10(1 + f / 700.)
    else:
        f = numpy.array(f)

        # Mel fn to match Slaney's Auditory Toolbox mfcc.m
        # Mel fn to match Slaney's Auditory Toolbox mfcc.m
        f_0 = 0.
        f_sp = 200. / 3.
        brkfrq = 1000.
        brkpt = (brkfrq - f_0) / f_sp
        logstep = numpy.exp(numpy.log(6.4) / 27)

        linpts = f < brkfrq

        z = numpy.zeros_like(f)
        # fill in parts separately
        z[linpts] = (f[linpts] - f_0) / f_sp
        z[~linpts] = brkpt + (numpy.log(f[~linpts] / brkfrq)) / numpy.log(logstep)

        if z.shape == (1,):
            return z[0]
        else:
            return z


def mel2hz(z, htk=True):
    """Convert an array of mel values in Hz.

    :param m: ndarray of frequencies to convert in Hz.

    :return: the equivalent values in Hertz.
    """
    if htk:
        return 700. * (10 ** (z / 2595.) - 1)
    else:
        z = numpy.array(z, dtype=float)
        f_0 = 0
        f_sp = 200. / 3.
        brkfrq = 1000.
        brkpt = (brkfrq - f_0) / f_sp
        logstep = numpy.exp(numpy.log(6.4) / 27)

        linpts = (z < brkpt)

        f = numpy.zeros_like(z)

        # fill in parts separately
        f[linpts] = f_0 + f_sp * z[linpts]
        f[~linpts] = brkfrq * numpy.exp(numpy.log(logstep) * (z[~linpts] - brkpt))

        if f.shape == (1,):
            return f[0]
        else:
            return f


def trfbank(fs, nfft, lowfreq, maxfreq, nlinfilt, nlogfilt, midfreq=1000):
    """Compute triangular filterbank for cepstral coefficient computation.

    :param fs: sampling frequency of the original signal.
    :param nfft: number of points for the Fourier Transform
    :param lowfreq: lower limit of the frequency band filtered
    :param maxfreq: higher limit of the frequency band filtered
    :param nlinfilt: number of linear filters to use in low frequencies
    :param  nlogfilt: number of log-linear filters to use in high frequencies
    :param midfreq: frequency boundary between linear and log-linear filters

    :return: the filter bank and the central frequencies of each filter
    """
    # Total number of filters
    nfilt = nlinfilt + nlogfilt

    # ------------------------
    # Compute the filter bank
    # ------------------------
    # Compute start/middle/end points of the triangular filters in spectral
    # domain
    frequences = numpy.zeros(nfilt + 2, dtype=PARAM_TYPE)
    if nlogfilt == 0:
        linsc = (maxfreq - lowfreq) / (nlinfilt + 1)
        frequences[:nlinfilt + 2] = lowfreq + numpy.arange(nlinfilt + 2) * linsc
    elif nlinfilt == 0:
        low_mel = hz2mel(lowfreq)
        max_mel = hz2mel(maxfreq)
        mels = numpy.zeros(nlogfilt + 2)
        # mels[nlinfilt:]
        melsc = (max_mel - low_mel) / (nfilt + 1)
        mels[:nlogfilt + 2] = low_mel + numpy.arange(nlogfilt + 2) * melsc
        # Back to the frequency domain
        frequences = mel2hz(mels)
    else:
        # Compute linear filters on [0;1000Hz]
        linsc = (min([midfreq, maxfreq]) - lowfreq) / (nlinfilt + 1)
        frequences[:nlinfilt] = lowfreq + numpy.arange(nlinfilt) * linsc
        # Compute log-linear filters on [1000;maxfreq]
        low_mel = hz2mel(min([1000, maxfreq]))
        max_mel = hz2mel(maxfreq)
        mels = numpy.zeros(nlogfilt + 2, dtype=PARAM_TYPE)
        melsc = (max_mel - low_mel) / (nlogfilt + 1)

        # Verify that mel2hz(melsc)>linsc
        while mel2hz(melsc) < linsc:
            # in this case, we add a linear filter
            nlinfilt += 1
            nlogfilt -= 1
            frequences[:nlinfilt] = lowfreq + numpy.arange(nlinfilt) * linsc
            low_mel = hz2mel(frequences[nlinfilt - 1] + 2 * linsc)
            max_mel = hz2mel(maxfreq)
            mels = numpy.zeros(nlogfilt + 2, dtype=PARAM_TYPE)
            melsc = (max_mel - low_mel) / (nlogfilt + 1)

        mels[:nlogfilt + 2] = low_mel + numpy.arange(nlogfilt + 2) * melsc
        # Back to the frequency domain
        frequences[nlinfilt:] = mel2hz(mels)

    heights = 2. / (frequences[2:] - frequences[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = numpy.zeros((nfilt, int(numpy.floor(nfft / 2)) + 1), dtype=PARAM_TYPE)
    # FFT bins (in Hz)
    n_frequences = numpy.arange(nfft) / (1. * nfft) * fs

    for i in range(nfilt):
        low = frequences[i]
        cen = frequences[i + 1]
        hi = frequences[i + 2]

        lid = numpy.arange(numpy.floor(low * nfft / fs) + 1, numpy.floor(cen * nfft / fs) + 1, dtype=numpy.int)
        left_slope = heights[i] / (cen - low)
        rid = numpy.arange(numpy.floor(cen * nfft / fs) + 1,
                           min(numpy.floor(hi * nfft / fs) + 1, nfft), dtype=numpy.int)
        right_slope = heights[i] / (hi - cen)
        fbank[i][lid] = left_slope * (n_frequences[lid] - low)
        fbank[i][rid[:-1]] = right_slope * (hi - n_frequences[rid[:-1]])

    return fbank, frequences


@coroutine
def mfcc(next):
    try:

        # Initialize the coroutine
        lowfreq=100
        maxfreq=4000
        nlinfilt=0
        nlogfilt=24
        nwin=0.025
        fs=8000
        nceps=13
        prefac=0.97
        window_length = int(round(nwin * fs))

        # Compute the filter bank here
        n_fft = 2 ** int(numpy.ceil(numpy.log2(int(round(nwin * fs)))))
        fbank = trfbank(fs, n_fft, lowfreq, maxfreq, nlinfilt, nlogfilt)[0]
        hanning_window = numpy.hanning(window_length)

        while True:

            # Get the input as a Flow object
            buf = yield
            buf.data = buf.data.squeeze()
            # Pre-emphasis filtering is applied
            buf.data -= numpy.c_[buf.data[numpy.newaxis, :][..., :1],
                                 buf.data[numpy.newaxis, :][..., :-1]].squeeze() * prefac

            # Compute the log-energy
            buf.energy = numpy.array([numpy.log((buf.data ** 2).sum())])

            # Apply Hanning windowing
            buf.data *= hanning_window

            # FFT
            fft = numpy.fft.rfft(buf.data, n_fft, axis=-1)

            # Power spectrum and flitering
            mspec = numpy.log(numpy.dot(fft.real ** 2 + fft.imag ** 2, fbank.T))

            # Use the DCT to 'compress' the coefficients (spectrum -> cepstrum domain)
            # The C0 term is removed as it is the constant term
            buf.data = scipy.fftpack.realtransforms.dct(mspec, type=2, norm='ortho', axis=-1)[1:nceps + 1]

            buf.data = buf.data[:, numpy.newaxis]
            next.send(buf)

    except GeneratorExit:
        next.close()


if __name__ == '__main__':
    import numpy
    import ringBuffer
    from testPipeline import audio_reader, sink
    from mfcc import mfcc

    output = sink()
    buf_cep = ringBuffer.ring_buffer(output, 2, 0, 13)
    cep = mfcc(buf_cep)
    buf_sig = ringBuffer.ring_buffer(cep, 200, 80)
    inp = audio_reader(buf_sig, "taaa.wav")
    next(inp)