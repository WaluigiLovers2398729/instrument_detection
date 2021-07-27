# Creating functions for converting all variety of audio recordings, 
# be them recorded from the microphone or digital audio files, into a 
# NumPy-array of digital samples. (Nicholas Won)

import numpy as np
from microphone import record_audio
import pathlib
from scipy.io import wavfile

def micsample(listentime):
    """
    Uses the microphone to record audio and returns a numpy array
    of digital samples

    Parameters
    ----------
    listentime : float
        length of recording in seconds
        
    Returns
    -------
    (samples, times) : Tuple[ndarray, ndarray]
        the shape-(N,) array of samples and the corresponding shape-(N,) array of times
    """
    frames, sampling_rate = record_audio(listentime)
    samples = np.hstack([np.frombuffer(i, np.int16) for i in frames])
    times = np.arange(samples.size) / sampling_rate
    return samples, times

def filesample(filename):
    """
    Uses librosa to read in audio samples from a sound file and returns
    a numpy array of digital samples

    Parameters
    ----------
    filename : string 
        file name of audio file to be analyzed
        
    Returns
    -------
    (samples, times) : Tuple[ndarray, ndarray]
        the shape-(N,) array of samples and the corresponding shape-(N,) array of times
    """
    sampling_rate, samples = wavfile.read(filename)
    times = np.arange(samples.size) / sampling_rate
    return samples, times