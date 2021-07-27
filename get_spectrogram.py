from numba import njit
import numpy as np
import pickle
import librosa
import matplotlib.pyplot as plt
from matplotlib.mlab import specgram
from typing import Tuple, List
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion, iterate_structure
from mygrad import sliding_window_view

def make_spectogram(digital_samples, sampling_rate):
    """
    Makes a spectogram
    
    Parameters
    ----------
    digital_samples : numpy.ndarray, shape-(H, W)
        numpy array of N audio samples
    
    
    Returns
    -------
    spectrogram:numpy.ndarray
        spectrogram is the 2-D array whose rows corresponds to frequencies and whose columns correspond to time. 
        freqs is an array of the frequency values corresponding to the rows
        times is an array of time values corresponding to the columns.
    """
   
     # The desired temporal duration (seconds) of each window of audio data
#     recorded_dt = np.diff(times)[0]
#     window_dt = recorded_dt  # you can adjust this value later


#     # Compute the number of samples that should fit in each
#     # window, so that each temporal window has a duration of `window_dt`
#     # Hint: remember that the audio samples have an associated
#     # sampling rate of 44100 samples per second

#     # Define this as `window_size` (an int)
#     # <COGINST>
#     extend_factor = 1  # adjust this later for trying overlapping windows
#     window_dt *= extend_factor
#     window_size = int(window_dt * 44100)
#     # </COGINST>

#     # Using the above window size and `sliding_window_view`, create an array 
#     # of non-overlapping windows of the audio data.
#     # What should the step-size be so that the windows are non-overlapping?

#     # Define `windowed_audio` to be a 2D array where each row contains the
#     # samples of the recording in a temporal window.
#     # The shape should be (M, N), where M is the number of temporal windows
#     # and N is the number of samples in each window
#     # <COGINST>
#     windowed_audio = sliding_window_view(
#         digital_samples, window_shape=(window_size,), step=window_size // extend_factor
#     )

#     M, N = windowed_audio.shape[:2]
#     ck_for_each_window = np.fft.rfft(windowed_audio, axis=-1)
#     ak_for_each_window = np.absolute(ck_for_each_window) / N
#     ak_for_each_window[:, 1 : (-1 if N % 2 == 0 else None)] *= 2
#     spectrogram = ak_for_each_window.T  # rows: freq, cols: time
#     return spectrogram
#     x = librosa.stft(digital_samples[:,0] / np.linalg.norm(digital_samples))
#     print(x.shape)
    
    sgram_0 = librosa.stft(digital_samples[:,0] / np.linalg.norm(digital_samples))
    mag_0,_ = librosa.magphase(sgram_0)
    mel_0 = librosa.amplitude_to_db(librosa.feature.melspectrogram(S=mag_0, sr=sampling_rate))
    sgram_1 = librosa.stft(digital_samples[:,1] / np.linalg.norm(digital_samples))
    mag_1,_ = librosa.magphase(sgram_1)
    mel_1 = librosa.amplitude_to_db(librosa.feature.melspectrogram(S=mag_1, sr=sampling_rate))
#     print(mel_0.shape)
    
#     spec0, freqs, times = specgram(digital_samples[:,0], NFFT=256, noverlap=0, Fs=sampling_rate)
#     np.clip(spec0, a_min=1e-20, a_max=None, out=spec0)
#     np.log(spec0, out=spec0)
#     spec1, freqs, times = specgram(digital_samples[:,1], NFFT=256, noverlap=0, Fs=sampling_rate)
#     np.clip(spec1, a_min=1e-20, a_max=None, out=spec1)
#     np.log(spec1, out=spec1)
    return np.stack((mel_0, mel_1), axis=-1)