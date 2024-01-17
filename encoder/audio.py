from scipy.ndimage import binary_dilation
from encoder.support_params import *
from pathlib import Path
from typing import Optional, Union
from warnings import warn
import numpy as np
import librosa
import struct
from IPython.display import Audio

try:
    import webrtcvad
except:
    warn("Unable to import 'webrtcvad'. This package enables noise removal and is recommended.")
    webrtcvad=None

int16_max = (2 ** 15) - 1

def normalize_volume(aud, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    
    dBFS_change = target_dBFS - 10 * np.log10(np.mean(aud ** 2))
    
    if (dBFS_change < 0 and increase_only) or (dBFS_change > 0 and decrease_only):
        return aud
    return aud * (10 ** (dBFS_change / 20))


def trim_long_silences(aud):
    """
    Ensures that segments without voice in the waveform remain no longer than a 
    threshold determined by the VAD parameters in support_params.py.

    :param aud: the raw waveform as a numpy array of floats 
    :return: the same waveform with silences trimmed away (length <= original aud length)
    """
    # Compute the voice detection window size
    samples_per_window = (vad_window_length * sampling_rate) // 1000
    
    # Trim the end of the audio to have a multiple of the window size
    aud = aud[:len(aud) - (len(aud) % samples_per_window)]
    
    # Convert the float waveform to 16-bit mono PCM
    pcm_aude = struct.pack("%dh" % len(aud), *(np.round(aud * int16_max)).astype(np.int16))
    
    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(aud), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_aude[window_start * 2:window_end * 2],
                                         sample_rate=sampling_rate))
    voice_flags = np.array(voice_flags)
    
    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width
    
    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool_)
    
    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)
    
    return aud[audio_mask == True]


def preprocess_aud(path_or_aud: Union[str, Path, np.ndarray],
                   source_sr: Optional[int] = None,
                   normalize: Optional[bool] = True,
                   trim_silence: Optional[bool] = True):
    """
    Applies the preprocessing operations used in training the Speaker Encoder to a waveform 
    either on disk or in memory. The waveform will be resampled to match the data hyperparameters.

    :param path_or_aud: either a filepath to an audio file (many extensions are supported, not 
    just .wav), either the waveform as a numpy array of floats.
    :param source_sr: if passing an audio waveform, the sampling rate of the waveform before 
    preprocessing. After preprocessing, the waveform's sampling rate will match the data 
    hyperparameters. If passing a filepath, the sampling rate will be automatically detected and 
    this argument will be ignored.
    """
    # Load the aud from disk if needed
    if isinstance(path_or_aud, str) or isinstance(path_or_aud, Path):
        aud, source_sr = librosa.load(str(path_or_aud), sr=None)
    else:
        aud = path_or_aud
    
    # Resample the aud if needed
    if source_sr is not None and source_sr != sampling_rate:
        aud = librosa.resample(y=aud, orig_sr=source_sr, target_sr=sampling_rate)
        
    # Apply the preprocessing: normalize volume and shorten long silences
    if normalize:
        aud = normalize_volume(aud, audio_norm_target_dBFS, increase_only=True)
    if webrtcvad and trim_silence:
        aud = trim_long_silences(aud)
    
    return aud


def aud_to_mel_spectrogram(aud):
    """
    Derives a mel spectrogram ready to be used by the encoder from a preprocessed audio waveform.
    Note: this not a log-mel spectrogram.
    """
    frames = librosa.feature.melspectrogram(
        y=aud,
        sr=sampling_rate,
        n_fft=int(sampling_rate * mel_window_length / 1000),
        hop_length=int(sampling_rate * mel_window_step / 1000),
        n_mels=mel_n_channels
    )
    return frames.astype(np.float32).T

# Testing the preprocessing
test = False
if test:
    aud = preprocess_aud("/Volumes/Storage/Git Repos/Real-Time-Voice-Cloning-master/samples/6829_00000.mp3")
    print(aud_to_mel_spectrogram(aud=aud)[0])
    
    from scipy.io.wavfile import write
    write('test.wav', sampling_rate, aud)