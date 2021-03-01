#!pip install soundfile
#!pip install librosa==0.8.0
#!pip install scipy==1.5.2
# use 3 lines above in console or Google Colaboratory (colab)

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display as display
import librosa.feature
import soundfile as sf
from scipy.signal import resample
import math
import torch

##################################
# audio generation utils
##################################

def extract_audio(filename):
    """
    Extract audio given the filename (.wav, .flac, etc format)
    """

    audio, rate = sf.read(filename, always_2d=True)
    audio = np.reshape(audio, (1, -1))
    audio = audio[0]

    time = []
    t = 0
    for i in range(len(audio)):
        time.append(t)
        t += 1 / rate
    return audio, time, rate


def generate_spec(audio_sequence, rate, n_fft=2048, hop_length=512):
    """
    Generate spectrogram using librosa
    audio_sequence: list representing waveform
    rate: sampling rate (16000 for all LibriSpeech audios)
    nfft and hop_length: stft parameters
    """
    S = librosa.feature.melspectrogram(audio_sequence, sr=rate, n_fft=n_fft, hop_length=hop_length, n_mels=128, fmin=20,
                                       fmax=8300)
    log_spectra = librosa.power_to_db(S, ref=np.mean, top_db=80)
    return log_spectra

def reconstruct_wave(spec, rate=16000, normalize_data=False):
    """
    Reconstruct waveform
    spec: spectrogram generated using Librosa
    rate: sampling rate
    """
    power = librosa.db_to_power(spec, ref=5.0)
    audio = librosa.feature.inverse.mel_to_audio(power, sr=rate, n_fft=2048, hop_length=512)
    out_audio = audio / np.max(audio) if normalize_data else audio
    return out_audio

def normalize(spec, eps=1e-6):
    """
    Normalize spectrogram with zero mean and unitary variance
    spec: spectrogram generated using Librosa
    """

    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    return spec_norm, (mean, std)

def minmax_scaler(spec):
  """
  min max scaler over spectrogram
  """
  spec_max = np.max(spec)
  spec_min = np.min(spec)

  return (spec-spec_min)/(spec_max - spec_min), (spec_max, spec_min)

def linear_scaler(spec):
  """
  linear scaler over spectrogram
  min value -> -1 and max value -> 1
  """
  spec_max = np.max(spec)
  spec_min = np.min(spec)
  m = 2/(spec_max-spec_min)
  n = (spec_max + spec_min)/(spec_min-spec_max)

  return m*spec + n, (m, n)

def split_specgram(example, clean_example, frames = 11):
  """
  Split specgram in groups of frames, the purpose is prepare data for the LSTM model input

  example: reverberant spectrogram
  clean_example: clean or target spectrogram

  return data input to the LSTM model and targets
  """
  clean_spec = clean_example[0, :, :]
  rev_spec = example[0, :, :]

  n, m = clean_spec.shape

  targets = torch.zeros((m-frames+1, n))
  data = torch.zeros((m-frames+1, n*frames))
  
  idx_target = frames//2
  for i in range(m-frames+1):
    try:
      targets[i, :] = clean_spec[:, idx_target]
      data[i, :] = torch.reshape(rev_spec[:, i:i+frames], (1, -1))[0, :]
      idx_target += 1
    except (IndexError):
      pass
  return data, targets

def split_realdata(example, frames = 11):

  """
  Split 1 specgram in groups of frames, the purpose is prepare data for the LSTM and MLP model input

  example: reverberant ''real'' (not simulated) spectrogram

  return data input to the LSTM or MLP model 
  """
  
  rev_spec = example[0, :, :]
  n, m = rev_spec.shape
  data = torch.zeros((m-frames+1, n*frames))
  for i in range(m-frames+1):
    data[i, :] = torch.reshape(rev_spec[:, i:i+frames], (1, -1))[0, :]
  return data

def prepare_data(X, y, display = False):
  """
  Use split_specgram to split all specgrams
  X: tensor containing reverberant spectrograms
  y: tensor containing target spectrograms
  """

  data0, target0 = split_specgram(X[0, :, :, :], y[0, :, :, :])

  total_data = data0.cuda()
  targets = target0.cuda()
  
  for i in range(1, X.shape[0]):
    
    if display: 
      print("Specgram n°" + str(i)) 
      
    data_i, target_i = split_specgram(X[i, :, :, :], y[i, :, :, :])
    
    total_data = torch.cat((total_data, data_i.cuda()), 0)
    targets = torch.cat((targets, target_i.cuda()), 0)

  return  total_data, targets


#################################
# reverberation utils
#################################

def zero_pad(x, k):
    """
    add k zeros to x signal
    """
    return np.append(x, np.zeros(k))


def awgn(signal, regsnr):
    """
    add random noise to signal
    regsnr: signal to noise ratio
    """
    sigpower = sum([math.pow(abs(signal[i]), 2) for i in range(len(signal))])
    sigpower = sigpower / len(signal)
    noisepower = sigpower / (math.pow(10, regsnr / 10))
    sample = np.random.normal(0, 1, len(signal))
    noise = math.sqrt(noisepower) * sample
    return noise


def discrete_conv(x, h, x_fs, h_fs, snr=30, aug_factor=1):
    """
    Convolution using fft
    x: speech waveform
    h: RIR waveform
    x_fs: speech signal sampling rate (if is not 16000 the signal will be resampled)
    h_fs: RIR signal sampling rate (if is not 16000 the signal will be resampled)

    Based on https://github.com/vtolani95/convolution/blob/master/reverb.py
    """

    numSamples_h = round(len(h) / h_fs * 16000)
    numSamples_x = round(len(x) / x_fs * 16000)

    if h_fs != 16000:
        h = resample(h, numSamples_h) # resample RIR

    if x_fs != 16000:
        x = resample(x, numSamples_x) # resample speech signal

    L, P = len(x), len(h)
    h_zp = zero_pad(h, L - 1)
    x_zp = zero_pad(x, P - 1)
    X = np.fft.fft(x_zp)
    output = np.fft.ifft(X * np.fft.fft(h_zp)).real
    output = aug_factor * output + x_zp
    output = output + awgn(output, snr)
    return output

###################################
#plot utils
###################################

def graph_spec(spec, rate=16000, title=False):
    """
    plot spectrogram
    spec: spectrogram generated using Librosa
    rate: sampling rate
    """
    plt.figure()
    display.specshow(spec, sr=rate, y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    if (title):
        plt.title('Log-Power spectrogram')
    plt.tight_layout()

def plot_time_wave(audio, rate=16000):
    """
    plot waveform given speech audio
    audio: array containing waveform
    rate: sampling rate

    """
    time = [0]
    for i in range(1, len(audio)):
        time.append(time[i - 1] + 1 / rate)

    plt.figure()
    plt.plot(time, audio)
    plt.xlabel("Time (secs)")
    plt.ylabel("Power")
