import os
#import wave
#import pylab
import matplotlib.pyplot as plt
import numpy as np
import librosa
import math
import pyrubberband as pyrb

inf = input('Enter input folder:')
outfd = 'I:\_spectrogram_subset'
if not os.path.exists(outfd):
    os.makedirs(outfd)

'''
def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate
'''
def stretching(inf, wav_file, target_tempo=120):
    # Load WAV Files
    ##sound_info, frame_rate = get_wav_info(os.path.join(inf,wav_file))
    sig, sr = librosa.load(os.path.join(inf,wav_file), sr=44100)
    tempo, beats = librosa.beat.beat_track(y=sig, sr=sr)
    print('Original_tempo:',tempo)
    return pyrb.time_stretch(sig, sr, float(target_tempo)/tempo), sr

def graph_spectrogram(outf, wav_file, sig, sr, max_length=0):
    if(max_length > 0):
        # Padding
        sig = np.append(sig, np.zeros(max_length - sig.shape[0]))

    # Plot

    plt.figure(num=None, figsize=(8, 6))
    plt.subplot(111)
    plt.axis('off')
    plt.margins(0.0)
    plt.specgram(sig, Fs=sr)
    plt.savefig(os.path.join(outfd,'spectrogram_%06d.png' % int(wav_file.split('.')[0])), bbox_inches='tight', pad_inches=0.0)
    plt.close()
    # pylab.figure(num=None, figsize=(17, 10))
    # pylab.subplot(111)
    # #pylab.title('spectrogram of %r' % wav_file)
    # pylab.axis('off')
    # pylab.margins(0.0)
    # pylab.specgram(sig, Fs=sr)
    # pylab.savefig(os.path.join(final_outf ,'spectrogram_%06d.png' % int(wav_file.split('.')[0])), bbox_inches='tight', pad_inches=0.0)
    # pylab.close()
    # #pylab.show()

ys, srs, names = [], [], []
max_length = 0
folder = os.listdir(inf)
for wav_file in folder:
    if wav_file.endswith('.wav'):
        print(os.path.join(inf,wav_file))
        y, sr = stretching(inf, wav_file)
        ys.append(y)
        srs.append(sr)
        names.append(wav_file)
        max_length = max(max_length, len(y))
 
for (y, sr, name) in zip(ys, srs, names):
    graph_spectrogram(outfd, name, y, sr, max_length)

'''
folder = os.listdir(inf)
for wav_file in folder:
    if os.path.isdir(os.path.join(inf, wav_file)):
        for wav in os.listdir(os.path.join(inf, wav_file)):
            if wav.endswith('.wav'):
                print(os.path.join(inf, wav_file, wav))
                graph_spectrogram(outfd, os.path.join(inf, wav_file), wav) 
    if wav_file.endswith('.wav'):
        print(os.path.join(inf,wav_file))
        graph_spectrogram(outfd, inf, wav_file)
'''

    