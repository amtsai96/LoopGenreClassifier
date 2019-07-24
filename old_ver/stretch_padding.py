import os
#import wave
#from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import librosa
#import pyrubberband as pyrb

inf = input('Enter input folder:')
#mode = input('enter train/test:')
outfd = os.path.join(inf, 'spec_200x200/')#+mode
#outfd_wav = 'I:\_loops\subset\stretched'
if not os.path.exists(outfd): os.makedirs(outfd)
#if not os.path.exists(outfd_wav): os.makedirs(outfd_wav)

def check_tempo(inf, wav_file):
    sig, sr = librosa.load(os.path.join(inf,wav_file), sr=44100)
    tempo, beats = librosa.beat.beat_track(y=sig, sr=sr)
    print('Original_tempo:',tempo)
    return tempo

def stretching(inf, wav_file, target_tempo=120):
    # Load WAV Files
    ##sound_info, frame_rate = get_wav_info(os.path.join(inf,wav_file))
    sig, sr = librosa.load(os.path.join(inf,wav_file), sr=44100)
    tempo, beats = librosa.beat.beat_track(y=sig, sr=sr)
    #print('Original_tempo:',tempo)
    return librosa.effects.time_stretch(sig, float(target_tempo)/tempo), sr
    #return pyrb.time_stretch(sig, sr, float(target_tempo)/tempo), sr

def graph_spectrogram(outf, wav_file, sig, sr, max_length=0):
    if(max_length > 0):
        if sig.shape[0] > max_length:
            sig = sig[:max_length]
        else:
            # Padding
            #print('Length after Padding:', max_length)
            sig = np.append(sig, np.zeros(max_length - sig.shape[0]))

    size = 2
    # Plot
    plt.figure(num=None, figsize=(size, size))
    plt.subplot(111)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.axis('off')
    plt.margins(0.0)
    #plt.tight_layout()
    plt.specgram(sig, Fs=sr)
    plt.savefig(os.path.join(outfd,'spectrogram_%06d.png' % int(wav_file.split('.')[0])), bbox_inches='tight', pad_inches=0.0)
    plt.close()


'''
max_length = 0
folder = os.listdir(inf)

with open(txt,'w') as o:
    for wav_file in folder:
        if wav_file.endswith('.wav'):
            print(wav_file)
            if check_tempo(inf, wav_file) == 0:
                print('tempo=0!!!')
                o.write(wav_file+'\n')
                #break
'''
'''
ys, srs, names = [], [], []
max_length = 0
folder = os.listdir(inf)
for wav_file in folder:
    if wav_file.endswith('.wav'):
        print(wav_file)
        y, sr = stretching(inf, wav_file)
        #wavfile.write(os.path.join(outfd_wav, wav_file.split('.')[0]+'_stretched'+'.wav'), sr, y)
        ys.append(y)
        srs.append(sr)
        names.append(wav_file)
        max_length = max(max_length, len(y))
print('-------------------')
for (y, sr, name) in zip(ys, srs, names):
    graph_spectrogram(outfd, name, y, sr, max_length)
    print(name + ' done.')
'''
txt = 'zero_files.txt'
max_length = 10
folder = os.listdir(inf)
for wav_file in folder:
    if wav_file.endswith('.wav'):
        print(wav_file)
        if check_tempo(inf, wav_file) == 0:
            with open(txt, 'a') as o:
                o.write(wav_file+'\n')
            print('tempo = 0!!!')
            continue
        y, sr = stretching(inf, wav_file)
        graph_spectrogram(outfd, wav_file, y, sr, max_length*sr)
        print(wav_file + ' done.')
'''
#origin ver.
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

    