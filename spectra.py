import os
import wave
import pylab
#import matplotlib.pylab as plt
import numpy as np
import librosa
import math

inf = input('Enter input folder:')
outf = '_spectrogram'
if not os.path.exists(outf):
    os.makedirs(outf)

def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate

def graph_spectrogram(outf, inf, wav_file):
    # Load WAV Files
    ##sound_info, frame_rate = get_wav_info(os.path.join(inf,wav_file))
    sound_info, frame_rate = librosa.load(os.path.join(inf,wav_file), sr=44100)

    # Padding
    div = sound_info.shape[0] / frame_rate
    div = math.ceil(div/3) * 3
    # if sound_info.shape[0] > div*frame_rate:
    #     print(sound_info.shape[0], div*frame_rate)
    sound_info = np.append(sound_info, np.zeros(div*frame_rate - sound_info.shape[0]))
    
    # Check folder
    final_outf = os.path.join(outf, str(div))
    if not os.path.exists(final_outf):
        os.makedirs(final_outf)
    # Plot
    pylab.figure(num=None, figsize=(17, 10))
    pylab.subplot(111)
    #pylab.title('spectrogram of %r' % wav_file)
    pylab.axis('off')
    pylab.margins(0.0)
    pylab.specgram(sound_info, Fs=frame_rate)
    pylab.savefig(os.path.join(final_outf ,'spectrogram_%06d.png' % int(wav_file.split('.')[0])), bbox_inches='tight', pad_inches=0.0)
    pylab.close()
    #pylab.show()
    
    

folder = os.listdir(inf)
for wav_file in folder:
    if os.path.isdir(os.path.join(inf, wav_file)):
        for wav in os.listdir(os.path.join(inf, wav_file)):
            if wav.endswith('.wav'):
                print(os.path.join(inf, wav_file, wav))
                graph_spectrogram(outf, os.path.join(inf, wav_file), wav) 
    if wav_file.endswith('.wav'):
        print(os.path.join(inf,wav_file))
        graph_spectrogram(outf, inf, wav_file)

# for root, dirs, files in os.walk(inf):
#     for wav_file in files:
#         print(os.path.join(inf,wav_file))
#         if wav_file.endswith('.wav'):
#             graph_spectrogram(outf, inf, wav_file)
    
    