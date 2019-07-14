import os
import matplotlib.pyplot as plt
import numpy as np
import librosa
#import pyrubberband as pyrb
import json
import cv2
####################### PARAMS ####################### 
root = input('[SPEC]Enter input(root) folder:')
inf = os.path.join(root, 'wav')
outfd = os.path.join(root), 'spec_200x200')
if not os.path.exists(outfd): os.makedirs(outfd)
######################################################
IMG_ONLY = False
datainf = input('[DATASET] Enter input data file:')
specf = outfd#input('Enter spec folder:')
dataf = 'label.txt'
namef = 'filename.txt'
imf = 'spec.npy'

genres=['Ambient', 'Blues', 'Chill Out','Cinematic','Classical',
'Dance','Drum And Bass','Dubstep','Electro','Ethnic','Funk', 
'Pop','Rap','Techno','Weird']
###################################################### 
####################### SPECTROGRAM ####################### 
def check_tempo(inf, wav_file):
    sig, sr = librosa.load(os.path.join(inf,wav_file), sr=44100)
    tempo, beats = librosa.beat.beat_track(y=sig, sr=sr)
    print('Original_tempo:',tempo)
    return tempo

def stretching(inf, wav_file, target_tempo=120):
    # Load WAV Files
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

def check_and_draw_spectrogram(inf, outfd, txt = 'zero_files.txt'):
    max_length = 10 # clip & padding to 10 sec.
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

def generate_dataset(IMG_ONLY, inf, specf):
    print('------------------ GENERATE DATASET ------------------')
    if(IMG_ONLY):
        imarr = []
        for f in os.listdir(specf):
            if f.startswith('spectrogram_'):
                with open(inf, 'r') as i:
                    for line in i.readlines():
                        if line=="\n": continue
                        data = json.loads(line)
                        if data["index"] != f.split('.png')[0].strip('spectrogram_'): continue
                        else:
                            print(data["index"])
                            for t in data['tags']:  
                                if t.strip('Loops').strip(' ') in genres:       
                                    im = cv2.imread(os.path.join(specf, f))
                                    imarr.append(np.array(im))
                                    print(np.array(imarr).shape)
                                    break
                            break
        np.save(imf, imarr)

    else:
        with open(dataf, 'w') as o:
            with open(namef, 'w') as oo:
                imarr = []
                for f in os.listdir(specf):
                    #if f.startswith('spectrogram_0001'): break
                    if f.startswith('spectrogram_'):
                        with open(datainf, 'r') as i:
                            for line in i.readlines():
                                if line=="\n": continue
                                data = json.loads(line)
                                if data["index"] != f.split('.png')[0].strip('spectrogram_'): continue
                                else:
                                    print(data["index"])
                                    for t in data['tags']:  
                                        if t.strip('Loops').strip(' ') in genres:
                                            tt = t.strip('Loops').strip(' ')
                                            o.write(str(genres.index(tt))+'\n')
                                            oo.write(data["index"]+'\n')
                                                
                                            im = cv2.imread(os.path.join(specf, f))
                                            #print(im.shape)
                                            imarr.append(np.array(im))
                                            print(np.array(imarr).shape)
                                            print('done')
                                            break
                                    break
                np.save(imf, imarr)

##################################################

check_and_draw_spectrogram(inf, outfd)
generate_dataset(IMG_ONLY, inf, specf)
