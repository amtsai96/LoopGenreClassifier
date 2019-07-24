import os
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
#import pyrubberband as pyrb
import json
import cv2
import pylab

SPEC = True#False

GEN = False#True
IMG_ONLY = False
FOLDER_MODE = True if int(input('FOLDER_MODE?(True=1)')) == 1 else False
####################### PARAMS #######################
width = int(input('Enter spectrogram output size(width):'))
height = int(input('Enter spectrogram output size(height):'))

mel = True if int(input('mel=1 or not=0:'))>0 else False
#mel = True if mel > 0 else False
stretch = int(input('stretch=1 or not=0:'))
stretch = True if stretch > 0 else False

rootin = input('Enter input(root) folder:')
root = input('Enter output(target) folder:')
txtoutfd = root
infs, outfds = [], []
if mel: postfix = '_mel'
else: postfix = ''
if stretch: postfix += '_stretch'
if SPEC and FOLDER_MODE:
    #root = input('Enter output(target) folder:')
    #txtoutfd = root
    #infs, outfds = [], []
    for d in os.listdir(rootin):
        if d.startswith('_g'):
            infs.append(os.path.join(rootin,d))
            outfd = os.path.join(root, '{}_spec_{}x{}{}'.format(d, width, height, postfix))
            outfds.append(outfd)
            if not os.path.exists(outfd): os.makedirs(outfd)
elif SPEC:
    infs.append(rootin)
    outfd = os.path.join(root, '{}_spec_{}x{}{}'.format(rootin.split('\\')[-1],width, height, postfix))
    print(outfd)
    outfds.append(outfd)
    if not os.path.exists(outfd): os.makedirs(outfd)
######################################################
if(GEN):
    datainf = input('[DATASET] Enter input data file:')
    dataoutfd = input('Enter data output folder:')
    dataf = os.path.join(dataoutfd, 'label.txt')
    namef = os.path.join(dataoutfd, 'filename.txt')
    imf = os.path.join(dataoutfd, 'spec.npy')
    # 64 genres
    genres=['8Bit Chiptune', 'Acid', 'Acoustic', 'Ambient', 'Big Room', 'Blues', 'Boom Bap', 'Breakbeat', 'Chill Out', 
    'Cinematic', 'Classical', 'Comedy', 'Country', 'Crunk', 'Dance', 'Dancehall', 'Deep House', 'Dirty', 'Disco', 'Drum And Bass',
    'Dub', 'Dubstep', 'EDM', 'Electro', 'Electronic', 'Ethnic', 'Folk', 'Funk', 'Fusion', 'Garage', 'Glitch', 'Grime', 'Grunge', 
    'Hardcore', 'Hardstyle', 'Heavy Metal', 'Hip Hop', 'House', 'Indie', 'Industrial', 'Jazz', 'Jungle', 'Lo-Fi', 'Moombahton', 
    'Orchestral', 'Pop', 'Psychedelic', 'Punk', 'Rap', 'Rave', 'Reggae', 'Reggaeton', 'Religious', 'RnB', 'Rock', 'Samba', 'Ska',
    'Soul', 'Spoken Word', 'Techno', 'Trance', 'Trap', 'Trip Hop', 'Weird']
    #print(len(genres))
    # 40 cates
    categories=['Accordion', 'Arpeggio', 'Bagpipe', 'Banjo', 'Bass', 'Bass Guitar', 'Bass Synth', 'Bass Wobble', 
    'Beatbox', 'Bells', 'Brass', 'Choir', 'Clarinet', 'Didgeridoo', 'Drum', 'Flute', 'Fx', 'Groove', 'Guitar Acoustic', 
    'Guitar Electric', 'Harmonica', 'Harp', 'Harpsichord', 'Mandolin', 'Orchestral', 'Organ', 'Pad', 'Percussion', 
    'Piano', 'Rhodes Piano', 'Scratch', 'Sitar', 'Soundscapes', 'Strings', 'Synth', 'Tabla', 'Ukulele', 'Violin', 'Vocal', 'Woodwind']
    #print(len(categories))
    #genres=['Ambient', 'Blues', 'Chill Out','Cinematic','Classical',
    #'Dance','Drum And Bass','Dubstep','Electro','Ethnic','Funk', 
    #'Pop','Rap','Techno','Weird']
    #genres=['Blues','Dance','Drum And Bass','Electro','Funk']
###################################################### 
####################### SPECTROGRAM ####################### 
def load_wav(inf, wav_file, txtoutfd, txt, stretch=True, target_tempo=120):
    # Load WAV Files
    sig, sr = librosa.load(os.path.join(inf,wav_file), sr=44100)
    if not stretch: return sig, sr
    tempo, beats = librosa.beat.beat_track(y=sig, sr=sr)
    if tempo == 0:
        with open(os.path.join(txtoutfd, txt), 'a') as o:
            o.write(wav_file+'\n')
            print('tempo = 0!!!')
    return librosa.effects.time_stretch(sig, float(target_tempo)/tempo), sr
    #return pyrb.time_stretch(sig, sr, float(target_tempo)/tempo), sr

def graph_spectrogram(outf, wav_file, sig, sr, max_length=0, mel=True):
    if(max_length > 0):
        if sig.shape[0] > max_length: sig = sig[:max_length]
        else:# Padding
            sig = np.append(sig, np.zeros(max_length - sig.shape[0]))
    # Plot
    if(mel):
        pylab.figure(num=None)#, figsize=(width/20, height/20))
        pylab.axis('off') # no axis 
        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge 
        S = librosa.feature.melspectrogram(y=sig, sr=sr, n_mels=320)#, fmax=20000) 
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max)) 
        pylab.savefig(os.path.join(outfd,'spectrogram_%06d.png' % int(wav_file.split('.')[0])), bbox_inches=None, pad_inches=0) 
        pylab.close() 
    else:
        plt.figure(num=None)#, figsize=(width/20, height/20))
        plt.specgram(sig, Fs=sr)
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.axis('off')
        plt.margins(0.0)
        plt.tight_layout()
        plt.savefig(os.path.join(outfd,'spectrogram_%06d.png' % int(wav_file.split('.')[0])), bbox_inches='tight', pad_inches=0.0)
        plt.close()

def check_and_draw_spectrogram(inf, outfd, txtoutfd, stretch, mel=True, txt = 'zero_files.txt'):
    max_length = 10 # clip & padding to 10 sec.
    folder = os.listdir(inf)
    for wav_file in folder:
        if wav_file.endswith('.wav'):
            print(wav_file)
            y, sr = load_wav(inf, wav_file, txtoutfd, txt, stretch=stretch)
            graph_spectrogram(outfd, wav_file, y, sr, max_length*sr, mel)
            print(wav_file + ' done.')

def generate_dataset(IMG_ONLY, datainf, specf, folder_mode):
    print('------------------ GENERATE DATASET ------------------')
    if(IMG_ONLY):
        imarr = []
        if(folder_mode):
            for d in os.listdir(specf):
                if d.startswith('_g'):
                    for f in os.listdir(os.path.join(specf, d)):
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
                                                im = cv2.imread(os.path.join(os.path.join(specf,d), f))
                                                imarr.append(np.array(im))
                                                print(np.array(imarr).shape)
                                                break
                                        break
        else: #FILE_MODE
            for f in os.listdir(specf):
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
                                        im = cv2.imread(os.path.join(os.path.join(specf,d), f))
                                        imarr.append(np.array(im))
                                        print(np.array(imarr).shape)
                                        break
                                break
        np.save(imf, imarr)

    else:
        with open(dataf, 'w') as o:
            with open(namef, 'w') as oo:
                imarr = []
                if(folder_mode):
                    for d in os.listdir(specf):
                        if d.startswith('_g'):
                            for f in os.listdir(os.path.join(specf,d)):
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
                                                        im = cv2.imread(os.path.join(os.path.join(specf,d), f))
                                                        imarr.append(np.array(im))
                                                        print(np.array(imarr).shape)
                                                        print('done')
                                                        break
                                                break
                else: # FILE_MODE
                    for f in os.listdir(specf):
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
                                                im = cv2.imread(os.path.join(os.path.join(specf,d), f))
                                                imarr.append(np.array(im))
                                                print(np.array(imarr).shape)
                                                print('done')
                                                break
                                        break
                np.save(imf, imarr)


##################################################
if(SPEC): 
    for inf, outfd in zip(infs, outfds):
        check_and_draw_spectrogram(inf, outfd, txtoutfd, stretch=stretch, mel=mel)
if(GEN): generate_dataset(IMG_ONLY, datainf, rootin, FOLDER_MODE)
