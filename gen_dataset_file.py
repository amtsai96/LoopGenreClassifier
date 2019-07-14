import os
import numpy as np
import json
import cv2

IMG_ONLY = False

inf = input('Enter input data file:')
#mode = input('enter train/test:')

specf = input('Enter spec folder:')#'I:/_loops/subset/spec_100x100/'#+mode
dataf = 'label.txt'
namef = 'filename.txt'
imf = 'spec.npy'

#genres = ['8Bit Chiptune', 'Acid', 'Acoustic', 'Ambient', 'Big Room', 'Blues', 'Boom Bap', 'Breakbeat', 'Chill Out', 'Cinematic', 'Classical', 'Comedy', 'Country', 'Crunk', 'Dance', 'Dancehall', 'Deep House', 'Dirty', 'Disco', 'Drum And Bass', 'Dub', 'Dubstep', 'EDM', 'Electro', 'Electronic', 'Ethnic', 'Folk', 'Funk', 'Fusion', 'Garage', 'Glitch', 'Grime', 'Grunge', 'Hardcore', 'Hardstyle', 'Heavy Metal', 'Hip Hop', 'House', 'Indie', 'Industrial', 'Jazz', 'Jungle', 'Lo-Fi', 'Moombahton', 'Orchestral', 'Pop', 'Psychedelic', 'Punk', 'Rap', 'Rave', 'Reggae', 'Reggaeton', 'Religious', 'RnB', 'Rock', 'Samba', 'Ska', 'Soul', 'Spoken Word', 'Techno', 'Trance', 'Trap', 'Trip Hop', 'Weird']
# genres = ['8Bit Chiptune','Classical','Dance','Dancehall','Deep House',
# 'Drum And Bass','Ethnic','Funk','Fusion','Glitch',
# 'Hardcore','Hardstyle','Heavy Metal', 'Reggae', 
# 'Techno', 'Trip Hop','Weird']
genres=['Ambient', 'Blues', 'Chill Out','Cinematic','Classical',
'Dance','Drum And Bass','Dubstep','Electro','Ethnic','Funk', 
'Pop','Rap','Techno','Weird']

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
                    with open(inf, 'r') as i:
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
