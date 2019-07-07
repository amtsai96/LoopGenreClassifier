import os
import numpy as np
import json
import cv2

inf = input('Enter input data file:')
specf = 'C:/Users/amandatsai/Desktop/VAEVAE/_spectrogram_padding/9'#'I:\_spectrogram'
#outf = '_label'
dataf = 'label_'
namef = 'filename_'
imf = 'spec_list'
# if not os.path.exists(outf):
#     os.makedirs(outf)
genres = ['8Bit Chiptune', 'Acid', 'Acoustic', 'Ambient', 'Big Room', 'Blues', 'Boom Bap', 'Breakbeat', 'Chill Out', 'Cinematic', 'Classical', 'Comedy', 'Country', 'Crunk', 'Dance', 'Dancehall', 'Deep House', 'Dirty', 'Disco', 'Drum And Bass', 'Dub', 'Dubstep', 'EDM', 'Electro', 'Electronic', 'Ethnic', 'Folk', 'Funk', 'Fusion', 'Garage', 'Glitch', 'Grime', 'Grunge', 'Hardcore', 'Hardstyle', 'Heavy Metal', 'Hip Hop', 'House', 'Indie', 'Industrial', 'Jazz', 'Jungle', 'Lo-Fi', 'Moombahton', 'Orchestral', 'Pop', 'Psychedelic', 'Punk', 'Rap', 'Rave', 'Reggae', 'Reggaeton', 'Religious', 'RnB', 'Rock', 'Samba', 'Ska', 'Soul', 'Spoken Word', 'Techno', 'Trance', 'Trap', 'Trip Hop', 'Weird']

with open(dataf, 'w') as o:
    with open(namef, 'w') as oo:
        #with open('imf.txt', 'w') as ooo:
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
                                    #tt = t.strip('Loops').strip(' ')
                                    #o.write(str(genres.index(tt))+'\n')
                                    #oo.write(data["index"]+'\n')
                                        
                                    im = cv2.imread(os.path.join(specf, f))
                                    #print(im.shape)
                                    imarr.append(np.array(im))
                                    print(np.array(imarr).shape)
                                    #np.save(imf, im)
                                    #print(im.shape) #(793, 1371, 3)
                                    #ooo.write(str(im)+'\n')
                                    print('done')
                                    break
                            break
        np.save(imf, np.array(imarr))



# for root, dirs, files in os.walk(inf):
#     for wav_file in files:
#         print(os.path.join(inf,wav_file))
#         if wav_file.endswith('.wav'):
#             graph_spectrogram(outf, inf, wav_file)
    
    