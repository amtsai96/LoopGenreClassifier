import numpy as np
import matplotlib.pyplot as plt
import json
import os

G_FLAG = True#False
C_FLAG = True
B_FLAG = True
K_FLAG = True#False

FILES_VER = False#True

dataf = input('Enter data file name:')
outfd = input('Enter output folder:')
if not os.path.exists(outfd): os.makedirs(outfd)
out_g = os.path.join(outfd, 'dataset_GENRE.txt')
out_c = os.path.join(outfd, 'dataset_CATEGORY.txt')
out_b = os.path.join(outfd, 'dataset_BPM.txt')
out_k = os.path.join(outfd, 'dataset_KEY.txt')
genres = ['8Bit Chiptune', 'Acid', 'Acoustic', 'Ambient', 'Big Room', 'Blues', 'Boom Bap', 'Breakbeat', 'Chill Out', 'Cinematic', 'Classical', 'Comedy', 'Country', 'Crunk', 'Dance', 'Dancehall', 'Deep House', 'Dirty', 'Disco', 'Drum And Bass', 'Dub', 'Dubstep', 'EDM', 'Electro', 'Electronic', 'Ethnic', 'Folk', 'Funk', 'Fusion', 'Garage', 'Glitch', 'Grime', 'Grunge', 'Hardcore', 'Hardstyle', 'Heavy Metal', 'Hip Hop', 'House', 'Indie', 'Industrial', 'Jazz', 'Jungle', 'Lo-Fi', 'Moombahton', 'Orchestral', 'Pop', 'Psychedelic', 'Punk', 'Rap', 'Rave', 'Reggae', 'Reggaeton', 'Religious', 'RnB', 'Rock', 'Samba', 'Ska', 'Soul', 'Spoken Word', 'Techno', 'Trance', 'Trap', 'Trip Hop', 'Weird']
categories = ['Accordion', 'Arpeggio', 'Bagpipe', 'Banjo', 'Bass', 'Bass Guitar', 'Bass Synth', 'Bass Wobble', 'Beatbox', 'Bells', 'Brass', 'Choir', 'Clarinet', 'Didgeridoo',
'Drum', 'Flute', 'Fx', 'Groove', 'Guitar Acoustic', 'Guitar Electric', 'Harmonica', 'Harp', 'Harpsichord', 'Mandolin', 'Orchestral', 'Organ', 'Pad', 'Percussion', 'Piano', 'Rhodes Piano', 'Scratch', 'Sitar', 'Soundscapes', 'Strings', 'Synth', 'Tabla', 'Ukulele', 'Violin', 'Vocal', 'Woodwind']
if FILES_VER:
    folder = input('enter the folder that contains files:')
    files = os.listdir(folder)

cats, gens, bpm, key = dict(), dict(), dict(), dict()
with open(dataf, 'r') as i:
    for line in i.readlines():
        if line=="\n": continue
        data = json.loads(line)
        for t in data['tags']: 
            if t.strip('Loops').strip(' ') in genres:
                if FILES_VER:
                    filename = data['index']+'.wav'
                    if filename not in files: continue
                tt = t.strip('Loops').strip(' ')
                if tt in gens: gens[tt] += 1
                else: gens[tt] = 1

            if t.strip('Loops').strip(' ') in categories:
                if FILES_VER:
                    filename = data['index']+'.wav'
                    if filename not in files: continue
                tt = t.strip('Loops').strip(' ')
                if tt in cats: cats[tt] += 1
                else: cats[tt] = 1  

            if t.endswith('bpm'):
                if FILES_VER:
                    filename = data['index']+'.wav'
                    if filename not in files: continue
                tt = t.split(' ')[0]
                tt = (str(int(tt)//10*10))
                if tt in bpm: bpm[tt] += 1
                else: bpm[tt] = 1

            if t.startswith('Key'):
                if FILES_VER:
                    filename = data['index']+'.wav'
                    if filename not in files: continue
                tt = t.split(' ')[-1]
                if tt in key: key[tt] += 1
                else: key[tt] = 1
if(G_FLAG):
    print('--------------------GENRE--------------------')
    # by Genres
    ts = [t for t in genres and gens]
    s = 5
    y_pos = np.arange(0, len(ts)*s, s)
    vs = [gens[t] for t in genres and gens]
    print(len(gens))
    print(gens)

    # save text to file
    with open(out_g, 'w') as o:
        for j in range(len(genres)):
            if genres[j] in gens:
                o.write(genres[j]+','+str(gens[genres[j]])+'\n')
            else:
                o.write(genres[j]+',0\n')

    # Plot - by Genres
    plt.figure(figsize=(23, 17), num='Looperman Dataset')
    plt.barh(y_pos, vs, align='center', alpha=0.5)
    for a, b in zip(y_pos, vs):
        plt.text(b+0.5, a-1.5, str(b))
    plt.yticks(y_pos, ts)
    plt.xlabel('Data #')
    plt.ylabel('Genres')
    plt.xlim([0,max(vs)+1])
    plt.ylim([-2,y_pos[-1]+2.3])
    plt.tick_params(axis='y', which='major', labelsize=7)
    plt.title('Looperman Dataset')

    plt.savefig(os.path.join(outfd, 'looperman_dataset_genres.png'), bbox_inches='tight', pad_inches=0.0)
    #plt.show()
    plt.close()

if(C_FLAG):
    print('--------------------Categories--------------------')
    # by categories
    ts = [t for t in categories and cats]
    s = 5
    y_pos = np.arange(0, len(ts)*s, s)
    vs = [cats[t] for t in categories and cats]
    print(len(cats))
    print(cats)

    # save text to file
    with open(out_c, 'w') as o:
        for j in range(len(categories)):
            if categories[j] in cats:
                o.write(categories[j]+','+str(cats[categories[j]])+'\n')
            else:
                o.write(categories[j]+',0\n')

    # Plot - by categories
    plt.figure(figsize=(23, 17), num='Looperman Dataset')
    plt.barh(y_pos, vs, align='center', alpha=0.5, color='plum')
    for a, b in zip(y_pos, vs):
        plt.text(b+0.5, a-1.5, str(b))
    plt.yticks(y_pos, ts)
    plt.xlabel('Data #')
    plt.ylabel('Categories')
    plt.xlim([0,max(vs)+1])
    plt.ylim([-2,y_pos[-1]+2.3])
    plt.tick_params(axis='y', which='major', labelsize=7)
    plt.title('Looperman Dataset')

    plt.savefig(os.path.join(outfd, 'looperman_dataset_categories.png'), bbox_inches='tight', pad_inches=0.0)
    #plt.show()
    plt.close()

if(K_FLAG):
    print('--------------------KEY--------------------')
    # by Keys
    k = sorted(key.items(), key=lambda a:a[0], reverse=True)
    ts = [t[0] for t in k]
    vs = [t[1] for t in k]
    #ts = [t for t in key]
    s = 5
    y_pos = np.arange(0, len(ts)*s, s)
    #vs = [key[t] for t in key]
    print(len(key))
    print(key)

    # save text to file
    with open(out_k, 'w') as o:
        for j in key: o.write(j+','+str(key[j])+'\n')

    # Plot
    plt.figure(figsize=(17, 10), num='Looperman Dataset')
    plt.barh(y_pos, vs, align='center', alpha=0.5, color='green')
    for a, b in zip(y_pos, vs):
        plt.text(b+0.5, a-1.5, str(b))
    plt.yticks(y_pos, ts)
    plt.xlabel('Data #')
    plt.ylabel('Key')
    plt.xlim([0,max(vs)+1])
    plt.ylim([-2,y_pos[-1]+2.3])
    plt.tick_params(axis='y', which='major', labelsize=7)
    plt.title('Looperman Dataset')

    plt.savefig(os.path.join(outfd, 'looperman_dataset_keys.png'), bbox_inches='tight', pad_inches=0.0)
    #plt.show()
    plt.close()

if(B_FLAG):
    print('--------------------BPM--------------------')
    # by BPM
    b = sorted(bpm.items(), key=lambda k:int(k[0]))
    ts = [t[0] for t in b]
    vs = [t[1] for t in b]
    s = 1
    y_pos = np.arange(0, len(ts)*s, s)
    
    print(len(bpm))
    print(bpm)

    # save text to file
    with open(out_b, 'w') as o:
        for j in bpm: o.write(j+','+str(bpm[j])+'\n')

    # Plot - by BPM
    plt.figure(figsize=(23, 17), num='Looperman Dataset')
    plt.bar(ts, vs, align='center', alpha=0.5, color='red')
    for a, b in zip(y_pos, vs):plt.text(a, b+2, str(b), ha='center')
    #plt.yticks(y_pos, ts)
    plt.ylabel('Data #')
    plt.xlabel('BPM')
    #plt.ylim([0,max(vs)+1])
    plt.xlim([-2,y_pos[-1]+2.3])
    plt.tick_params(axis='x', which='major', labelsize=7)
    plt.title('Looperman Dataset')

    plt.savefig(os.path.join(outfd, 'looperman_dataset_bpm.png'), bbox_inches='tight', pad_inches=0.0)
    #plt.show()
    plt.close()
