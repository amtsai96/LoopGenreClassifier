import numpy as np
import matplotlib.pyplot as plt
import json

G_FLAG = False
B_FLAG = True
K_FLAG = False#True

dataf = 'data_draw.txt'#'data_all.txt'#input('Enter data file name:')
out = 'out.txt'#input('Enter output data file name:')
out_b = 'out_BPM.txt'
out_k = 'out_KEY.txt'
genres = ['8Bit Chiptune', 'Acid', 'Acoustic', 'Ambient', 'Big Room', 'Blues', 'Boom Bap', 'Breakbeat', 'Chill Out', 'Cinematic', 'Classical', 'Comedy', 'Country', 'Crunk', 'Dance', 'Dancehall', 'Deep House', 'Dirty', 'Disco', 'Drum And Bass', 'Dub', 'Dubstep', 'EDM', 'Electro', 'Electronic', 'Ethnic', 'Folk', 'Funk', 'Fusion', 'Garage', 'Glitch', 'Grime', 'Grunge', 'Hardcore', 'Hardstyle', 'Heavy Metal', 'Hip Hop', 'House', 'Indie', 'Industrial', 'Jazz', 'Jungle', 'Lo-Fi', 'Moombahton', 'Orchestral', 'Pop', 'Psychedelic', 'Punk', 'Rap', 'Rave', 'Reggae', 'Reggaeton', 'Religious', 'RnB', 'Rock', 'Samba', 'Ska', 'Soul', 'Spoken Word', 'Techno', 'Trance', 'Trap', 'Trip Hop', 'Weird']

tags, bpm, key = dict(), dict(), dict()
with open(dataf, 'r') as i:
    for line in i.readlines():
        if line=="\n": continue
        data = json.loads(line)
        for t in data['tags']:  
            if t.strip('Loops').strip(' ') in genres:
                tt = t.strip('Loops').strip(' ')
                if tt in tags: tags[tt] += 1
                else: tags[tt] = 1
            if t.endswith('bpm'):
                tt = t.split(' ')[0]
                tt = (str(int(tt)//10*10))
                if tt in bpm: bpm[tt] += 1
                else: bpm[tt] = 1
            if t.startswith('Key'):
                tt = t.split(' ')[-1]
                if tt in key: key[tt] += 1
                else: key[tt] = 1
if(G_FLAG):
    print('--------------------GENRE--------------------')
    # by Genres
    ts = [t for t in genres and tags]#[t for t in tags]
    s = 5
    y_pos = np.arange(0, len(ts)*s, s)
    vs = [tags[t] for t in genres and tags]
    print(len(tags))
    print(tags)

    # save text to file
    with open(out, 'w') as o:
        for j in range(len(genres)):
            if genres[j] in tags:
                o.write(genres[j]+','+str(tags[genres[j]])+'\n')
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

    plt.savefig('looperman_dataset_genres.png', bbox_inches='tight', pad_inches=0.0)
    #plt.show()

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
    plt.ylim([0,max(vs)+1])
    plt.xlim([-2,y_pos[-1]+2.3])
    plt.tick_params(axis='x', which='major', labelsize=7)
    plt.title('Looperman Dataset')

    plt.savefig('looperman_dataset_bpm.png', bbox_inches='tight', pad_inches=0.0)
    #plt.show()

if(K_FLAG):
    print('--------------------KEY--------------------')
    # by Keys
    ts = [t for t in key]
    s = 5
    y_pos = np.arange(0, len(ts)*s, s)
    vs = [key[t] for t in key]
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

    plt.savefig('looperman_dataset_keys.png', bbox_inches='tight', pad_inches=0.0)
    #plt.show()

