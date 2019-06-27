import os
import json
index = int(input('Enter original_start index:'))
end = int(input('Enter original_end index:'))
st = int(input('Enter new_start index:'))
new = 'data_new_'+str(index)+'-'+str(end)+'_'+str(st)+'-'+str(st+end-index)+'.txt'
with open('data_test.txt','r+') as f:
    with open(new,'w') as o:
        i=0
        for line in f.readlines():
            if line=="\n": continue
            data = json.loads(line)
            if index+i > end or data["index"] != str("%06d" % int(index+i)):
                o.write(line)
                continue
            else:
                print(data['index'])
                data["index"] = str("%06d" % int(st+i))
                print('change to',data['index'])
                json.dump(data, o)
                o.write('\n')
            i+=1

