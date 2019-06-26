import os
import json
# import datetime
dirs = os.listdir(os.getcwd())
index = int(input('Enter original_start index:'))
end = int(input('Enter original_end index:'))
st = int(input('Enter new_start index:'))
for i in range(index, end+1):
    for files in dirs:
        if os.path.isfile(files) and files.startswith("%06d" % int(i)):
            os.rename(files, "%06d" % int(i+st-index)+".wav")
            break
        
    # if os.path.isfile(files) and files.startswith('looperman') and files.endswith('.wav') and str('-'+num+'-') in str(files):
    #     print('*'+files)

