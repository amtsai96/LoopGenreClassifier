import os
import json
# import datetime
dirs = os.listdir(os.getcwd())
index = int(input('Enter original_start index:'))
end = int(input('Enter original_end index:'))
st = int(input('Enter new_start index:'))
for i in range(end, index-1, -1):#index, end+1):
    #flag = False
    for files in dirs:
        if os.path.isfile(files) and files.startswith("%06d" % int(i)):
            os.rename(files, "%06d" % int(i+st-index)+".wav")
            print("%06d" % int(i)+"=>%06d" % int(i+st-index)+".wav")
            # flag = True
            break

    # if flag ==False:
    #     print("%06d" % int(i))

