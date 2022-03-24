import os
import shutil

machine_path = r"C:\Users\msi-pc\Desktop\Datas\sorted\Machine"
Men_path = r"C:\Users\msi-pc\Desktop\Datas\sorted\People"
total_path = r"C:\Users\msi-pc\Desktop\Datas\result"
listdir = os.listdir(total_path)
listdir = listdir[1:]
#print(listdir)
for i in listdir:
    total_path2 = total_path + '\\' + i
    #print(total_path2)
    listdir2 = os.listdir(total_path2)
    #print(listdir2[0])
    #print(listdir2[1])
    if len(listdir2) > 1:
        shutil.copyfile(total_path2 + '\\' + listdir2[0], Men_path + '\\' + listdir2[0])
        shutil.copyfile(total_path2 + '\\' + listdir2[1], machine_path + '\\' + listdir2[0] + '_TR')