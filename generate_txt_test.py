from os import listdir
from os.path import isfile, isdir, join
import random
# 指定要列出所有檔案的目錄
mypath = "Annotations/"

# 取得所有檔案與子目錄名稱
files = listdir(mypath)
print(len(files))
# 以迴圈處理
f1 = open('test.txt','w')
f2 = open('trainval.txt','w')
f3 = open('train.txt','w')
f4 = open('val.txt','w')
test_num = 0
trainval_num = 0
train_num = 0
val_num = 0
for f in files:
  # 產生檔案的絕對路徑
  #fullpath = join(mypath, f)
  # 判斷 fullpath 是檔案還是目錄
  a = random.random()
  if a < 0.2 :
    f1.write(f[0:len(f)-4]+'\n')
    test_num += 1
  else :
    f2.write(f[0:len(f)-4]+'\n')
    trainval_num += 1
    b = random.random()
    if b<0.5 :
      f3.write(f[0:len(f)-4]+'\n')
      train_num += 1
    else :
      f4.write(f[0:len(f)-4]+'\n')
      val_num += 1
print('train=',train_num)
print('val=',val_num)
print('trainval=',trainval_num)
print('test=',test_num)
