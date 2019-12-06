'''
_setting.py
    1) 사용자마다 경로 주소가 모두 다르기 때문에 이에 대해 보다 편하게 반영해주는 모듈의 필요성
    2) _setting.txt에 입력된 정보를 바탕으로 경로가 지정되며 입력받지 않은 경우 기본형태로 경로를 설정한다
        * Default
        a. dir_base : os.getcwd()
        b. others   : dir_base 내의 폴더
    3) Return = (Dict)Setting
    4) Usage  = from _setting import setting -> setting[...]
'''
import os

f = open('_setting.txt', 'r')
lines = f.readlines()

setting = {}
for line in lines:
    if line[0] == '#': continue
    dir = line.split(" = "); dir_name = dir[0]; dir = (dir[1]).strip('\n') 
    print(dir)
    if dir == '': # 파일 경로가 명시 되지 않은 경우
        dir = dir_name.split('dir_')[1]
        if dir == 'package': dir = os.getcwd()+'/'
        else: dir = setting['dir_package']+dir+'/'
    setting[dir_name] = dir

f.close()

if __name__ == "__main__":
    print("\n--------------------------------------- setting.py test ---------------------------------------")
    for key in setting.keys():
        print(key, setting[key], sep=' : ')
    print("------------------------------------------ Success!! ------------------------------------------", end='\n\n')


