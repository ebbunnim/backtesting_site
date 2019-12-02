f = open('_setting.txt', 'r')
lines = f.readlines()

setting = {}
for line in lines:
    item = line.split(" = ")
    setting[item[0]] = (item[1]).strip('\n')
    
if __name__ == "__main__":
    print("\n\n--------------------------------------- setting.py test ---------------------------------------")
    for key in setting.keys():
        print(key, setting[key], sep=' : ')
    print("\n--------------------------------------- No problem!!!!! ---------------------------------------", end='\n\n')
    