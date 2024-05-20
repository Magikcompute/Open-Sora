
# eval("sudo fuser -v /dev/nvidia* > tmp.txt")
with open('tmp.txt', 'r') as f:
    lines = f.readlines()

for ele in lines[0][:-1].split(' ')[1:]:
    print(f'kill {ele}')

