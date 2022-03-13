import re
import sys


def main(args):
    txt_file = args[1]
    clean_txt_file = args[2]
    clean_lines = []
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        lines = lines[0].split('.')
        #print(lines)
        for line in lines:
            valids = re.sub(r"[^A-Za-z]+", ' ', line)
            clean_lines.append(valids)
    #print(clean_lines)
    with open(clean_txt_file, 'w') as f:
        for line in clean_lines:
            f.write(line + '\n')

if __name__ == '__main__':
    main(sys.argv)