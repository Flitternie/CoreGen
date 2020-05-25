import codecs
import argparse

def del_repeat(input_file, output_file):
    output_file = codecs.open(output_file, 'w', 'utf-8')
    with codecs.open(input_file, 'r', 'utf-8') as f:
        while True:
            pline = f.readline()
            if pline == '':
                break
            pphrase = pline.strip().split(' ')[:-1]
            ophrase = []
            for p in pphrase:
                if p in ophrase:
                    continue
                else:
                    ophrase.append(p)
            output_file.write(" ".join(ophrase) + '\n')

def del_last(input_file, output_file):
    output_file = codecs.open(output, 'w', 'utf-8')
    with codecs.open(input_file, 'r', 'utf-8') as f:
        while True:
            pline = f.readline()
            if pline == '':
                break
            pphrase = pline.strip().split(' ')[:-1]
            output_file.write(" ".join(pphrase) + '\n')


def to_lower(input_file, output_file):
    output_file = codecs.open(output_file, 'w', 'utf-8')
    with codecs.open(input_file, 'r', 'utf-8') as f:
        while True:
            pline = f.readline()
            if pline=="":
                break
            pline = pline.lower()
            output_file.write(pline)

def main():
    parser = argparse.ArgumentParser(description='postprocess.py')
    parser.add_argument('-input', required=True,
                        help='Path of output file to be processed')
    parser.add_argument('-output', required=True,
                        help='Path to save')
    parser.add_argument('-func', required=True,
                        help='Functions to choose: to_lower , del_repeat')
                        
    opt = parser.parse_args()
    input_file = opt.input
    output_file = opt.output 
    if opt.func == "to_lower":
        to_lower(input_file, output_file)
    elif opt.func == "del_repeat":
        del_repeat(input_file, output_file)


if __name__ == '__main__':
    main()
    