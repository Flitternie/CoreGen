import codecs
import argparse
import re
import wordninja

def subtokenize(pred, output):
    out_file = codecs.open(output, 'w', 'utf-8')
    with codecs.open(pred, 'r', 'utf-8') as f:
        for sent in f:
            sent = re.split('[ _]',sent.strip())
            out = []
            for word in sent:
                if word.isalpha() and not word.isupper() and not word.islower() and not word.istitle():
                    word = wordninja.split(word)
                    out += word
                else:
                    out.append(word)
            out_file.write(" ".join(out) + '\n')


def main():
    parser = argparse.ArgumentParser(description='subtokenizer.py')
    parser.add_argument('-input', required=True,
                        help='Path of output file to be processed')
    parser.add_argument('-output', required=True,
                        help='Path to save')
                 
    opt = parser.parse_args()
    for f in ["train.diff", "test.diff", "valid.diff"]:
        print("processing",f)
        subtokenize(opt.input+f, opt.output+"sub."+f)
        print(f,"finished")
    for f in ["train.msg", "test.msg", "valid.msg"]:
        print("processing",f)
        subtokenize(opt.input+f, opt.output+"sub."+f)
        print(f,"finished")

if __name__ == '__main__':
    main()
    