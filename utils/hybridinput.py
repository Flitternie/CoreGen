import codecs
import argparse

def hybridprocess(diff, retrieval, output):
    sep_token = "<SEP>"
    out_file = codecs.open(output, 'w', 'utf-8')
    with codecs.open(diff, 'r', 'utf-8') as f:
        with codecs.open(retrieval, 'r', 'utf-8') as r:
            while True:
                pline = f.readline()
                rline = r.readline()
                if pline == '' or rline == '':
                    break
                line = "%s %s %s" % (pline.strip(), sep_token, rline.strip())
                out_file.write( line + '\n')

def main():
    parser = argparse.ArgumentParser(description='hybridprocess.py')
    parser.add_argument('-diff', required=True,
                        help='Path of code diff file to be processed')
    parser.add_argument('-ret', required=True,
                        help='Path of retrieved best msg file to be processed')                    
    parser.add_argument('-output', required=True,
                        help='Path to save')                    
    opt = parser.parse_args()
    retrieval = opt.ret
    output = opt.output
    for file in ["train","valid","test"]:
        diff = opt.diff + file + ".diff"
        retrieval = opt.ret + file + ".msg"
        output = opt.output + file + ".hybrid"
        hybridprocess(diff, retrieval, output)

if __name__ == '__main__':
    main()