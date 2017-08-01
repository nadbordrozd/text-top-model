import string
from unidecode import unidecode

neg_path = 'data/polarity/rt-polarity.neg'
pos_path = 'data/polarity/rt-polarity.pos'
out_path = 'data/polarity.txt'


def is_punctuation(s):
    return s.translate(None, string.punctuation) == ""


def fix_line(line):
    line = unidecode(line.decode('latin-1'))
    return " ".join(w for w in line.split() if not is_punctuation(w))


def main():
    with open(neg_path, 'rb') as neg, open(pos_path, 'rb') as pos, open(out_path, 'wb') as out:
        for line in neg:
            out.write('NEG\t' + fix_line(line) + '\n')

        for line in pos:
            out.write('POS\t' + fix_line(line) + '\n')


if __name__ == '__main__':
    main()
