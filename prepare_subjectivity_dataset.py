import string

subj_path = "data/subjectivity_dataset/quote.tok.gt9.5000"
obj_path = "data/subjectivity_dataset/plot.tok.gt9.5000"
out_path = "data/subjectivity_10k.txt"


def is_punctuation(s):
    return s.translate(None, string.punctuation) == ""


def fix_line(line):
    return " ".join(w for w in line.split() if not is_punctuation(w))


def get_fixed_lines(path):
    with open(path, 'rb') as f:
        return [fix_line(line) for line in f]


def main():
    with open(out_path, 'wb') as out:
        for line in get_fixed_lines(subj_path):
            out.write('SUB\t' + line + '\n')

        for line in get_fixed_lines(obj_path):
            out.write('OBJ\t' + line + '\n')


if __name__ == '__main__':
    main()
