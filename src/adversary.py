import csv
import os
import pickle
import sys
from subprocess import run


class BibleAdversary(object):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        try:
            self.mapping = pickle.load(open(self.args.labels, 'rb'))
        except EOFError:
            self.mapping = {}

    def _load_data(self, fpath):
        map_i = 0
        for line in open(fpath):
            row = line[:-1].split(' || ')
            for ix_label, ix_text in [(0, 2), (1, 3)]:
                if not self.mapping.get(row[ix_label]):
                    self.mapping[row[ix_label]] = map_i
                    map_i += 1
                yield self.mapping[row[ix_label]], row[ix_text]
        pickle.dump(self.mapping, open(self.args.labels, 'wb'))

    def ft_format(self, path='../data/bibles/bibstyles.{}', base='../',
                  split='train', invert=False):
        fpath = os.path.join(base, path.format(split))
        with open(fpath.replace('.t', '_ft.t'), 'w') as fo:
            if invert:
                label, text = None, None
            for i, row in enumerate(self._load_data(fpath)):
                if not invert:
                    label, text = row
                    if split == 'test' and i % 2:
                        continue
                    fo.write('__label__' + str(label) + ' ' + text + "\n")
                else:
                    if split == 'test' and i % 2:
                        text = row[1]
                        fo.write('__label__' + str(label) + ' ' + text + "\n")
                    if split == 'test' and not i % 2:
                        label = row[0]
                    if split == 'train':
                        label, text = row
                        fo.write('__label__' + str(label) + ' ' + text + "\n")

    def run(self):
        self.ft_format(split='train')
        self.ft_format(split='test')
        bash_stack = f'./fastText/fasttext supervised -input \
                      ../data/bibles/bibstyles_ft.train -output \
                      ./results/bibstyles \
                      -dim {self.dim} -lr {self.lr} -wordNgrams \
                      {self.n_grams} -minCount {self.min_count} -minn \
                      {self.minn} -maxn {self.maxn} -bucket {self.bucket} \
                      -epoch {self.epoch} -thread {self.thread}'.split()
        run(bash_stack)

        bash_stack = f'./fastText/fasttext test {self.args.model} \
                     ../data/bibles/bibstyles_ft.test'.split()

        run(bash_stack)

    def eval(self, res_file):
        ofn = res_file.replace('.translations.csv', '.ft.txt')
        of = open(ofn, 'w')
        for line in open(res_file).read().split('\n'):
            row = line.split('\t')
            if len(row) > 1:
                of.write('__label__' + str(self.mapping[row[0]]) + ' ' +
                         row[4] + '\n')
        bash_stack = f'./fastText/fasttext test {self.args.model} \
                     {ofn}'.split()
        run(bash_stack)


if __name__ == '__main__':
    import argparse  # FIXME
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('--model', default='../results/bibstyles.bin')
    parser.add_argument('--labels', default='./label_mapping.pickle')
    parser.add_argument('--max_length', type=int, default=100)
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()

    ft = BibleAdversary(dim=100, lr=0.01, n_grams=2, min_count=2,
                            minn=3, maxn=3, bucket=1000000, epoch=20,
                            thread=20, args=args)
    if args.train:
        ft.run()
    ft.eval(args.input)
