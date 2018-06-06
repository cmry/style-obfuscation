
import os


def load_pairs(basedir, subset, tt=False):
    with open(os.path.join(basedir, 'bibstyles.{}'.format(subset))) as f:
        for line in f:
            src, trg, src_sent, trg_sent = line.strip().split(' || ')
            if len(src_sent.split()) > 0 and len(trg_sent.split()) > 0:
                src_sent, trg_sent = src_sent.split(), trg_sent.split()
                # prepend style token
                if tt:
                    src_sent = ['<{}>'.format(trg.lower())] + src_sent

                yield src_sent, src, trg_sent, trg


def load_sents(basedir, subset):
    with open(os.path.join(basedir, 'bibstyles.{}'.format(subset)), 'r') as f:
        for line in f:
            src, trg, src_sent, trg_sent = line.strip().split(' || ')
            if len(src_sent.split()) > 0:
                yield src, src_sent.split()
            if len(trg_sent.split()) > 0:
                yield trg, trg_sent.split()
