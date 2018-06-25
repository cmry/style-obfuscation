
import os


def load_pairs(basedir, subset, tt=False, max_lines=-1):
    total = 0
    with open(os.path.join(basedir, 'bibstyles.{}'.format(subset))) as f:
        for line in f:
            if max_lines > 0 and total >= max_lines:
                break
            src, trg, src_sent, trg_sent = line.strip().split(' || ')
            if len(src_sent.split()) > 0 and len(trg_sent.split()) > 0:
                src_sent, trg_sent = src_sent.split(), trg_sent.split()
                # prepend style token
                if tt:
                    src_sent = ['<{}>'.format(trg.lower())] + src_sent

                yield src_sent, src, trg_sent, trg
                total += 1


def load_sents(basedir, subset, max_lines=-1):
    total = 0
    with open(os.path.join(basedir, 'bibstyles.{}'.format(subset)), 'r') as f:
        for line in f:
            if max_lines > 0 and total >= max_lines:
                break
            src, trg, src_sent, trg_sent = line.strip().split(' || ')
            if len(src_sent.split()) > 0:
                yield src, src_sent.split()
                total += 1
            if len(trg_sent.split()) > 0:
                yield trg, trg_sent.split()
                total += 1
