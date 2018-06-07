
import os
import torch
import tqdm
from seqmod.utils import load_model
from data import load_pairs


def add_noise(mean=0, stddev=0.05):
    """
    Add gaussian noise to the context before starting decode
    """
    def callback(m, state):
        noise = torch.zeros_like(state.context).normal_(mean, stddev)
        state.context = state.context + noise
        return

    return callback


def chunks(it, size):
    """
    Chunk a generator into a given size (last chunk might be smaller)
    """
    buf = []
    for s in it:
        buf.append(s)
        if len(buf) == size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path', help="Path to model")
    parser.add_argument('--basedir', default='/home/manjavacas/code/python/' +
                        'style-obfuscation/data/bibles/')
    parser.add_argument('--tt', action='store_true', help='Is model TT?')
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--beam_width', default=5, type=int)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--stddev', default=0.0, type=float)
    parser.add_argument('--method', default='translate_beam',
                        help='translate,translate_beam')
    parser.add_argument('--store_path', default='/tmp/', help='If passed, model '
                        'output will be saved to the given directory along '
                        'with the reference sentences')
    args = parser.parse_args()

    model_path = os.path.dirname(args.path)
    model_name = os.path.basename(model_path)
    print("Translating {} ...".format(model_path))

    print("Loading model...")
    model = load_model(args.path)
    d = model.encoder.embeddings.d
    reserved = set([d.index(c) for c in d.reserved])
    conds_d = load_model(os.path.join(model_path, 'dict.pt'))
    model.eval()

    src, src_conds, trg, trg_conds = \
        zip(*load_pairs(args.basedir, 'test', tt=args.tt))
    src, src_conds = list(src), list(src_conds)
    trg, trg_conds = list(trg), list(trg_conds)

    batches = chunks(zip(src, src_conds, trg, trg_conds), args.batch_size)

    outputfile = 'm-{}.w-{}.noise-{:.3f}.obfuscation'.format(
        args.method, args.beam_width, args.stddev)
    outputfile = os.path.join(model_path, outputfile)
    if os.path.isfile(outputfile):
        raise ValueError("Output file [{}] already exists".format(outputfile))

    with open(outputfile, 'w+') as f:
        for chunk in tqdm.tqdm(batches, total=len(src)//args.batch_size):
            src, src_cond, trg, trg_cond = zip(*chunk)
            inp, lengths = d.pack(list(d.transform(src)), return_lengths=True)

            conds = None
            if model.decoder.conditional:
                conds = [conds_d.pack(trg_conds)]

            on_init_state = None
            if args.stddev > 0:
                on_init_state = add_noise(stddev=args.stddev)

            with torch.no_grad():
                scores, hyps, _ = getattr(model, args.method)(
                    inp, lengths, conds=conds, on_init_state=on_init_state)

            for b in range(len(hyps)):
                hyp = ' '.join(d.vocab[c] for c in hyps[b] if c not in reserved)
                score = scores[b]

                f.write('{}\t{}\t{}\t{}\t{}\t{:g}\n'.format(
                    src_cond[b],
                    trg_cond[b],
                    ' '.join(src[b]),
                    ' '.join(trg[b]),
                    hyp,
                    score))
