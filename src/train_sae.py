
import copy
import random

import torch
from torch import optim

from seqmod.modules.encoder_decoder import make_grl_rnn_encoder_decoder
from seqmod.modules.encoder_decoder import make_rnn_encoder_decoder
from seqmod.misc import PairedDataset, Trainer, Dict, EarlyStopping, Checkpoint
from seqmod.misc.loggers import StdLogger
from seqmod import utils as u

from data import load_sents
from utils import make_check_hook


def report(trainer, items):
    dataset, d = trainer.datasets['valid'], trainer.model.encoder.embeddings.d
    # sample batch
    inp, _ = dataset[random.randint(0, len(dataset) - 1)]

    # drop off condition
    if trainer.model.encoder.conditional:
        inp, _ = inp
    inp, lengths = inp

    # only take so many inputs
    inp, lengths = inp[:, :items], lengths[:items]

    # run
    scores, hyps, _ = trainer.model.translate_beam(inp, lengths)

    # stringify output
    trg = inp.transpose(0, 1).tolist()

    report = ''
    for num, (score, hyp, trg) in enumerate(zip(scores, hyps, trg)):
        report += u.format_hyp(score, hyp, num+1, d, trg=trg)

    return report


def conditional_report(trainer, items):
    dataset, d = trainer.datasets['valid'], trainer.model.encoder.embeddings.d
    _, conds_d = trainer.datasets['train'].d['trg']
    # sample batch
    inp, (_, *conds) = dataset[random.randint(0, len(dataset) - 1)]

    # drop off condition from encoder
    if trainer.model.encoder.conditional:
        inp, _ = inp
    inp, lengths = inp

    # only take so many inputs
    inp, lengths, conds = inp[:, :items], lengths[:items], [c[:items] for c in conds]

    # resample conds
    tconds = [torch.zeros_like(c).random_(len(conds_d)) for c in conds]

    # run
    scores, hyps, _ = trainer.model.translate_beam(inp, lengths, conds=tconds)

    # stringify output
    trg = inp.transpose(0, 1).tolist()
    conds = ['+'.join([conds_d.vocab[c[b]] for c in conds]) for b in range(len(trg))]
    tconds = ['+'.join([conds_d.vocab[c[b]] for c in tconds]) for b in range(len(trg))]
    conds = ['<{}>=>{}>'.format(c, tc) for c, tc in zip(conds, tconds)]

    report = ''
    for score, hyp, trg, cond in zip(scores, hyps, trg, conds):
        report += u.format_hyp(score, hyp, cond, d, trg=trg)

    return report


def make_report_hook(valid, items, conditional=False):

    def hook(trainer, epoch, batch, checkpoint):
        reporter = conditional_report if conditional else report
        trainer.log("info", "Generating some stuff...")
        trainer.log("info", reporter(trainer, items))

    return hook


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--basedir', default='/home/manjavacas/code/python/' +
                        'style-obfuscation/data/bibles/')
    parser.add_argument('--max_size', type=int, default=100000)
    parser.add_argument('--max_length', type=int, default=100)
    parser.add_argument('--max_lines', type=int, default=1000000)
    # model
    parser.add_argument('--grl', action='store_true')
    parser.add_argument('--emb_dim', type=int, default=300)
    parser.add_argument('--hid_dim', type=int, default=1000)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--cell', default='LSTM')
    parser.add_argument('--encoder_summary', default='inner-attention')
    parser.add_argument('--cond_emb', type=int, default=0)
    parser.add_argument('--train_init', action='store_true')
    parser.add_argument('--init_embeddings', action='store_true')
    parser.add_argument('--embeddings_path',
                        default='/home/corpora/word_embeddings/' +
                        'glove.twitter.27B.100d.txt')
    # training
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--word_dropout', type=float, default=0.0)
    parser.add_argument('--optimizer', default='Adam')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_schedule_epochs', type=int, default=1)
    parser.add_argument('--lr_schedule_factor', type=float, default=0.75)
    parser.add_argument('--max_norm', type=float, default=5.)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--checkpoint', type=int, default=100)
    parser.add_argument('--hook', type=int, default=1)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    print("Loading data...")
    train_conds, train = zip(*load_sents(args.basedir, 'train', max_lines=args.max_lines))
    train_conds, train = list(train_conds), list(train)
    d = Dict(
        eos_token=u.EOS, bos_token=u.BOS, unk_token=u.UNK,
        pad_token=u.PAD, max_size=args.max_size, force_unk=True
    ).fit(train)
    d2 = copy.deepcopy(d)
    d2.align_right = args.reverse
    conds_d = Dict(sequential=False).fit(train_conds)

    conditional = args.cond_emb > 0

    # AE+GRL+C
    if args.grl and conditional:
        src, trg = (train, train_conds), (train, train_conds)
        dicts = {'src': (d, conds_d), 'trg': (d2, conds_d)}
    # AE+GRL
    elif args.grl:
        src, trg = (train, train_conds), train
        dicts = {'src': (d, conds_d), 'trg': d2}
    # AE+C
    elif conditional:
        src, trg = train, (train, train_conds)
        dicts = {'src': d, 'trg': (d2, conds_d)}
    # AE
    else:
        src, trg = train, train
        dicts = {'src': d, 'trg': d2}

    train = PairedDataset(
        src, trg, dicts, batch_size=args.batch_size, device=args.device
    ).shuffle_()
    train, valid = train.splits(test=None, dev=0.2)

    key = lambda it: len(it[0]) if conditional else len(it)
    train.sort_(sort_by='trg', key=key)

    cond_dims, cond_vocabs = None, None
    if conditional or args.grl:
        cond_dims, cond_vocabs = (args.cond_emb,), (len(conds_d),)

    print("Building model...")
    if args.grl:
        m = make_grl_rnn_encoder_decoder(
            args.num_layers, args.emb_dim, args.hid_dim, d, cell=args.cell,
            encoder_summary=args.encoder_summary, dropout=args.dropout,
            tie_weights=True, word_dropout=args.word_dropout,
            cond_dims=cond_dims, cond_vocabs=cond_vocabs,
            conditional_decoder=conditional, add_init_jitter=True,
            train_init=args.train_init, reverse=args.reverse)

        # don't rely on GRL for early stopping (set its weight to 0)
        losses = ('ppl', {'loss': 'grl', 'format': 'ppl'})
        weights = {'ppl': 1, 'grl': 0}

    else:
        m = make_rnn_encoder_decoder(
            args.num_layers, args.emb_dim, args.hid_dim, d, cell=args.cell,
            encoder_summary=args.encoder_summary, dropout=args.dropout,
            tie_weights=True, word_dropout=args.word_dropout,
            reuse_hidden=False, input_feed=False, att_type=None,
            cond_dims=cond_dims, cond_vocabs=cond_vocabs,
            add_init_jitter=True, train_init=args.train_init,
            reverse=args.reverse)

        losses, weights = ('ppl',), None

    print(m)
    print('* number of params: ', sum(p.nelement() for p in m.parameters()))

    u.initialize_model(m)
    if args.init_embeddings:
        m.encoder.embeddings.init_embeddings_from_file(
            args.embeddings_path, verbose=True)

    m.to(args.device)

    optimizer = getattr(optim, args.optimizer)(m.parameters(), lr=args.lr)
    # Decrease lr by a factor after each epoch
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, args.lr_schedule_epochs, args.lr_schedule_factor)

    model_name = 'AE.GRL{}.C{}'.format(str(args.grl), str(conditional))
    trainer = Trainer(
        m, {'train': train, 'valid': valid}, optimizer, losses=losses,
        max_norm=args.max_norm, scheduler=scheduler, weights=weights)

    trainer.add_loggers(StdLogger())

    # Hooks
    # - early stopping & checkpoint
    early_stopping = None
    if args.patience:
        early_stopping = EarlyStopping(args.patience)
    checkpoint = None
    if not args.test:
        checkpoint = Checkpoint(model_name, keep=3).setup(args, d=conds_d)
    trainer.add_hook(make_check_hook(early_stopping, checkpoint),
                     hooks_per_epoch=args.hook)

    # - print hook
    trainer.add_hook(make_report_hook(valid, 5, conditional), hooks_per_epoch=args.hook)

    (best_model, valid_loss), _ = trainer.train(
        args.epochs, args.checkpoint, shuffle=True)
    if not args.test:
        checkpoint.save(best_model, valid_loss)
