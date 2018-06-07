
import random

from torch import optim

from seqmod.modules.encoder_decoder import make_grl_rnn_encoder_decoder
from seqmod.modules.encoder_decoder import make_rnn_encoder_decoder
from seqmod.misc import PairedDataset, Trainer, Dict, EarlyStopping, Checkpoint
from seqmod.misc.loggers import StdLogger
from seqmod import utils as u

from data import load_pairs
from utils import make_early_stopping_hook


def report(trainer, items):
    dataset, d = trainer.datasets['valid'], trainer.model.decoder.embeddings.d
    # sample batch
    batch = dataset[random.randint(0, len(dataset) - 1)]

    src, (trg, _) = batch
    if trainer.model.encoder.conditional:
        src, _ = src
    src, src_lengths = src

    # only take so many inputs
    src, src_lengths, trg = src[:, :items], src_lengths[:items], trg[:, :items]
    scores, hyps, _ = trainer.model.translate_beam(src, src_lengths)

    trg, src = trg.transpose(0, 1).tolist(), src.transpose(0, 1).tolist()

    report = ''
    for num, (score, hyp, trg) in enumerate(zip(scores, hyps, trg)):
        report += u.format_hyp(score, hyp, num+1, d, trg=trg)

    return report


def make_report_hook(valid, items, conditional=False):

    def hook(trainer, epoch, batch, checkpoint):
        trainer.log("info", "Generating some stuff...")
        trainer.log("info", report(trainer, items))

    return hook


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--basedir', default='/home/manjavacas/code/python/' +
                        'style-obfuscation/data/bibles/')
    parser.add_argument('--max_size', type=int, default=100000)
    # model
    parser.add_argument('--grl', action='store_true')
    parser.add_argument('--tt', action='store_true')
    parser.add_argument('--layers', default=1, type=int)
    parser.add_argument('--cell', default='LSTM', type=str)
    parser.add_argument('--emb_dim', default=100, type=int)
    parser.add_argument('--hid_dim', default=200, type=int)
    parser.add_argument('--att_type', default='general', type=str)
    parser.add_argument('--deepout_layers', default=0, type=int)
    parser.add_argument('--tie_weights', action='store_true')
    parser.add_argument('--train_init', action='store_true')
    parser.add_argument('--encoder_summary', default='inner-attention')
    # training
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--batch_size', default=200, type=int)
    parser.add_argument('--optim', default='Adam', type=str)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--lr_schedule_epochs', type=int, default=1)
    parser.add_argument('--lr_schedule_factor', type=float, default=1)
    parser.add_argument('--max_norm', default=5., type=float)
    parser.add_argument('--init_embeddings', action='store_true')
    parser.add_argument('--embeddings_path',
                        default='/home/corpora/word_embeddings/' +
                        'glove.twitter.27B.100d.txt')
    parser.add_argument('--dropout', default=0.25, type=float)
    parser.add_argument('--word_dropout', default=0.0, type=float)
    parser.add_argument('--patience', default=0, type=int)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--checkpoint', default=100, type=int)
    parser.add_argument('--hook', default=1, type=int)
    parser.add_argument('--test', action='store_true', help="Don't save")
    parser.add_argument('--reverse', action='store_true')
    args = parser.parse_args()

    print("Loading data...")
    max_size, tokenize = args.max_size, True

    src, src_conds, trg, trg_conds = \
        zip(*load_pairs(args.basedir, 'train', tt=args.tt))
    src, src_conds = list(src), list(src_conds)
    trg, trg_conds = list(trg), list(trg_conds)

    conds_d = Dict(sequential=False).fit(src_conds, trg_conds)
    d = Dict(
        eos_token=u.EOS, bos_token=u.BOS, unk_token=u.UNK,
        pad_token=u.PAD, max_size=max_size, force_unk=True,
        align_right=args.reverse
    ).fit(src, trg)

    # S2S+GRL
    if args.grl:
        if args.tt:
            raise ValueError("GRL+TT doesn't quite make sense")
        src, trg = (src, src_conds), trg
        dicts = {'src': (d, conds_d), 'trg': d}
    # S2S or TT
    else:
        dicts = {'src': d, 'trg': d}

    train, valid = PairedDataset(
        src, trg, dicts, batch_size=args.batch_size, device=args.device
    ).splits(test=None, dev=0.2)

    train.sort_(sort_by='trg')  # minimize padding

    print("Building model...")
    if args.grl:
        m = make_grl_rnn_encoder_decoder(
            args.layers, args.emb_dim, args.hid_dim, d, cell=args.cell,
            encoder_summary=args.encoder_summary, dropout=args.dropout,
            tie_weights=True, word_dropout=args.word_dropout,
            cond_dims=(args.emb_dim,), cond_vocabs=(len(conds_d),),
            conditional_decoder=False,
            train_init=args.train_init, reverse=args.reverse)

        # don't rely on GRL for early stopping (set its weight to 0)
        losses, weights = ('ppl', 'grl'), {'ppl': 1, 'grl': 0}

    else:
        m = make_rnn_encoder_decoder(
            args.layers, args.emb_dim, args.hid_dim, d, cell=args.cell,
            encoder_summary=args.encoder_summary, dropout=args.dropout,
            tie_weights=True, word_dropout=args.word_dropout,
            att_type=args.att_type,
            # only reuse if using full attention model
            reuse_hidden=args.encoder_summary == 'full',
            # only do input feeding for attentional models
            input_feed=args.att_type.lower() != 'none',
            train_init=args.train_init, reverse=args.reverse)

        losses, weights = ('ppl',), None

    print(m)
    print('* number of params: ', sum(p.nelement() for p in m.parameters()))

    u.initialize_model(m)
    if args.init_embeddings:
        m.encoder.embeddings.init_embeddings_from_file(
            args.embeddings_path, verbose=True)

    m.to(args.device)

    optimizer = getattr(optim, args.optim)(m.parameters(), lr=args.lr)
    # Decrease lr by a factor after each epoch
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, args.lr_schedule_epochs, args.lr_schedule_factor)

    model_name = 'S2S.GRL{}.TT{}'.format(str(args.grl), str(args.tt))
    trainer = Trainer(
        m, {'train': train, 'valid': valid}, optimizer, losses=losses,
        max_norm=args.max_norm, scheduler=scheduler, weights=weights,
        checkpoint=Checkpoint(model_name, keep=3).setup(args, d=conds_d))

    trainer.add_loggers(StdLogger())

    # Hooks
    # - early stopping
    early_stopping = None
    if args.patience:
        early_stopping = EarlyStopping(args.patience)
        trainer.add_hook(
            make_early_stopping_hook(early_stopping), hooks_per_epoch=2)

    # - print hook
    trainer.add_hook(
        make_report_hook(valid, args.batch_size), hooks_per_epoch=args.hook)

    (best_model, valid_loss), test_loss = trainer.train(
        args.epochs, args.checkpoint, shuffle=True)
