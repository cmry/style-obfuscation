
import copy
import random

from torch import optim

from seqmod.modules.encoder_decoder import make_grl_rnn_encoder_decoder
from seqmod.modules.encoder_decoder import make_rnn_encoder_decoder
from seqmod.misc import PairedDataset, Trainer, Dict, EarlyStopping, Checkpoint
from seqmod.misc.loggers import StdLogger
from seqmod import utils as u

from data import load_pairs
from utils import make_check_hook


def report(trainer, items):
    dataset, d = trainer.datasets['valid'], trainer.model.encoder.embeddings.d
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
    parser.add_argument('--basedir', default='../data/bibles/')
    parser.add_argument('--max_size', type=int, default=100000)
    parser.add_argument('--max_lines', type=int, default=1000000)
    # model
    parser.add_argument('--grl', action='store_true')
    parser.add_argument('--tt', action='store_true')
    parser.add_argument('--layers', default=1, type=int)
    parser.add_argument('--cell', default='LSTM', type=str)
    parser.add_argument('--emb_dim', default=300, type=int)
    parser.add_argument('--hid_dim', default=1000, type=int)
    parser.add_argument('--att_type', default=None)
    parser.add_argument('--deepout_layers', default=0, type=int)
    parser.add_argument('--tie_weights', action='store_true')
    parser.add_argument('--train_init', action='store_true')
    parser.add_argument('--encoder_summary', default='inner-attention')
    # training
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--optim', default='Adam', type=str)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--lr_schedule_epochs', type=int, default=1)
    parser.add_argument('--lr_schedule_factor', type=float, default=1)
    parser.add_argument('--max_norm', default=5., type=float)
    parser.add_argument('--init_embeddings', action='store_true')
    parser.add_argument('--embeddings_path',
                        default='')
    parser.add_argument('--dropout', default=0.25, type=float)
    parser.add_argument('--word_dropout', default=0.0, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--checkpoint', default=1000, type=int)
    parser.add_argument('--hook', default=1, type=int)
    parser.add_argument('--test', action='store_true', help="Don't save")
    args = parser.parse_args()

    print("Loading data...")
    src, src_conds, trg, trg_conds = \
        zip(*load_pairs(args.basedir, 'train', tt=args.tt, max_lines=args.max_lines))
    src, src_conds = list(src), list(src_conds)
    trg, trg_conds = list(trg), list(trg_conds)

    d = Dict(
        eos_token=u.EOS, bos_token=u.BOS, unk_token=u.UNK,
        pad_token=u.PAD, max_size=args.max_size, force_unk=True,
    ).fit(src, trg)
    d2 = copy.deepcopy(d)
    d2.align_right = args.reverse
    conds_d = Dict(sequential=False).fit(src_conds, trg_conds)

    # S2S+GRL
    if args.grl:
        if args.tt:
            raise ValueError("GRL+TT doesn't quite make sense")
        src, trg = (src, src_conds), trg
        dicts = {'src': (d, conds_d), 'trg': d2}
    # S2S or TT
    else:
        dicts = {'src': d, 'trg': d2}

    train = PairedDataset(
        src, trg, dicts, batch_size=args.batch_size, device=args.device
    ).shuffle_()
    train, valid = train.splits(test=None, dev=0.2)

    train.sort_(sort_by='trg')  # minimize padding

    print("Building model...")
    if args.grl:
        m = make_grl_rnn_encoder_decoder(
            args.layers, args.emb_dim, args.hid_dim, d, cell=args.cell,
            encoder_summary=args.encoder_summary, dropout=args.dropout,
            tie_weights=True, word_dropout=args.word_dropout,
            cond_dims=(args.emb_dim,), cond_vocabs=(len(conds_d),),
            conditional_decoder=False, train_init=args.train_init,
            reverse=args.reverse)

        # don't rely on GRL for early stopping (set its weight to 0)
        losses = ('ppl', {'loss': 'grl', 'format': 'ppl'})
        weights = {'ppl': 1, 'grl': 0}

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
    trainer.add_hook(make_report_hook(valid, 5), hooks_per_epoch=args.hook)

    (best_model, valid_loss), _ = trainer.train(
        args.epochs, args.checkpoint, shuffle=True)
    if not args.test:
        checkpoint.save(best_model, valid_loss)
