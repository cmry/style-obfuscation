
import copy


def make_check_hook(early_stopping=None, checkpoint=None):

    def hook(trainer, epoch, batch, _):
        loss, model = None, None
        if early_stopping is not None or checkpoint is not None:
            model = copy.deepcopy(trainer.model).to('cpu')
            loss = trainer.validate_model()
            trainer.log("validation_end", {"epoch": epoch, "loss": loss.pack()})
        if early_stopping is not None:
            early_stopping.add_checkpoint(loss.reduce(), model)
        if checkpoint is not None:
            checkpoint.save(model, loss.reduce())

    return hook
