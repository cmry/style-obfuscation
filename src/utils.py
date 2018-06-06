
import copy


def make_early_stopping_hook(early_stopping):

    def hook(trainer, epoch, batch, checkpoint):
        loss = trainer.validate_model()
        trainer.log("validation_end", {"epoch": epoch, "loss": loss.pack()})
        early_stopping.add_checkpoint(
            loss.reduce(), copy.deepcopy(trainer.model).to('cpu'))

    return hook
