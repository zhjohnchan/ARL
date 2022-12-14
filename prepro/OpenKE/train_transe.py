from OpenKE.openke.config import Trainer
from OpenKE.openke.data import TrainDataLoader
from OpenKE.openke.module.loss import MarginLoss
from OpenKE.openke.module.model import TransE
from OpenKE.openke.module.strategy import NegativeSampling


def train_transe():
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path="data/knowledge/",
        nbatches=100,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0)

    # define the model
    transe = TransE(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=256,
        p_norm=1,
        norm_flag=True)

    # define the loss function
    model = NegativeSampling(
        model=transe,
        loss=MarginLoss(margin=5.0),
        batch_size=train_dataloader.get_batch_size()
    )

    # train the model
    trainer = Trainer(model=model, data_loader=train_dataloader, train_times=1000, alpha=1.0, use_gpu=True)
    trainer.run()
    transe.save_checkpoint('data/knowledge/ent_embeddings.ckpt')
