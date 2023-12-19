import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


from scripts.pipeline import Train_Dataset, DataModule
from scripts.utils import plot_images
from networks.pix2pix import Pix2Pix
from networks.autoencoder import Autoencoder


PATH_TO_DATA = "./dataset/train"
PATH_TO_CSV = "./dataset/train_dataset.csv"
SEED = 42
BATCH_SIZE = 128

# Defining a seed for testing

torch.manual_seed(SEED)

training_dataset = DataLoader(Train_Dataset(), batch_size=BATCH_SIZE, shuffle = True, num_workers=10)
val_dataset = DataLoader(Train_Dataset("./dataset/val_dataset.csv", "./dataset/val"), batch_size=BATCH_SIZE, shuffle=True)

for batch in val_dataset:
    test_batch = batch
    break

# Defining trainer

model = Autoencoder(0.0002, 0.5, 0.5, test_batch)

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='./',
    filename='./checkpoints/auto64-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    mode='min'
)

"""

early_stop_callback = EarlyStopping(
   monitor='val_loss',
   min_delta=0.05,
   patience=5,
   verbose=False,
   mode='min',
)

"""

logger = WandbLogger(name = "Colorisation-autoencoder")
trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=100,
    logger = logger,
    precision='16-mixed',
    callbacks=[checkpoint_callback]
)



trainer.fit(model, train_dataloaders=training_dataset, val_dataloaders=val_dataset)
torch.save(model, './checkpoints/auto64_weights')