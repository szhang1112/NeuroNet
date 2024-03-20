from argparse import ArgumentParser

import h5py
import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src.models import Beluga, ConvNet
from sklearn.metrics import average_precision_score, precision_recall_curve, auc, roc_curve

import warnings

class ChromatinModel(pl.LightningModule):
    def __init__(self,
            learning_rate=1e-3,
            batch_size=64,
            num_tasks = 4,
            ):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = ConvNet(num_tasks)
        self.lossfn = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        embedding = self.backbone(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()
        y_hat = self.forward(x)
        loss = self.lossfn(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss
    
    def e_step(self, batch, split='valid'):
        x, y = batch
        x = x.float()
        y = y.float()
        y_hat = self.forward(x)
        loss = self.lossfn(y_hat, y)
        y_hat = F.sigmoid(y_hat)
        return {f"{split}_loss": loss,
            "y": y.cpu(),
            "y_hat": y_hat.cpu()
        }
    
    def e_epoch_end(self, outputs, split='valid'):
        tmp = []
        for o in outputs:
            tmp.append(o[f'{split}_loss'])
        self.log(f'{split}:loss', torch.Tensor(tmp).mean())
        tmp_y = []
        tmp_yhat = []
        for o in outputs:
            tmp_y.append(o['y'])
            tmp_yhat.append(o['y_hat'])
        y = torch.cat(tmp_y, 0)
        y_hat = torch.cat(tmp_yhat, 0)

        targets = [
            'MN-H3K27ac',
            'MN-H3K4me1',
            'MN-H3K4me3',
            'MN-ATAC'
        ]
        colors = [
            'red',
            'green',
            'blue',
            'pink'
        ]
        
        aurocs = []        
        figure, ax = plt.subplots()
        for i, target, color in zip(range(y.shape[1]), targets, colors):
            fpr, tpr, thresholds = roc_curve(y[:,i].numpy(), y_hat[:,i].numpy(), pos_label=1)
            auroc = auc(fpr,tpr)
            aurocs.append(auroc)
            self.log(f'{split}-{target}-AUROC', auroc)
            plt.plot(fpr, tpr, marker='.', label=f'{target}-AUROC:{auroc:.4f}', color=color,  markersize=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        tensorboard = self.logger.experiment
        tensorboard.add_figure(f'ROC-{split}(Epoch:{self.current_epoch})', figure)
            
        auprcs = []
        figure, ax = plt.subplots()
        for i, target, color in zip(range(y.shape[1]), targets, colors):
            precision, recall, thresholds = precision_recall_curve(y[:,i].numpy(), y_hat[:,i].numpy(), pos_label=1)
            auprc = auc(recall, precision)
            auprcs.append(auprc)
            self.log(f'{split}-{target}-AUPRC', auc(recall, precision))
            plt.plot(recall, precision, marker='.', label=f'{target}-AUPRC:{auprc:.4f}', color=color, markersize=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        tensorboard = self.logger.experiment
        tensorboard.add_figure(f'PRC-{split}(Epoch:{self.current_epoch})', figure)
        
        self.log(f'{split}-mean-AUROC', np.mean(aurocs))
        self.log(f'{split}-mean-AUPRC', np.mean(auprcs))

    def validation_step(self, batch, batch_idx):
        return self.e_step(batch, 'valid')

    def validation_epoch_end(self, validation_step_outputs):
        self.e_epoch_end(validation_step_outputs, 'valid')

    def test_step(self, batch, batch_idx):
        return self.e_step(batch, 'test')
    
    def test_epoch_end(self, test_step_outputs):
        self.e_epoch_end(test_step_outputs, 'test')

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate,weight_decay=1e-6, momentum= 0.9)
        return opt
    
    def train_dataloader(self):
        train_dataset = H5Dataset('data/four_marker/train.mat', 'trainxdata', 'traindata')
        return DataLoader(train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=32)

    def val_dataloader(self):
        valid_dataset = H5Dataset('data/four_marker/valid.mat', 'validxdata', 'validdata')
        return DataLoader(valid_dataset, batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        test_dataset = H5Dataset('data/four_marker/test.mat', 'testxdata', 'testdata')
        return DataLoader(test_dataset, batch_size=self.hparams.batch_size, num_workers=32)


class H5Dataset(Dataset):
    def __init__(self, path, dataX, dataY):
        self.file_path = path
        self.dataX = dataX
        self.dataY = dataY
        self.X=None
        self.Y=None

        with h5py.File(self.file_path, 'r') as file:
            self.dataset_len = file[self.dataX].shape[2]
 
    def __getitem__(self, index):
        if self.X is None:
            self.X = h5py.File(self.file_path, 'r')[self.dataX]
            self.Y = h5py.File(self.file_path, 'r')[self.dataY]
        return self.X[:,:,index].transpose(), self.Y[:,index]
 
    def __len__(self):
        return self.dataset_len

def cli_main():
    warnings.filterwarnings("ignore")
    
    pl.seed_everything(42)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--learning_rate', default=0.2, type=float)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # ------------
    # model
    # ------------
    model = ChromatinModel(batch_size=args.batch_size,learning_rate=args.learning_rate)
    
    # ------------
    # training
    # ------------
    checkpoint_callback = ModelCheckpoint(
        monitor = 'valid-mean-AUPRC',
        mode='max',
        save_weights_only=True
    )
    early_stop_callback = EarlyStopping(
        monitor='valid-mean-AUPRC',
        min_delta=0.00,
        patience=20,
        verbose=False,
        mode="max"
    )
    trainer = pl.Trainer.from_argparse_args(
        args,
        enable_model_summary=True,
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            ModelSummary(max_depth=-1)
        ],
        default_root_dir = 'four_new_logs',
    )
    trainer.fit(model)

    # ------------
    # testing
    # ------------
    result = trainer.test()
    print(result)


if __name__ == '__main__':
    cli_main()


