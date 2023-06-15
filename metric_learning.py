# =====================================================================
# Copyright (c) 2022-2023 Sony Computer Science Laboratories, Inc.,
# Tokyo, Japan. All rights reserved.
# This source code is licensed under the license found in the LICENSE
# file.
# =====================================================================
import pickle
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import Parameter
from collections import OrderedDict
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm
from sklearn import metrics
from itertools import chain
from simclr import SimCLR
from simclr.modules import NT_Xent
import datetime
import faiss
import numpy as np
import os
import math

# Audio Augmentations
from torchaudio_augmentations import (
    RandomApply,
    ComposeMany,
    RandomResizedCrop,
    PolarityInversion,
    Noise,
    Gain,
    HighLowPass,
    Delay,
    PitchShift,
    Reverb,
)

from clmr.data import ContrastiveDataset
from clmr.datasets import get_dataset
from clmr.evaluation import evaluate
from clmr.models import SampleCNN
from clmr.utils import yaml_config_hook



# NormSoftmaxLoss is based on classification_metric_learning implementation
# https://github.com/azgo14/classification_metric_learning/blob/master/metric_learning/modules/losses.py
class NormSoftmaxLoss(nn.Module):
    """
    L2 normalize weights and apply temperature scaling on logits.
    """
    def __init__(self,
                 dim,
                 num_instances,
                 temperature=0.05):
        super(NormSoftmaxLoss, self).__init__()
        print(f"NormSoftMaxLoss :: dim={dim}")
        self.weight = Parameter(torch.Tensor(num_instances, dim))
        # Initialization from nn.Linear (https://github.com/pytorch/pytorch/blob/v1.0.0/torch/nn/modules/linear.py#L129)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        self.temperature = temperature
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, embeddings, instance_targets):
        norm_embeddings = nn.functional.normalize(embeddings, dim=1)
        prediction_logits = nn.functional.linear(norm_embeddings, self.weight)
        loss = self.loss_fn(prediction_logits / self.temperature, instance_targets)
        return loss



class MetricLearning(LightningModule):
    def __init__(self, args, encoder, output_dim):
        super().__init__()
        self.new_arch1 = True
        self.save_hyperparameters(args)

        self.encoder = encoder
        self.output_dim = output_dim
        self.loss_weight = args.alpha
        self.loss2_weight = args.loss2_weight
        self.loss1_ratio = args.loss1_ratio
        self.n_features = (
            self.encoder.fc.in_features
        )  # get dimensions of last fully-connected layer
        self.model = SimCLR(self.encoder, self.hparams.projection_dim, self.n_features)
        self.remap = nn.Sequential(nn.Linear(self.n_features, self.output_dim))
        self.standardize = nn.LayerNorm(self.n_features, elementwise_affine=False)
        self.criterion, self.criterion2 = self.configure_criterion()
        print(f'MetricLearning :: alpha={self.loss_weight} loss2_weight={self.loss2_weight} loss1_ratio={self.loss1_ratio}')

    def forward(self, x_i, x_j, y):
        #_, _, z_i, z_j = self.model(x_i, x_j)
        z_i = self.model.encoder(x_i)
        z_j = self.model.encoder(x_j)
        sz_i = self.standardize(z_i)
        sz_j = self.standardize(z_j)
        #preds_i = self.remap(sz_i)
        #preds_j = self.remap(sz_j)
        if self.new_arch1:
            z_i = self.model.projector(z_i)
            z_j = self.model.projector(z_j)
        loss1 = self.criterion(z_i, z_j) * self.loss_weight / self.loss1_ratio
        loss2 = (self.criterion2(sz_i, y)+self.criterion2(sz_j, y))/2 * self.loss2_weight
        return loss1+loss2, sz_i

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_i = x[:, 0, :]
        x_j = x[:, 1, :]
        loss, emb = self.forward(x_i, x_j, y)
        self.log("Train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_i = x[:, 0, :]
        x_j = x[:, 1, :]
        loss, emb = self.forward(x_i, x_j, y)
        self.log("Valid/loss", loss)
        return loss

    def configure_criterion(self):
        batch_size = self.hparams.batch_size

        criterion = NT_Xent(batch_size, self.hparams.temperature, world_size=1)
        criterion2 = NormSoftmaxLoss(self.n_features, self.output_dim, temperature=1)
        return criterion, criterion2

    def configure_optimizers(self):
        scheduler = None
        print("self.hparams.finetuner_learning_rate=",self.hparams.finetuner_learning_rate)
        optimizer = torch.optim.Adam(
            chain(
                self.model.parameters(), self.remap.parameters(), self.criterion2.parameters()
            ),
            lr=self.hparams.finetuner_learning_rate, weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.1,
            patience=5,
        )
        if scheduler:
            return {"optimizer": optimizer, "lr_scheduler": scheduler,"monitor": "Valid/loss"}
        else:
            return {"optimizer": optimizer}



def load_model_checkpoint(state_dict) -> OrderedDict:
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if "model." in k:
            new_state_dict[k.replace("model.", "")] = v
    return new_state_dict



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SSML")

    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    # additional args.
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--load_only_clmr_part', type=int, default=1)
    parser.add_argument('--finetune_aug_less', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--loss2_weight', type=float, default=1.0)
    parser.add_argument('--loss1_ratio', type=float, default=22.0)
    parser.add_argument('--eval_only', type=int, default=0)
    parser.add_argument('--train_only', type=int, default=0)
    #parser.add_argument('', default=1)

    args = parser.parse_args()
    pl.seed_everything(args.seed)

    # ------------
    # data augmentations
    # ------------
    min_gain = -20
    max_gain = -1
    pitch_cents_min = -700
    pitch_cents_max = 700
    lowpass_freq_low = 2200.0
    lowpass_freq_high = 4000.0
    highpass_freq_low = 200.0
    highpass_freq_high = 1200.0
    delay_func = Delay
    min_snr=0.0001
    max_snr=0.01

    if args.finetune_aug_less:
        print('finetune aug less')
        train_transform = [RandomResizedCrop(n_samples=args.audio_length)]
        num_augmented_samples = 2
    else:
        train_transform = [
            RandomResizedCrop(n_samples=args.audio_length),
            RandomApply([PolarityInversion()], p=args.transforms_polarity),
            RandomApply([Noise(min_snr=min_snr, max_snr=max_snr)], p=args.transforms_noise),
            RandomApply([Gain(min_gain=min_gain, max_gain=max_gain)], p=args.transforms_gain),
            RandomApply(
                [HighLowPass(
                    sample_rate=args.sample_rate,
                    lowpass_freq_low=lowpass_freq_low,
                    lowpass_freq_high=lowpass_freq_high,
                    highpass_freq_low=highpass_freq_low,
                    highpass_freq_high=highpass_freq_high
                )], p=args.transforms_filters
            ),
            RandomApply([delay_func(sample_rate=args.sample_rate)], p=args.transforms_delay),
            RandomApply(
                [
                    PitchShift(
                        n_samples=args.audio_length,
                        sample_rate=args.sample_rate,
                        pitch_cents_min=pitch_cents_min,
                        pitch_cents_max=pitch_cents_max,
                    )
                ],
                p=args.transforms_pitch,
            ),
            RandomApply(
                [Reverb(sample_rate=args.sample_rate)], p=args.transforms_reverb
            ),
        ]
        num_augmented_samples = 2

    # ------------
    # dataloaders
    # ------------
    train_dataset = get_dataset(args.dataset, args.dataset_dir, subset="train")
    valid_dataset = get_dataset(args.dataset, args.dataset_dir, subset="valid")
    if args.dataset=='msd3var':
        train_dataset.set_max_track_length(args.msd3var_length)
        valid_dataset.set_max_track_length(args.msd3var_length)

    contrastive_train_dataset = ContrastiveDataset(
        train_dataset,
        input_shape=(1, args.audio_length),
        transform=ComposeMany(
            train_transform, num_augmented_samples=num_augmented_samples
        ),
    )

    contrastive_valid_dataset = ContrastiveDataset(
        valid_dataset,
        input_shape=(1, args.audio_length),
        transform=ComposeMany(
            train_transform, num_augmented_samples=num_augmented_samples
        ),
    )

    train_loader = DataLoader(
        contrastive_train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True,
        shuffle=True,
    )

    valid_loader = DataLoader(
        contrastive_valid_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True,
        shuffle=False,
    )

    # ------------
    # encoder
    # ------------
    encoder = SampleCNN(
        strides=[3, 3, 3, 3, 3, 3, 3, 3, 3],
        supervised=args.supervised,
        out_dim=train_dataset.n_classes,
    )

    # ------------
    # model
    # ------------
    module = MetricLearning(args, encoder, output_dim=train_dataset.n_classes)
    if 0:
        print(" *** Freeze SimCLR model ***")
        module.model.eval()
        for param in module.model.parameters():
            #print(param)
            param.requires_grad = False

    logger = TensorBoardLogger("runs", name="ML-NewArch1-a1-{}".format(args.dataset))
    if args.checkpoint_path:
        print("load checkpoint data")
        state_dict = torch.load(args.checkpoint_path)
        if args.load_only_clmr_part:
            print("load clmr part")
            new_state_dict = load_model_checkpoint(state_dict['state_dict'])
            module.model.load_state_dict(new_state_dict)
        else:
            print("load all part")
            module.load_state_dict(state_dict['state_dict'])
        print("load complete")

    # ------------
    # training
    # ------------
    if not args.eval_only:
        early_stop_callback = EarlyStopping(
            monitor="Valid/loss", patience=10
        )
        trainer = Trainer.from_argparse_args(
            args,
            logger=logger,
            max_epochs=args.max_epochs,
            log_every_n_steps=100,
            check_val_every_n_epoch=1,
            callbacks=[early_stop_callback],
        )
        print('[[[ START ]]]',datetime.datetime.now())
        trainer.fit(module, train_loader, valid_loader)
        trainer.save_checkpoint("runs/current.ckpt")
        print('[[[ FINISH ]]]',datetime.datetime.now())

    # ------------
    # evaluation
    # ------------
    if args.train_only:
        print('skip evaluation')
        import sys
        sys.exit(0)
    test_dataset = get_dataset(args.dataset, args.dataset_dir, subset="test")
    if args.dataset=='msd3var':
        train_dataset.set_max_track_length(500) # max length(in sec) of msd
    contrastive_test_dataset = ContrastiveDataset(
        test_dataset,
        input_shape=(1, args.audio_length),
        transform=None,
    )
    test_loader = DataLoader(
        contrastive_test_dataset,
        batch_size=args.finetuner_batch_size,
        num_workers=args.workers,
        shuffle=False,
    )
    device = "cuda:0" if args.gpus else "cpu"
    module.to(device)
    module.eval()
    track_emb_array = []
    est_array = []
    gt_array = []
    SPLIT_UNIT = 250
    with torch.no_grad():
        for idx in tqdm(range(len(contrastive_test_dataset))):
            _, label = contrastive_test_dataset[idx]
            batch = contrastive_test_dataset.concat_clip(idx, args.audio_length)
            batch = batch.to(device)
            #print('batch.shape', batch.shape)

            if batch.shape[0]>SPLIT_UNIT:
                #print('split')
                n = math.ceil(batch.shape[0]/SPLIT_UNIT)
                splitted_batch = torch.chunk(batch, n, dim=0)
                _ = []
                for mini_batch in splitted_batch:
                    z = module.model.encoder(mini_batch)
                    #print('z.shape', z.shape)
                    _.append(z)
                z = torch.cat(_, dim=0)
            else:
                z = module.model.encoder(batch)
            embeddings = module.standardize(z)
            #print('embeddings.shape', embeddings.shape)
            norm_embeddings = nn.functional.normalize(embeddings, dim=1)
            track_emb = norm_embeddings.mean(dim=0)
            prediction_logits = nn.functional.linear(norm_embeddings, module.criterion2.weight)
            preds = torch.sigmoid(prediction_logits)
            track_prediction = preds.mean(dim=0)

            track_emb_array.append(track_emb)
            est_array.append(track_prediction)
            gt_array.append(label)
    track_emb_array = torch.stack(track_emb_array, dim=0).cpu().numpy()
    est_array = torch.stack(est_array, dim=0).cpu().numpy()
    gt_array = torch.stack(gt_array, dim=0).cpu().numpy()

    ########################
    ### AUC evaluation
    ########################
    roc_aucs = metrics.roc_auc_score(gt_array, est_array, average="macro")
    pr_aucs = metrics.average_precision_score(gt_array, est_array, average="macro")
    print("ROC-AUC:",roc_aucs)
    print("PR-AUC:",pr_aucs)

    ########################
    ### R@K evaluation
    ########################
    gt_array_org = gt_array
    gt_array = []
    emb_array_org = track_emb_array
    emb_array = []
    zero_tag_count = 0
    for i in range(len(gt_array_org)):
        if gt_array_org[i].sum() == 0:
            zero_tag_count += 1
        else:
            gt_array.append(gt_array_org[i])
            emb_array.append(emb_array_org[i])
    gt_array = np.array(gt_array)
    emb_array = np.array(emb_array)
    print('zero_tag_count=',zero_tag_count)
    print(gt_array.shape)
    print(emb_array.shape)

    for i in range(len(emb_array)):
        emb_array[i] = emb_array[i] / np.linalg.norm(emb_array[i], ord=2)


    d = emb_array.shape[1]
    print('class=',d)
    index = faiss.IndexFlatIP(d) 
    index.add(emb_array)

    k = 8
    D, I = index.search(emb_array, k+1)

    results = I
    labels = gt_array
    expected_result_size = k + 1
    recall_at_k = np.zeros((k,))

    for i in range(len(labels)):
        pos = 0 # keep track recall at pos
        j = 0 # looping through results
        _ = np.zeros((k,))
        _r = np.zeros(len(labels[0]), dtype=np.int)
        while pos < k:
            if i==results[i,j]:
                # Only skip the document when query and index sets are the exact same
                j += 1
                continue
            _r = _r | labels[results[i,j]].astype('int')
            n_and = (labels[i].astype('int') & _r).sum()
            n = labels[i].sum()
            if n!=0:
                _[pos] = n_and / n
            j += 1
            pos += 1
        recall_at_k += _

    recall_at_k_ratio = recall_at_k / float(len(labels)) * 100.0

    # result
    for i in range(len(recall_at_k_ratio)):
        if i+1==1 or i+1==2 or i+1==4 or i+1==8:
            print(f"R@{i+1} = {recall_at_k_ratio[i]}")
