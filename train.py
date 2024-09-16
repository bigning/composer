# %%
import json
import re
import os
import sys
from glob import glob

import time
import mlflow
import random
import numpy as np
import pandas as pd
import torch
import scanpy as sc
from datasets import Dataset, load_dataset
from transformers import TrainingArguments, HfArgumentParser

"""
from utils.data.readwrite_s3 import ReadWriteData
from data.common.tokenize import shard_hf_dataset
from data.common.main_loop import on_disk_sharding_loop

from train.masking_args import MaskingArguments
from train.masking_trainer import MaskingTrainer, MaskingDataCollator
from models.tokenizer.gene_tokenizer import GeneTokenizer
from models.scgpt.model import scGPTConfig, scGPTModel
from models.cscgpt.model import cscGPTConfig, cscGPTModel
from models.nicheformer.model import NicheformerConfig, NicheformerModel
from models.geneformer.model import GeneformerConfig, GeneformerModel
"""


# Composer
from composer.models.huggingface import HuggingFaceModel
from composer.utils import reproducibility
from composer import Trainer
from composer import Callback, Event, Logger, State
from composer.optim import DecoupledAdamW
from composer.optim.scheduler import (ConstantWithWarmupScheduler,
                                      CosineAnnealingWithWarmupScheduler,
                                      LinearWithWarmupScheduler)
from streaming import StreamingDataset
from torch.utils.data import DataLoader
from composer.utils import dist, get_device
import subprocess
from composer.loggers import MLFlowLogger

dist.initialize_dist(get_device(None), timeout=300)


print("haha done on rank: {dist.get_global_rank()}")
