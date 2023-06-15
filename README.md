# Self-supervised Metric Learning

This repo hosts the code and model of Self-supervision to Metric Learning for Music Similarity-based Retrieval and Auto-tagging.

### Installation

- This repo follows and use the [CLMR repo](https://github.com/Spijkervet/CLMR), Installation and preparation and pre-train follow that repo.
- Clone this repo and create a link to CLMR as follows.
```
git clone https://github.com/ta603/Self-supervised_Metric_Learning.git
cd Self-supervised_Metric_Learning/
pip install -r requirements.txt
ln -s path/to/CLMR/clmr clmr
ln -s path/to/CLMR/data data
ln -s path/to/CLMR/config config
```

### Training and Evaluation
- Below is the command corresponding to "ours G" in Table1.
```
python metric_learning.py --dataset magnatagatune --gpus 1 --workers 8 --max_epochs 200 --alpha 1.0 --load_only_clmr_part 1 --checkpoint_path path/to/pratrain_model.ckpt --finetune_aug_less 1
```

### License
This project is under the CC-BY-SA 4.0 license. See [LICENSE](LICENSE) for details.
