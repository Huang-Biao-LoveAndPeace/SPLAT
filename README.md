## SPLAT

This repository contains the code for reproducing the results in our paper:


---

### Contents

1. [Pre-requisites](#pre-requisites)
2. [Training Models](#training-models)
3. [Crafting Adversarial Examples](#crafting-adversarial-examples)

&nbsp;

---

### Pre-requisites

Download the TinyImageNet dataset from this [link](https://tiny-imagenet.herokuapp.com/), and unzip the downloaded file under `datasets/originals`. The following command will help.

```
  $ mkdir -p datasets/originals
  $ unzip tiny-imagenet-200.zip datasets/originals/
  $ python datasets.py
```

----

### Training Models

You can use the following documents to train multi-exit models (SDNs,MSNET,MSDNET,subtitudemodel).

```
  sh log.train_sdns.sh
  sh log.train_msnet.sh
  sh log.train_msdnet.sh
  sh log.train_subtitude_models.sh
```

The trained model will be stored under the `models` folder.
(e.g. `models/<dataset>/<dataset>_<network>_<nettype>`)


----

### Crafting Adversarial Examples

To craft DeepSloth adversarial samples and baseline examples, you can use the following documents.You should train models before crafting adversarial examples
```
  sh log.run_deepsloth_attack.sh
  sh log.run_ours_attack.sh
  sh log.run_deepsloth_transfer_attack
```



