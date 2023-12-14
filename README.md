# CS482/682 Final Project Report Group 18: Semi-Supervised Domain Adaptation (SSDA)

## Install

`pip install -r requirements.txt`

## Data preparation (DomainNet)

To get data, run

`sh download_data.sh`

The images will be stored in the following way.

`./data/multi/real/category_name`,

`./data/multi/sketch/category_name`

The dataset split files are stored as follows,

`./data/txt/multi/labeled_source_images_real.txt`,

`./data/txt/multi/unlabeled_target_images_sketch_3.txt`,

`./data/txt/multi/validation_target_images_sketch_3.txt`.

## Training

To run MME training using resnet34,

`sh run_mme_train.sh gpu_id MME resnet34`

To run ISCL training using resnet34,

`sh run_iscl_train.sh gpu_id resnet34`

where, gpu_id = 0,1,2,3...

### Reference

The code is built on repository contributed by [Kuniaki Saito](http://cs-people.bu.edu/keisaito/) and [Donghyun Kim](https://cs-people.bu.edu/donhk/)
If you consider using this code or its derivatives, please consider citing:

```
@article{saito2019semi,
  title={Semi-supervised Domain Adaptation via Minimax Entropy},
  author={Saito, Kuniaki and Kim, Donghyun and Sclaroff, Stan and Darrell, Trevor and Saenko, Kate},
  journal={ICCV},
  year={2019}
}
```
