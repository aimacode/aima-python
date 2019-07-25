# tf-faster-rcnn is deprecated:
For a good and more up-to-date implementation for faster/mask RCNN with multi-gpu support, please see the example in TensorPack [here](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN).

# tf-faster-rcnn
A Tensorflow implementation of faster RCNN detection framework by Xinlei Chen (xinleic@cs.cmu.edu). This repository is based on the python Caffe implementation of faster RCNN available [here](https://github.com/rbgirshick/py-faster-rcnn).

**Note**: Several minor modifications are made when reimplementing the framework, which give potential improvements. For details about the modifications and ablative analysis, please refer to the technical report [An Implementation of Faster RCNN with Study for Region Sampling](https://arxiv.org/pdf/1702.02138.pdf). If you are seeking to reproduce the results in the original paper, please use the [official code](https://github.com/ShaoqingRen/faster_rcnn) or maybe the [semi-official code](https://github.com/rbgirshick/py-faster-rcnn). For details about the faster RCNN architecture please refer to the paper [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](http://arxiv.org/pdf/1506.01497.pdf).

### Detection Performance
The current code supports **VGG16**, **Resnet V1** and **Mobilenet V1** models. We mainly tested it on plain VGG16 and Resnet101 (thank you @philokey!) architecture. As the baseline, we report numbers using a single model on a single convolution layer, so no multi-scale, no multi-stage bounding box regression, no skip-connection, no extra input is used. The only data augmentation technique is left-right flipping during training following the original Faster RCNN. All models are released.

With VGG16 (``conv5_3``):
  - Train on VOC 2007 trainval and test on VOC 2007 test, **70.8**.
  - Train on VOC 2007+2012 trainval and test on VOC 2007 test ([R-FCN](https://github.com/daijifeng001/R-FCN) schedule), **75.7**.
  - Train on COCO 2014 [trainval35k](https://github.com/rbgirshick/py-faster-rcnn/tree/master/models) and test on [minival](https://github.com/rbgirshick/py-faster-rcnn/tree/master/models) (*Iterations*: 900k/1190k), **30.2**.

With Resnet101 (last ``conv4``):
  - Train on VOC 2007 trainval and test on VOC 2007 test, **75.7**.
  - Train on VOC 2007+2012 trainval and test on VOC 2007 test (R-FCN schedule), **79.8**.
  - Train on COCO 2014 trainval35k and test on minival (900k/1190k), **35.4**.

More Results:
  - Train Mobilenet (1.0, 224) on COCO 2014 trainval35k and test on minival (900k/1190k), **21.8**.
  - Train Resnet50 on COCO 2014 trainval35k and test on minival (900k/1190k), **32.4**.
  - Train Resnet152 on COCO 2014 trainval35k and test on minival (900k/1190k), **36.1**.

Approximate *baseline* [setup](https://github.com/endernewton/tf-faster-rcnn/blob/master/experiments/cfgs/res101-lg.yml) from [FPN](https://arxiv.org/abs/1612.03144) (this repository does not contain training code for FPN yet):
  - Train Resnet50 on COCO 2014 trainval35k and test on minival (900k/1190k), **34.2**.
  - Train Resnet101 on COCO 2014 trainval35k and test on minival (900k/1190k), **37.4**.
  - Train Resnet152 on COCO 2014 trainval35k and test on minival (900k/1190k), **38.2**.

**Note**:
  - Due to the randomness in GPU training with Tensorflow especially for VOC, the best numbers are reported (with 2-3 attempts) here. According to my experience, for COCO you can almost always get a very close number (within ~0.2%) despite the randomness.
  - The numbers are obtained with the **default** testing scheme which selects region proposals using non-maximal suppression (TEST.MODE nms), the alternative testing scheme (TEST.MODE top) will likely result in slightly better performance (see [report](https://arxiv.org/pdf/1702.02138.pdf), for COCO it boosts 0.X AP).
  - Since we keep the small proposals (\< 16 pixels width/height), our performance is especially good for small objects.
  - We do not set a threshold (instead of 0.05) for a detection to be included in the final result, which increases recall.
  - Weight decay is set to 1e-4.
  - For other minor modifications, please check the [report](https://arxiv.org/pdf/1702.02138.pdf). Notable ones include using ``crop_and_resize``, and excluding ground truth boxes in RoIs during training.
  - For COCO, we find the performance improving with more iterations, and potentially better performance can be achieved with even more iterations.
  - For Resnets, we fix the first block (total 4) when fine-tuning the network, and only use ``crop_and_resize`` to resize the RoIs (7x7) without max-pool (which I find useless especially for COCO). The final feature maps are average-pooled for classification and regression. All batch normalization parameters are fixed. Learning rate for biases is not doubled.
  - For Mobilenets, we fix the first five layers when fine-tuning the network. All batch normalization parameters are fixed. Weight decay for Mobilenet layers is set to 4e-5.
  - For approximate [FPN](https://arxiv.org/abs/1612.03144) baseline setup we simply resize the image with 800 pixels, add 32^2 anchors, and take 1000 proposals during testing.
  - Check out [here](http://ladoga.graphics.cs.cmu.edu/xinleic/tf-faster-rcnn/)/[here](http://xinlei.sp.cs.cmu.edu/xinleic/tf-faster-rcnn/)/[here](https://drive.google.com/open?id=0B1_fAEgxdnvJSmF3YUlZcHFqWTQ) for the latest models, including longer COCO VGG16 models and Resnet ones.
  
![](data/imgs/gt.png)      |  ![](data/imgs/pred.png)
:-------------------------:|:-------------------------:
Displayed Ground Truth on Tensorboard |  Displayed Predictions on Tensorboard

### Additional features
Additional features not mentioned in the [report](https://arxiv.org/pdf/1702.02138.pdf) are added to make research life easier:
  - **Support for train-and-validation**. During training, the validation data will also be tested from time to time to monitor the process and check potential overfitting. Ideally training and validation should be separate, where the model is loaded every time to test on validation. However I have implemented it in a joint way to save time and GPU memory. Though in the default setup the testing data is used for validation, no special attempts is made to overfit on testing set.
  - **Support for resuming training**. I tried to store as much information as possible when snapshoting, with the purpose to resume training from the latest snapshot properly. The meta information includes current image index, permutation of images, and random state of numpy. However, when you resume training the random seed for tensorflow will be reset (not sure how to save the random state of tensorflow now), so it will result in a difference. **Note** that, the current implementation still cannot force the model to behave deterministically even with the random seeds set. Suggestion/solution is welcome and much appreciated.
  - **Support for visualization**. The current implementation will summarize ground truth boxes, statistics of losses, activations and variables during training, and dump it to a separate folder for tensorboard visualization. The computing graph is also saved for debugging.

### Prerequisites
  - A basic Tensorflow installation. The code follows **r1.2** format. If you are using r1.0, please check out the r1.0 branch to fix the slim Resnet block issue. If you are using an older version (r0.1-r0.12), please check out the r0.12 branch. While it is not required, for experimenting the original RoI pooling (which requires modification of the C++ code in tensorflow), you can check out my tensorflow [fork](https://github.com/endernewton/tensorflow) and look for ``tf.image.roi_pooling``.
  - Python packages you might not have: `cython`, `opencv-python`, `easydict` (similar to [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)). For `easydict` make sure you have the right version. I use 1.6.
  - Docker users: Since the recent upgrade, the docker image on docker hub (https://hub.docker.com/r/mbuckler/tf-faster-rcnn-deps/) is no longer valid. However, you can still build your own image by using dockerfile located at `docker` folder (cuda 8 version, as it is required by Tensorflow r1.0.) And make sure following Tensorflow installation to install and use nvidia-docker[https://github.com/NVIDIA/nvidia-docker]. Last, after launching the container, you have to build the Cython modules within the running container. 

### Installation
1. Clone the repository
  ```Shell
  git clone https://github.com/endernewton/tf-faster-rcnn.git
  ```

2. Update your -arch in setup script to match your GPU
  ```Shell
  cd tf-faster-rcnn/lib
  # Change the GPU architecture (-arch) if necessary
  vim setup.py
  ```

  | GPU model  | Architecture |
  | ------------- | ------------- |
  | TitanX (Maxwell/Pascal) | sm_52 |
  | GTX 960M | sm_50 |
  | GTX 1080 (Ti) | sm_61 |
  | Grid K520 (AWS g2.2xlarge) | sm_30 |
  | Tesla K80 (AWS p2.xlarge) | sm_37 |

  **Note**: You are welcome to contribute the settings on your end if you have made the code work properly on other GPUs. Also even if you are only using CPU tensorflow, GPU based code (for NMS) will be used by default, so please set **USE_GPU_NMS False** to get the correct output.


3. Build the Cython modules
  ```Shell
  make clean
  make
  cd ..
  ```

4. Install the [Python COCO API](https://github.com/pdollar/coco). The code requires the API to access COCO dataset.
  ```Shell
  cd data
  git clone https://github.com/pdollar/coco.git
  cd coco/PythonAPI
  make
  cd ../../..
  ```

### Setup data
Please follow the instructions of py-faster-rcnn [here](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to setup VOC and COCO datasets (Part of COCO is done). The steps involve downloading data and optionally creating soft links in the ``data`` folder. Since faster RCNN does not rely on pre-computed proposals, it is safe to ignore the steps that setup proposals.

If you find it useful, the ``data/cache`` folder created on my side is also shared [here](http://ladoga.graphics.cs.cmu.edu/xinleic/tf-faster-rcnn/cache.tgz).

### Demo and Test with pre-trained models
1. Download pre-trained model
  ```Shell
  # Resnet101 for voc pre-trained on 07+12 set
  ./data/scripts/fetch_faster_rcnn_models.sh
  ```
  **Note**: if you cannot download the models through the link, or you want to try more models, you can check out the following solutions and optionally update the downloading script:
  - Another server [here](http://xinlei.sp.cs.cmu.edu/xinleic/tf-faster-rcnn/).
  - Google drive [here](https://drive.google.com/open?id=0B1_fAEgxdnvJSmF3YUlZcHFqWTQ).

2. Create a folder and a soft link to use the pre-trained model
  ```Shell
  NET=res101
  TRAIN_IMDB=voc_2007_trainval+voc_2012_trainval
  mkdir -p output/${NET}/${TRAIN_IMDB}
  cd output/${NET}/${TRAIN_IMDB}
  ln -s ../../../data/voc_2007_trainval+voc_2012_trainval ./default
  cd ../../..
  ```

3. Demo for testing on custom images
  ```Shell
  # at repository root
  GPU_ID=0
  CUDA_VISIBLE_DEVICES=${GPU_ID} ./tools/demo.py
  ```
  **Note**: Resnet101 testing probably requires several gigabytes of memory, so if you encounter memory capacity issues, please install it with CPU support only. Refer to [Issue 25](https://github.com/endernewton/tf-faster-rcnn/issues/25).

4. Test with pre-trained Resnet101 models
  ```Shell
  GPU_ID=0
  ./experiments/scripts/test_faster_rcnn.sh $GPU_ID pascal_voc_0712 res101
  ```
  **Note**: If you cannot get the reported numbers (79.8 on my side), then probably the NMS function is compiled improperly, refer to [Issue 5](https://github.com/endernewton/tf-faster-rcnn/issues/5).

### Train your own model
1. Download pre-trained models and weights. The current code support VGG16 and Resnet V1 models. Pre-trained models are provided by slim, you can get the pre-trained models [here](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models) and set them in the ``data/imagenet_weights`` folder. For example for VGG16 model, you can set up like:
   ```Shell
   mkdir -p data/imagenet_weights
   cd data/imagenet_weights
   wget -v http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
   tar -xzvf vgg_16_2016_08_28.tar.gz
   mv vgg_16.ckpt vgg16.ckpt
   cd ../..
   ```
   For Resnet101, you can set up like:
   ```Shell
   mkdir -p data/imagenet_weights
   cd data/imagenet_weights
   wget -v http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
   tar -xzvf resnet_v1_101_2016_08_28.tar.gz
   mv resnet_v1_101.ckpt res101.ckpt
   cd ../..
   ```

2. Train (and test, evaluation)
  ```Shell
  ./experiments/scripts/train_faster_rcnn.sh [GPU_ID] [DATASET] [NET]
  # GPU_ID is the GPU you want to test on
  # NET in {vgg16, res50, res101, res152} is the network arch to use
  # DATASET {pascal_voc, pascal_voc_0712, coco} is defined in train_faster_rcnn.sh
  # Examples:
  ./experiments/scripts/train_faster_rcnn.sh 0 pascal_voc vgg16
  ./experiments/scripts/train_faster_rcnn.sh 1 coco res101
  ```
  **Note**: Please double check you have deleted soft link to the pre-trained models before training. If you find NaNs during training, please refer to [Issue 86](https://github.com/endernewton/tf-faster-rcnn/issues/86). Also if you want to have multi-gpu support, check out [Issue 121](https://github.com/endernewton/tf-faster-rcnn/issues/121).

3. Visualization with Tensorboard
  ```Shell
  tensorboard --logdir=tensorboard/vgg16/voc_2007_trainval/ --port=7001 &
  tensorboard --logdir=tensorboard/vgg16/coco_2014_train+coco_2014_valminusminival/ --port=7002 &
  ```

4. Test and evaluate
  ```Shell
  ./experiments/scripts/test_faster_rcnn.sh [GPU_ID] [DATASET] [NET]
  # GPU_ID is the GPU you want to test on
  # NET in {vgg16, res50, res101, res152} is the network arch to use
  # DATASET {pascal_voc, pascal_voc_0712, coco} is defined in test_faster_rcnn.sh
  # Examples:
  ./experiments/scripts/test_faster_rcnn.sh 0 pascal_voc vgg16
  ./experiments/scripts/test_faster_rcnn.sh 1 coco res101
  ```

5. You can use ``tools/reval.sh`` for re-evaluation


By default, trained networks are saved under:

```
output/[NET]/[DATASET]/default/
```

Test outputs are saved under:

```
output/[NET]/[DATASET]/default/[SNAPSHOT]/
```

Tensorboard information for train and validation is saved under:

```
tensorboard/[NET]/[DATASET]/default/
tensorboard/[NET]/[DATASET]/default_val/
```

The default number of training iterations is kept the same to the original faster RCNN for VOC 2007, however I find it is beneficial to train longer (see [report](https://arxiv.org/pdf/1702.02138.pdf) for COCO), probably due to the fact that the image batch size is one. For VOC 07+12 we switch to a 80k/110k schedule following [R-FCN](https://github.com/daijifeng001/R-FCN). Also note that due to the nondeterministic nature of the current implementation, the performance can vary a bit, but in general it should be within ~1% of the reported numbers for VOC, and ~0.2% of the reported numbers for COCO. Suggestions/Contributions are welcome.

### Citation
If you find this implementation or the analysis conducted in our report helpful, please consider citing:

    @article{chen17implementation,
        Author = {Xinlei Chen and Abhinav Gupta},
        Title = {An Implementation of Faster RCNN with Study for Region Sampling},
        Journal = {arXiv preprint arXiv:1702.02138},
        Year = {2017}
    }
    
Or for a formal paper, [Spatial Memory Network](https://arxiv.org/abs/1704.04224):

    @article{chen2017spatial,
      title={Spatial Memory for Context Reasoning in Object Detection},
      author={Chen, Xinlei and Gupta, Abhinav},
      journal={arXiv preprint arXiv:1704.04224},
      year={2017}
    }

For convenience, here is the faster RCNN citation:

    @inproceedings{renNIPS15fasterrcnn,
        Author = {Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun},
        Title = {Faster {R-CNN}: Towards Real-Time Object Detection
                 with Region Proposal Networks},
        Booktitle = {Advances in Neural Information Processing Systems ({NIPS})},
        Year = {2015}
    }
