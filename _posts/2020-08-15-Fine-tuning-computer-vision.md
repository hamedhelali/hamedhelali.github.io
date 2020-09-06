---
title: "Task-agnostic pretraining and task-specific fine tuning in computer vision"
author: "Hamed Helali"
date: 2020-08-15
tags: [Computer Vision, Semi-supervised Learning, Pretraining, Fine tuning]
categories: Blog-post
header:
  image: "/images/computer-vision.jpg"
excerpt: "Computer Vision, Semi-supervised Learning, Pretraining, Fine tuning"
mathjax: "true"
---

Labeled data is expensive and scarce and computer vision researchers seek to make more effective use of unlabeled data to train models. Google Brain’s Ting Chen and colleagues, proposed a new semi-supervised learning algorithm that outperforms the previous state-of-the-art by a large margin. The framework of pretraining a big (deep and wide) network with a large amount of unlabeled data and fine-tuning it on a few labeled examples is widely used in natural language processing. However, it has received little attention in computer vision. The suggested semi-supervised algorithm employs such a paradigm to learn on ImageNet.  

The presented framework comprises three steps: unsupervised pretraining of a big ResNet, supervised fine-tuning of the model and distillation with unlabeled examples.
* **Pretraining:** Authors propose SimCLRv2 as a modification to SimCLR (i.e. previous state-of-the-art method). Similar to the previous version, for each image in a randomly sampled mini-batch, two different views are created using random crop, color distortion and Gaussian blur. Then a contrastive loss function is defined based on similarity of pairs of images augmented from the same image and dissimilarity of non-related images. In comparison to the previous version, improvements are made by using larger ResNet models, a three layer projection head and fine tuning outputs of the first layer rather than throwing away the entire head.
* **Fine-tuning:** The pretrained model is then fine-tuned to the specific classification task using a few labeled examples in a supervised learning paradigm.
* **Distillation:** To make further improvement, the fine-tuned model is used as a teacher model to train a new student model. It means that output distributions of the teacher model are used as labels for the student model when it learns on unlabeled examples. Consequently, a smaller student model performs better (compared to being trained by labeled data from scratch) due to the transfer of task-specific knowledge from the teacher. Even if the student model is the same size as the teacher, the ultimate performance will be improved.

This framework is evaluated on an ImageNet dataset with 1.28 million images and 1000 classes. It achieves 76.6% top-1 accuracy (compared to 63.0% with the previous state-of-the-art) in case of using 1% of labeled data. With only 10% of labeled data, this framework outperforms the case of supervised learning with 100% of the labeled data. The other significant result is that if you have fewer labels, it becomes more important to have bigger models. It is a considerable improvement in leveraging unlabeled data in computer vision. Furthermore, the last aforementioned result is counter intuitive. It could be expected that bigger networks easily overfit a few labeled examples. However, self-supervision seems to be counter to the notion of overfitting. 
