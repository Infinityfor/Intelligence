本文件夹用于记录人工智能导论课程学习进度及其作业提交



项目题目：生成狗图像
项目来源：kaggle（https://www.kaggle.com/c/generative-dog-images/overview）
项目拟解决问题：提供一种识别狗的思路
项目简介：生成对抗网络（GAN）是Ian Goodfellow在2014年发明的一类机器学习系统。两个神经网络在游戏中相互竞争。在给定训练集的情况下，该技术将学习生成具有与训练集相同的统计数据的新数据。
本项目将训练生成模型来创建狗的图像。仅这一次……没有基本真实数据可供您预测。在这里，您将提交图像并根据来自预训练的神经网络的图像被分类为狗的程度进行评分。
为什么是狗？我们之所以选择狗，是因为，谁不喜欢看可爱的小狗的照片？此外，狗可以分为许多子类别（品种，颜色，大小），使其成为生成图像的理想候选者。
生成方法（尤其是GAN）目前在Kaggle上的各个地方都用于数据增强。他们的潜力是巨大的。他们可以学习模仿任何领域的数据分布：照片，素描，音乐和散文。如果成功的话，您不仅可以帮助您改进生成图像的技术水平，而且还可以使我们将来在各种领域中进行更多的实验。
选题原因：项目趣味性大。作为一个深度学习实例，该项目涉及到卷积网络多个知识点（卷积，池化，残差，损失函数）的同时项目难度较小，但我们将学习使用tensorflow或pytorch重新构建这个项目，因此。项目周期符合课程课时。

数据集来源：Stanford Dogs Dataset
可能涉及到的深度学习模型:GANs(DCGGAN，BigGAN,)，CNN
可能使用的框架：tensorflow/pytorch
项目来源：kaggle


参考项目:
  比赛结果: https://www.kaggle.com/c/generative-dog-images/discussion/106305 
pytorch:
https://www.kaggle.com/tikutiku/gan-dogs-starter-biggan
https://www.kaggle.com/yukia18/sub-rals-ac-biggan-with-minibatchstddev

tensorflow2.0:
https://www.kaggle.com/jobzhf88/generative-dog-images-dcgan-1-commit
https://www.kaggle.com/harishvutukuri/dcgan-dogs

项目入门:
GAN家族原理介绍：https://blog.floydhub.com/gans-story-so-far/
pytorch DCGAN实战: https://www.w3cschool.cn/pytorch/pytorch-rqs93bn0.html


