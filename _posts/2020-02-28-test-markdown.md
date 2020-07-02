---
layout: post
title: Privacy Aware Machine Learning
subtitle: Extending differential privacy to machine learning algorithms
gh-repo: daattali/beautiful-jekyll
gh-badge: [star, fork, follow]
tags: [test]
comments: true
---

The past decade has seen an exponential growth in Machine Learning and Artificial Intelligence - A field which not only aims on automation and prediction, but also relies on extensive datasets from users in the real world. The quality and quantity of a dataset is crucial to any Machine Learning model, but with the surge in the use of these public models which are available to everyone, people often turn a blind eye to one of the most important things: the security and privacy of the users. This has been a huge blind spot in the field of ML until very recently, when Cynthia Dwork from Microsoft Research talked about, and published her work on how to attain Privacy-aware-Machine-Learning models employing Differentially Private Algorithms.

"Why should I care?"


https://cdn-images-1.medium.com/max/750/0*6LDd7btZistasjgF

Few years ago, Netflix posted an "anonymous" dataset on Kaggle for a contest to get machine learning enthusiasts to work on efficient movie recommendation algorithms, only later realising that there was a privacy breach where some participants were able to actually infer individual user data. This is just one of many examples that highlights the low-privacy levels of ML models. This gives us the motivation to study it more before its too late and we get exploited.

### Popular attack and defense strategies

#### Shadow Models and Membership Inference attacks

https://cdn-images-1.medium.com/max/750/0*jOE0cos-c1iOXdTT.png

There are many notions and definitions of privacy and even more methods/attempts at trying to establish privacy, but most of them have flaws which can be exploited by an adversary through something called Membership Inference Attacks. Adversaries can attack the model to extract user data in a lot of different ways.

One of the modern ways to attack is to use a Machine learning model that trains on multiple other models called "shadow models" which has a black-box mechanism that predicts the inputs using the known public outputs with varying levels of confidence. Once the model is trained and is able to predict user data with sufficient amounts of confidence, the data leaks and the user is exposed, hence free to be exploited by the attacker. In an era where all our information is stored in databases on the internet, this leaked data could be absolutely anything, ranging from your family's genetic disorders, to predicting your current account balance.
