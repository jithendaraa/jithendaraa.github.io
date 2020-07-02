---
layout: post
title: Privacy Aware Machine Learning
subtitle: Extending differential privacy to machine learning algorithms
gh-repo: daattali/beautiful-jekyll
gh-badge: [star, fork, follow]
tags: [test]
comments: true
---

<img src="https://cdn-images-1.medium.com/max/1000/0*0cN6yO_kHXZklVg1.jpg" alt="Markdown Monster icon" style="margin: 10px;" />

The past decade has seen an exponential growth in Machine Learning and Artificial Intelligence - A field which not only aims on automation and prediction, but also relies on extensive datasets from users in the real world. The quality and quantity of a dataset is crucial to any Machine Learning model, but with the surge in the use of these public models which are available to everyone, people often turn a blind eye to one of the most important things: the security and privacy of the users. This has been a huge blind spot in the field of ML until very recently, when Cynthia Dwork from Microsoft Research talked about, and published her work on how to attain Privacy-aware-Machine-Learning models employing Differentially Private Algorithms.

> Why should I care?

<img src="https://cdn-images-1.medium.com/max/750/0*6LDd7btZistasjgF" alt="Markdown Monster icon" style="float: left; margin: 10px;" />

<br/>

Few years ago, Netflix posted an "anonymous" dataset on Kaggle for a contest to get machine learning enthusiasts to work on efficient movie recommendation algorithms, only later realising that there was a privacy breach where some participants were able to actually infer individual user data. This is just one of many examples that highlights the low-privacy levels of ML models. This gives us the motivation to study it more before its too late and we get exploited.

### Popular attack and defense strategies

* **Shadow Models and Membership Inference attacks**

<img src="https://cdn-images-1.medium.com/max/750/0*jOE0cos-c1iOXdTT.png" alt="Markdown Monster icon" style="float: left; margin: 10px;" />

There are many notions and definitions of privacy and even more methods/attempts at trying to establish privacy, but most of them have flaws which can be exploited by an adversary through something called Membership Inference Attacks. Adversaries can attack the model to extract user data in a lot of different ways.

One of the modern ways to attack is to use a Machine learning model that trains on multiple other models called **"shadow models"** which has a black-box mechanism that predicts the inputs using the known public outputs with varying levels of confidence. Once the model is trained and is able to predict user data with sufficient amounts of confidence, the data leaks and the user is exposed, hence free to be exploited by the attacker. In an era where all our information is stored in databases on the internet, this leaked data could be absolutely anything, ranging from your family's genetic disorders, to predicting your current account balance. **Insert hackerman meme**

<img src="https://66.media.tumblr.com/6e5d75677d297bdfc1a4b80b1660daa3/tumblr_opdkedNera1ticdtho1_400.png" alt="hackaaa" style="margin: 10px;" />
<br/>

* **The mediator**

Another way is to have a mediator who can somehow selectively give information to the adversary based on whether there will be loss of a particular individual's data or not. But the problem with this is that, just by the mediator telling "I can't give you this information", the attacker can infer that there is some sensitive data and can make predictions on what that might be.

* **Perturbing the data with random noise**

<img src="https://cdn-images-1.medium.com/max/1000/0*BkyU0pQV6gZ2e0H0.png" alt="Markdown Monster icon" style="float: left; margin: 10px;"/>

Yet another interesting approach would be the addition of random noises from a uniform distribution to our dataset. This approach too fails, since for larger data we can simply average out the values and can get an estimate of our original dataset.

* **Count queries**

A common and efficient way to attack is using count queries, where our attacker asks a query Q like: "What are the marks of student A?" to get a count and this leaks individual data. A direct approach everyone intuitively takes to this problem is: "Why don't you restrict the queries to only group queries? This won't leak individual data. Right?". But, this is a rather naive approach since the attacker , with a few cleverly placed group queries can infer individual user data. For example, by asking "How many students got above 90 in the class?" and "How many students other than student A got above 90 in the class?", the attacker can infer student A's marks simply by subtracting the two results. Thus restricting ourselves to group query also, fails. This is big brain time.

<img src="https://i.ytimg.com/vi/42_1ur2PcVo/hqdefault.jpg" alt="Markdown Monster icon" style="margin: 10px;" />

## An Introduction to Differential Privacy

Differential privacy has been an existing concept for long in Mathematics. The main idea of this concept, is as follows:

>If our model is able to predict things with or without your data to an amount not differing by an amount more than a multiplicative exp(ε), then we are said to be ε-differentially private or ε-DP

Let's get into the math!
Spoiler: This section is probably gonna contain more memes than math, haha :)

<img src="https://cdn.kapwing.com/final_5c11554bb472020012f692ef_968276.jpg" alt="Markdown Monster icon" style="margin: 10px;" />

Formally, a randomized mechanism M: X → Y is ε-Differentially private if for all neighboring inputs x' near x, and for all outputs E ∈ Y we have:
**P[M(x) ∈ Y] ≤ exp(ε) P[M(x') ∈ Y] or**
**P[M(x) ∈ Y] ≤ (1+ε) * P[M(x') ∈ Y] for small ε**

