---
layout: post
title: An introduction to Privacy Aware Machine Learning
subtitle: Extending differential privacy to machine learning algorithms
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

Differential privacy has been an existing concept for long in Mathematics. On a very basic level, we can think of this as providing some sort of measure on the privacy that we have. Once we have this notion, the task becomes very clear: increase this privacy measure to the maximum extent possible. 

The main idea of this concept, is as follows:

>If our model is able to predict things with or without your data to an amount not differing by an amount more than a multiplicative exp(ε), then we are said to be ε-differential private or ε-DP

Let's get into the math!
Spoiler: This section is probably gonna contain more memes than math, haha :)

<img src="https://i.kym-cdn.com/entries/icons/mobile/000/021/464/14608107_1180665285312703_1558693314_n.jpg" alt="Markdown Monster icon" style="margin: 10px;" />

Formally, a randomized mechanism M: X → Y is ε-Differentially private if for all neighboring inputs **x'** near **x**, and for all outputs E ∈ Y we have:
**P[M(x) ∈ Y] ≤ exp(ε) P[M(x') ∈ Y] or**     
**P[M(x) ∈ Y] ≤ (1+ε) * P[M(x') ∈ Y] for small ε**

Note that a neighboring data set **x'** of **x** means that **x** and **x'** differ by exactly one entry - ie., the participation or non-participation of exactly one user. This is like saying 
> "Hey, our performance didn't take a significant hit with or without your data, so you might as well participate by giving your data and joining the survey!"   

<img src="https://cdn-images-1.medium.com/max/1000/0*h41f8kdsMrBwXiBH.gif" alt="Markdown Monster icon" style="margin: 10px;" />

It is important to note that the equation is also symmetrical in nature (with respect to **x** and **x'**). ε is a privacy parameter or the privacy budget. A lower ε means higher privacy and vice-versa.
For example, privacy budget at 0 is said to be completely private though in reality it might not be possible to attain complete privacy. Attaining **ε=0** is very ideal and analogous to attaining an efficiency of 100% in an engine - we'd love to get there but it isn't realistic or practical. Why is this impractical? For this, let us look at a DP-algorithm (there are quite a few of them, but let us look at the laplace mechanism)

## The Laplace Mechanism and attaining ε-DP

<img src="https://cdn-images-1.medium.com/max/750/0*bEV9Y0BB-yZDBGfz.png" alt="Markdown Monster icon" style="margin: 10px;" />

I'm going to contradict my previous statement on how "random noise" addition doesn't help with attaining privacy. This is true, but not always. Random noise addition can be useful in some exceptional cases like sampling noises from the Laplacian Distribution as it is a differentially-private curve. That is, it satisfies the notion of differential privacy (the equations in the above section).
As the name of this method suggests, we attain differential privacy by applying a laplacian noise on our dataset. This has the power to immediately transform our dataset into a differentially private one with a privacy budget ε, if the scale of our laplacian noise was **Δf/ε**, where **Δf** is the L1-sensitivity of our dataset. This however is only for a single count query. For a column vector query of d dimensions, the new **ε** becomes d times the old one by the <a href="http://proceedings.mlr.press/v37/kairouz15.pdf">**composition theorem**</a> .

<br/>
Hence to make this d **ε** -DP to **ε** -DP we need to multiply the scale by d which in turn increases the amount of noise. Thus, it is clearly visible that for large datasets, our laplacian scale gets really large thus having a higher impact on our dataset thereby affecting our performance with increasing d. One way to overcome this is to collect and use more data entries(n) during training so that the large noise added is compensated by the increased number of entries. But this is practically not feasible. Thus our focus now falls on reducing dimensions to a smaller value to make sure of the optimal trade-off between privacy and utility. Privacy and utility are negatively correlated in the sense that increase in one compromises on the other. Thus, there is an increasing motivation to look at dimensionality reduction techniques like PCA,VAE, ALI-GAN, etc., with the aim of minimizing information loss to optimise the privacy-utility trade-off.

as this will compromise on our performance and our model will lose all meaning.  Our aim is to get our ε as close to 0 as possible. On the other hand, as **ε** → ∞, we are said to be "blatantly non-private" or completely non-private.

<img src="https://cdn-images-1.medium.com/max/750/0*lsQ_2UbtuDyVLiGm" alt="Markdown Monster icon" style="margin: 10px;" />
<br/>

The results of count queries are noisy due to the DP-algorithm and the attacker is unable to infer useful data. In this way our model still doesn't lose significant performance but also doesn't leak user data.Differential privacy can be attained through many methods like the Exponential Mechanism or the Laplace Mechanism. Our focus however, will be limited to the Laplace mechanism.



