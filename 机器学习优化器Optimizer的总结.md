# 机器学习优化器Optimizer的总结

机器学习中，通常有很多方法来试图寻找模型的最优解。比如常见的

**梯度下降法(Gradient Descent)、**

> 随机梯度下降法SGD
> 批量梯度下降法BGD

**动量优化法(Momentum)、**

**自适应学习率优化算法**

> AdaGrad算法
> RMSProp算法
> Adam算法
> lazyadam算法

下面来一一介绍：

## 一、梯度下降法(Gradient Descent)

在微积分中，对**多元函数的参数求 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta) 偏导数**，把求得的各个参数的导数以向量的形式写出来就是梯度。梯度就是函数变化最快的地方。梯度下降是迭代法的一种，在求解机器学习算法的模型参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta) 时，即无约束问题时，梯度下降是最常采用的方法之一。顾名思义，**梯度下降法的计算过程就是沿梯度下降的方向求解极小值，也可以沿梯度上升方向求解最大值。** 假设模型参数为 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta) ，损失函数为 ![[公式]](https://www.zhihu.com/equation?tex=J%5Cleft%28%5Ctheta%5Cright%29) ，损失函数 ![[公式]](https://www.zhihu.com/equation?tex=J%5Cleft%28%5Ctheta%5Cright%29) 关于参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta) 的偏导数，也就是梯度为 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctriangledown+_%7B%5Ctheta%7DJ%5Cleft+%28+%5Ctheta++%5Cright+%29) *，学习率为* ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) *，则使用梯度下降法更新参数为：*
![[公式]](https://www.zhihu.com/equation?tex=%5C%5C+%5Ctheta_%7Bt%2B1%7D+%3D+%5Ctheta_%7Bt%7D+-%5Calpha+%5Ccdot+%5Ctriangledown+_%7B%5Ctheta%7DJ%5Cleft+%28+%5Ctheta++%5Cright+%29+)

![img](https://pic4.zhimg.com/80/v2-1647d6c9052fe2fd60d366c91326d2bb_1440w.jpg)

评价：梯度下降法主要有两个缺点:

> **训练速度慢**：每走一步都要要计算调整下一步的方向，下山的速度变慢。在应用于大型数据集中，每输入一个样本都要更新一次参数，且每次迭代都要遍历所有的样本。会使得训练过程及其缓慢，需要花费很长时间才能得到收敛解。
> **容易陷入局部最优解：**由于是在有限视距内寻找下山的反向。当陷入平坦的洼地，会误以为到达了山地的最低点，从而不会继续往下走。所谓的局部最优解就是鞍点。落入鞍点，梯度为0，使得模型参数不在继续更新。

梯度下降法目前主要分为三种方法,区别在于**每次参数更新时计算的样本数据量不同**：

> 批量梯度下降法(BGD, Batch Gradient Descent)
> 随机梯度下降法(SGD, Stochastic Gradient Descent)
> 小批量梯度下降法(Mini-batch Gradient Descent)

## 1.1 批量梯度下降法BGD

假设训练样本总数为n，样本为 ![[公式]](https://www.zhihu.com/equation?tex=%5Cleft%5C%7B++%5Cleft%28+x%5E%7B1%7D%2Cy%5E%7B1%7D+%5Cright%29%2C+%5Ccdots%2C+%5Cleft%28x%5E%7Bn%7D%2C+y%5E%7Bn%7D%5Cright%29+%5Cright%5C%7D) ，模型参数为 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta) ，损失函数为 ![[公式]](https://www.zhihu.com/equation?tex=J%5Cleft%28%5Ctheta%5Cright%29) ，在第i对样本 ![[公式]](https://www.zhihu.com/equation?tex=%5Cleft+%28+x%5E%7Bi%7D%2Cy%5E%7Bi%7D+%5Cright+%29) 上损失函数关于参数的梯度为 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctriangledown_%7B%5Ctheta%7DJ_%7Bi%7D%5Cleft%28%5Ctheta%2C+x%5E%7Bi%7D%2C+y%5E%7Bi%7D+%5Cright%29) , 学习率为 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha)，则使用BGD更新参数为：

![[公式]](https://www.zhihu.com/equation?tex=%5C%5C++%5Ctheta_%7Bt%2B1%7D+%3D+%5Ctheta_%7Bt%7D+-%5Calpha_%7Bt%7D+%5Ccdot+%5Csum_%7Bi%3D1%7D%5E%7Bn%7D+%5Ctriangledown_%7B%5Ctheta%7DJ_%7Bi%7D%5Cleft%28%5Ctheta%2C+x%5E%7Bi%7D%2C+y%5E%7Bi%7D+%5Cright%29+)

由上式可以看出，**每进行一次参数更新，需要计算整个数据样本集，因此导致批量梯度下降法的速度会比较慢，**尤其是数据集非常大的情况下，收敛速度就会非常慢，但是由于每次的下降方向为总体平均梯度，它得到的会是一个全局最优解。

评价：

> 批量梯度下降法比标准梯度下降法训练时间短，且每次下降的方向都很正确。

## 1.2 随机梯度下降法SGD

随机梯度下降法，不像BGD每一次参数更新，需要计算整个数据样本集的梯度，而是每次参数更新时，**仅仅选取一个样本** ![[公式]](https://www.zhihu.com/equation?tex=%5Cleft+%28+x%5E%7Bi%7D%2Cy%5E%7Bi%7D%5Cright+%29) 计算其梯度，参数更新公式为：

![[公式]](https://www.zhihu.com/equation?tex=%5C%5C+%5Ctheta_%7Bt%2B1%7D+%3D+%5Ctheta_%7Bt%7D+-%5Calpha+%5Ccdot+%5Ctriangledown_%7B%5Ctheta%7DJ_%7Bi%7D%5Cleft%28%5Ctheta%2C+x%5E%7Bi%7D%2C+y%5E%7Bi%7D+%5Cright%29+)

可以看到BGD和SGD是两个极端，**SGD由于每次参数更新仅仅需要计算一个样本的梯度**，训练速度很快，即使在样本量很大的情况下，可能只需要其中一部分样本就能迭代到最优解，由于每次迭代并不是都向着整体最优化方向，导致梯度下降的波动非常大，更容易从一个局部最优跳到另一个局部最优，准确度下降。

**SGD缺点：**

- 选择合适的learning rate比较困难 ，学习率太低会收敛缓慢，学习率过高会使收敛时的波动过大
- 所有参数都是用同样的learning rate
- SGD容易收敛到局部最优，并且在某些情况下可能被困在鞍点

## 1.3 小批量梯度下降法

小批量梯度下降法就是结合BGD和SGD的折中，对于含有n个训练样本的数据集，每次参数更新，选择一个**大小为m** ![[公式]](https://www.zhihu.com/equation?tex=%5Cleft+%28+m+%3C+n+%5Cright+%29) 的mini-batch数据样本计算其梯度，其参数更新公式如下：

![[公式]](https://www.zhihu.com/equation?tex=%5C%5C++%5Ctheta_%7Bt%2B1%7D+%3D+%5Ctheta_%7Bt%7D+-%5Calpha+%5Csum_%7Bi%3Dx%7D%5E%7Bi%3Dx%2Bm-1%7D+%5Ccdot+%5Ctriangledown_%7B%5Ctheta%7DJ_%7Bi%7D%5Cleft%28%5Ctheta%2C+x%5E%7Bi%7D%2C+y%5E%7Bi%7D+%5Cright%29+)

小批量梯度下降法即保证了训练的速度，又能保证最后收敛的准确率，目前的SGD默认是小批量梯度下降算法。

```text
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss, global_step=global_step)
```

评价：

> 虽然**小批量梯度下降法**需要走很多步的样子，但是对梯度的要求很低（计算梯度快）。而对于引入噪声，大量的理论和实践工作证明，只要噪声不是特别大，都能很好地收敛。
> 应用大型数据集时，训练速度很快。比如每次从百万数据样本中，取几百或者几千个数据点，算一个梯度，更新一下模型参数。相比于**批量梯度下降法BGD**的遍历全部样本，每输入一个样本更新一次参数，要快得多。
> **小批量梯度下降法**在选择小批量样本时，同时会引入噪声，使得权值更新的方向不一定正确。

## 二、动量优化法

动量优化方法引入物理学中的动量思想，**加速梯度下降**，Momentum算法。当我们将一个小球从山上滚下来，没有阻力时，它的动量会越来越大，但是如果遇到了阻力，速度就会变小，动量优化法就是借鉴此思想，**使得梯度方向在不变的维度上，参数更新变快，梯度有所改变时，更新参数变慢，这样就能够加快收敛并且减少动荡。**

## 2.1 Momentum

momentum算法思想：**参数更新时在一定程度上保留之前更新的方向，同时又利用当前batch的梯度微调最终的更新方向，简言之就是通过积累之前的动量来(\*previous_sum_of_gradient\*)加速当前的梯度。**

假设 *gradient* 表示t时刻的动量(t时刻的梯度)， ![[公式]](https://www.zhihu.com/equation?tex=u) 表示动量因子，通常取值0.9或者近似值，在**随机梯度下降法**SGD的基础上增加动量，则参数更新公式如下：

> *sum_of_gradient = u \* gradient + previous_sum_of_gradient \* decay_rate*
> *delta = -learning_rate \* sum_of_gradient*
> *theta += delta*

**在梯度方向改变时，momentum能够降低参数更新速度，从而减少震荡；**
**在梯度方向相同时，momentum可以加速参数更新， 从而加速收敛。**

总而言之，momentum能够加速SGD收敛，抑制震荡。

> 动量移动得更快(因为它积累的所有动量)
> **动量有机会逃脱局部极小值**(因为动量可能推动它脱离局部极小值)。同样，我们将在后面看到，它也将更好地通过高原区

## 三、自适应学习率优化算法

每个参与训练的参数设置不同的学习率，在整个学习过程中通过一些算法自动适应这些参数的学习率。

**自适应学习率优化算法**针对于机器学习模型的学习率，**传统的优化算法要么将学习率设置为常数要么根据训练次数调节学习率**。极大忽视了**学习率其他变化的可能性**。然而，学习率对模型的性能有着显著的影响，因此需要采取一些策略来想办法更新学习率，从而提高训练速度。

> 我们先来看一下使用统一的全局学习率的缺点可能出现的问题：
> 对于某些参数，通过**算法已经优化到了极小值附近，但是有的参数仍然有着很大的梯度。**
> 如果学习率太小，则梯度很大的参数会有一个很慢的收敛速度；
> 如果学习率太大，则已经优化得差不多的参数可能会出现不稳定的情况。
> 解决方案：
> 对**每个参与训练的参数**设置不同的学习率，在整个学习过程中通过一些算法自动适应这些参数的学习率。
> Delta-ba-delta：
> ***如果损失与某一指定参数的偏导的符号相同，那么学习率应该增加；***
> ***如果损失与该参数的偏导的符号不同，那么学习率应该减小。***

**基于小批量的训练数据**的性能更好的自适应学习率算法主要有：

> AdaGrad算法
> RMSProp算法
> Adam算法
> lazyadam算法

## 3.1 AdaGrad算法

思想：AdaGrad(**Ada**ptive **Grad**ient)算法，**独立地适应所有模型参数的学习率**，**缩放每个参数**反比于 其**所有梯度历史平均值 总和 的平方根**。

> 具有代价函数**最大梯度的参数**相应地有个**快速下降的学习率**，
> 而具有小梯度的参数在学习率上有相对较小的下降。

算法描述：

AdaGrad算法优化策略一般可以表示为：

> *sum_of_gradient_squared = previous_sum_of_gradient_squared + gradient²*
> *delta = -learning_rate \* gradient* ***/ sqrt(sum_of_gradient_squared)***
> *theta += delta*
>
> **详细执行流程表述：**
>
> **全局学习率 ![[公式]](https://www.zhihu.com/equation?tex=%5Csigma)** ，初始化的参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Comega) ，一个为了数值稳定而创建的小常数 ![[公式]](https://www.zhihu.com/equation?tex=%5Cdelta) （建议默认取 ![[公式]](https://www.zhihu.com/equation?tex=%5Cdelta%3D10x%5E%7B-7%7D) ），以及一个梯度累积变量 ![[公式]](https://www.zhihu.com/equation?tex=r) (初始化 ![[公式]](https://www.zhihu.com/equation?tex=r%3D0) )。算法主体，循环执行以下步骤，在没有达到停止的条件前不会停止。
> (1)取出小批量数据 ![[公式]](https://www.zhihu.com/equation?tex=%5Cleft%5C%7B+x+_1%2Cx_2%2C...%2Cx_m%5Cright%5C%7D) 数据对应的目标用 ![[公式]](https://www.zhihu.com/equation?tex=y_%7Bi%7D) 表示
> (2)在小批量数据的基础上按照以下公式计算梯度：
> ![[公式]](https://www.zhihu.com/equation?tex=g%5Cleftarrow%5Cfrac%7B1%7D%7Bm%7D%5Cvee_%7B%5Comega%7D%5Csum_%7Bi%7DL%28%7Bf%28x_i%3B%5Comega%29%2Cy_i%7D%29)
> (3)累积平方梯度，并刷新r，过程如公式：
> ![[公式]](https://www.zhihu.com/equation?tex=r%5Cleftarrow+r%2Bg%5Codot+g)
> (4)计算参数更新量( ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Csigma%7D%7B%5Cdelta%2B%5Csqrt%7Br%7D%7D) 会被逐元素应用)：
> ![[公式]](https://www.zhihu.com/equation?tex=%5CDelta%5Comega%3D-%5Cfrac%7B%5Csigma%7D%7B%5Cdelta%2B%5Csqrt%7Br%7D%7D%5Codot+g)
> (5)根据 ![[公式]](https://www.zhihu.com/equation?tex=%5CDelta%5Comega) 更新参数:
> ![[公式]](https://www.zhihu.com/equation?tex=%5Comega%5Cleftarrow+%5Comega%2B%5CDelta%5Comega)

Adagrad 解决这个问题的思路是: **你已经更新的特征（幅度）越多，你将来更新的就越少，这样就有机会让其它特征(例如稀疏特征)赶上来。**用可视化的术语来说，更新这个特征的程度即在这个维度中移动了多少，这个概念由梯度平方的累积和表达。

> 稀疏特征的平均梯度通常很小，所以这些特征的训练速度要慢得多。

这个属性让 AdaGrad (以及其它类似的基于梯度平方的方法，如 RMSProp 和 Adam)**更好地避开鞍点。**Adagrad 将采取直线路径，而梯度下降(或相关的动量)采取的方法是“让我先滑下陡峭的斜坡，然后才可能担心较慢的方向”。有时候，原版梯度下降可能非常满足的仅仅停留在鞍点，那里两个方向的梯度都是0。

![img](https://pic3.zhimg.com/80/v2-0633c31b6c9bab8f0d4e64fd2d4c1936_1440w.jpg)

假定一个多分类问题，i表示第i个分类，t表示第t迭代同时也表示分类i累计出现的次数。 ![[公式]](https://www.zhihu.com/equation?tex=%5Ceta_%7B0%7D) 表示初始的学习率取值一般为0.01，ϵ是一个取值很小的数（一般为1e-8）为了避免分母为0。 ![[公式]](https://www.zhihu.com/equation?tex=W_%7Bt%7D) 表示t时刻即第t迭代模型的参数， ![[公式]](https://www.zhihu.com/equation?tex=g_%7Bt%2Ci%7D+%3D+%5CDelta+J%28W_%7Bt%2Ci%7D%29) 表示t时刻，指定分类i，代价函数J(⋅)关于W的梯度。

> 从表达式可以看出，
> 对出现比较多的类别数据，Adagrad给予越来越小的学习率，
> 而对于比较少的类别数据，会给予较大的学习率。
> 因此Adagrad适用于数据稀疏或者分布不平衡的数据集。
>
> Adagrad 的主要优势在于不需要人为的调节学习率，它可以自动调节；
> 缺点在于，随着迭代次数增多，学习率会越来越小，最终会趋近于0。

## 3.2 RMSprop（root mean square prop）

更好的理解RMSprop

[模型优化-RMSprop_温染的笔记-CSDN博客_rmsprop](https://link.zhihu.com/?target=https%3A//blog.csdn.net/weixin_43378396/article/details/90744928)

在AdaGrad算法的基础上经过修改得到。AdaGrad中，每个参数的![[公式]](https://www.zhihu.com/equation?tex=%5CDelta%5Comega+)都反比于**其所有梯度历史平方值总和的平方根**，但RMSProp算法采用了**指数衰减平均的方式**淡化遥远过去的历史对当前步骤参数更新量![[公式]](https://www.zhihu.com/equation?tex=%5CDelta%5Comega+)的影响。RMSProp引入了一个新的参数![[公式]](https://www.zhihu.com/equation?tex=%5Crho)（*decay_rate*），用于控制历史梯度值的衰减速率。

> *sum_of_gradient_squared = previous_sum_of_gradient_squared* **** decay_rate**+ gradient²* **** (1- decay_rate)***
> *delta = -learning_rate \* gradient / sqrt(sum_of_gradient_squared)*
> *theta += delta*
>
> 详细执行流程表述：
>
> 设全局学习率为 ![[公式]](https://www.zhihu.com/equation?tex=%5Csigma) ，历史梯度之的衰减速率参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Crho) ，初始化的参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Comega) ，一个为了数值稳定而创建的小常数 ![[公式]](https://www.zhihu.com/equation?tex=%5Cdelta) （建议默认取 ![[公式]](https://www.zhihu.com/equation?tex=%5Cdelta%3D10x%5E%7B-7%7D) ），以及一个梯度累积变量 ![[公式]](https://www.zhihu.com/equation?tex=r) (初始化 ![[公式]](https://www.zhihu.com/equation?tex=r%3D0) )。算法主体，循环执行以下步骤，在没有达到停止的条件前不会停止。
> (1)取出小批量数据 ![[公式]](https://www.zhihu.com/equation?tex=%5Cleft%5C%7B+x+_1%2Cx_2%2C...%2Cx_m%5Cright%5C%7D) 数据对应的目标用 ![[公式]](https://www.zhihu.com/equation?tex=y_%7Bi%7D) 表示
> (2)在小批量数据的基础上按照以下公式计算梯度：
> ![[公式]](https://www.zhihu.com/equation?tex=g%5Cleftarrow%5Cfrac%7B1%7D%7Bm%7D%5Cvee_%7B%5Comega%7D%5Csum_%7Bi%7DL%28%7Bf%28x_i%3B%5Comega%29%2Cy_i%7D%29)
> (3)累积平方梯度，并刷新r，过程如公式：
> ![[公式]](https://www.zhihu.com/equation?tex=r%5Cleftarrow+%5Crho+r%2B%281-%5Crho%29g%5Codot+g)
> (4)计算参数更新量( ![[公式]](https://www.zhihu.com/equation?tex=-%5Cfrac%7B%5Csigma%7D%7B%5Csqrt%7B%5Cdelta%2Br%7D%7D) 会被逐元素应用)：
> ![[公式]](https://www.zhihu.com/equation?tex=%5CDelta%5Comega%3D-%5Cfrac%7B%5Csigma%7D%7B%5Csqrt%7B%5Cdelta%2Br%7D%7D%5Codot+g)
> (5)根据 ![[公式]](https://www.zhihu.com/equation?tex=%5CDelta%5Comega) 更新参数:
> ![[公式]](https://www.zhihu.com/equation?tex=%5Comega%5Cleftarrow+%5Comega%2B%5CDelta%5Comega)

更精确地说，梯度的平方和实际上是梯度平方的衰减和。衰减率表明的是只是最近的梯度平方有意义，而很久以前的梯度基本上会被遗忘。顺便说一句，“衰减率”这个术语有点用词不当。与我们在动量中看到的衰减率不同，除了衰减之外，这里的衰减率还有一个缩放效应: 它以一个因子(1 - 衰减率)向下缩放整个项。换句话说，如果衰减率设置为0.99，除了衰减之外，梯度的平方和将是 sqrt (1-0.99) = 0.1，因此对于相同的学习率，这一步大10倍。

## 3.4 Adam: Adaptive Moment Estimation

Adam (**Ada**ptive **M**oment Estimation)同时兼顾了动量和 RMSProp 的优点。Adam在实践中效果很好，因此在最近几年，它是深度学习问题的常用选择。

让我们来看看它是如何工作的:

> *sum_of_gradient = previous_sum_of_gradient \* beta1 + gradient \* (1 - beta1)* [类似Momentum]
> *sum_of_gradient_squared = previous_sum_of_gradient_squared \* beta2 + gradient² \* (1- beta2)* [类似RMSProp]
> *delta = -learning_rate \* sum_of_gradient / sqrt(sum_of_gradient_squared)*
> *theta += delta*
>
> **详细执行流程表述：**
>
> 设全局学习率为 ![[公式]](https://www.zhihu.com/equation?tex=%5Csigma)(建议默认 ![[公式]](https://www.zhihu.com/equation?tex=%5Csigma%3D0.001))，**矩估计的指数衰减速率**为 ![[公式]](https://www.zhihu.com/equation?tex=%5Crho_1) 和 ![[公式]](https://www.zhihu.com/equation?tex=%5Crho_2) ( ![[公式]](https://www.zhihu.com/equation?tex=%5Crho_1) 和 ![[公式]](https://www.zhihu.com/equation?tex=%5Crho_2) 在区间 ![[公式]](https://www.zhihu.com/equation?tex=%5B0%2C1%29) 内，建议默认分别为0.9和0.990)，初始化的参数为 ![[公式]](https://www.zhihu.com/equation?tex=%5Comega) ，一个为了数值稳定而创建的小常数 ![[公式]](https://www.zhihu.com/equation?tex=%5Cdelta) （建议默认取 ![[公式]](https://www.zhihu.com/equation?tex=%5Cdelta%3D10x%5E%7B-8%7D) ），**初始值为0的一阶和二阶矩变量** ![[公式]](https://www.zhihu.com/equation?tex=s%2Cr) ，**以及一个时间步计数器** ![[公式]](https://www.zhihu.com/equation?tex=t) (初始化 ![[公式]](https://www.zhihu.com/equation?tex=t%3D0) )。然后就是算法的主体，循环执行以下步骤，在没有达到停止的条件前不会停止。
> (1)取出小批量数据 ![[公式]](https://www.zhihu.com/equation?tex=%5Cleft%5C%7B+x+_1%2Cx_2%2C...%2Cx_m%5Cright%5C%7D) 数据对应的目标用 ![[公式]](https://www.zhihu.com/equation?tex=y_%7Bi%7D) 表示
> (2)在小批量数据的基础上按照以下公式计算梯度：
> ![[公式]](https://www.zhihu.com/equation?tex=g%5Cleftarrow%5Cfrac%7B1%7D%7Bm%7D%5Cvee_%7B%5Comega%7D%5Csum_%7Bi%7DL%28%7Bf%28x_i%3B%5Comega%29%2Cy_i%7D%29)
> (3)刷新时间步：
> ![[公式]](https://www.zhihu.com/equation?tex=t%5Cleftarrow+t%2B1)
> (4)**更新一阶有偏矩估计**：
> ![[公式]](https://www.zhihu.com/equation?tex=s%5Cleftarrow%5Crho_1s%2B%281-%5Crho_1%29g)
> (5)**更新二阶有偏矩估计**：
> ![[公式]](https://www.zhihu.com/equation?tex=r%5Cleftarrow%5Crho_2+r%2B%281-%5Crho_2%29g%5Codot+g)
> (6)对一阶矩的偏差进行修正：
> ![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7Bs%7D%5Cleftarrow%5Cfrac%7Bs%7D%7B1-%5Crho_%7B1%7D%5E%7Bt%7D%7D)
> (7)对二阶矩的偏差进行修正：
> ![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7Br%7D%5Cleftarrow%5Cfrac%7Bs%7D%7B1-%5Crho_%7B2%7D%5E%7Bt%7D%7D)
> (8)计算参数的更新量：
> ![[公式]](https://www.zhihu.com/equation?tex=%5CDelta%5Comega%3D-%5Csigma%5Cfrac%7B%5Ctilde%7Bs%7D%7D%7B%5Csqrt%7B%5Ctilde%7Br%7D%2B%5Cdelta%7D%7D)
> (9)根据 ![[公式]](https://www.zhihu.com/equation?tex=%5CDelta%5Comega) 更新参数：
> ![[公式]](https://www.zhihu.com/equation?tex=%5Comega%5Cleftarrow%5Comega%2B%5CDelta%5Comega)

Beta1（ ![[公式]](https://www.zhihu.com/equation?tex=%5Crho_1) ）是一阶矩梯度之和(动量之和)的衰减率，通常设置为0.9。Beta2（ ![[公式]](https://www.zhihu.com/equation?tex=%5Crho_2) ）是二阶矩梯度平方和的衰减率，通常设置为0.999。Adam 的速度来自于动量和RMSProp 适应不同方向的梯度的能力。这两者的结合使它变得更强大。

Adam梯度经过偏置校正后，每一次迭代学习率都有一个固定范围，使得参数比较平稳。
结合了Adagrad善于处理稀疏梯度和RMSprop善于处理非平稳目标的优点
为不同的参数计算不同的自适应学习率
也适用于大多非凸优化问题——适用于大数据集和高维空间。

## 3.5 LazyAdam

LazyAdam 是 Adam 优化器的一种变体，可以更高效地处理稀疏更新。原始的 Adam 算法为每个可训练变量维护两个**移动平均累加器**，这些累加器在每一步都会更新。**而此类为稀疏变量提供了更加懒惰的梯度更新处理。它仅更新当前批次中\*出现的稀疏变量索引的移动平均累加器\*，而不是更新所有索引的累加器。**与原始的 Adam 优化器相比，它可以大幅提高某些应用的模型训练吞吐量。但是，它的语义与原始的 Adam 算法略有不同，这可能会产生不同的实验结果。

[优化方法总结以及Adam存在的问题(SGD, Momentum, AdaDelta, Adam, AdamW，LazyAdam)blog.csdn.net/yinyu19950811/article/details/90476956![img](https://pic4.zhimg.com/v2-4ff69acb9a05192f3a26b36125e2259f_180x120.jpg)](https://link.zhihu.com/?target=https%3A//blog.csdn.net/yinyu19950811/article/details/90476956)
##  四、牛顿法

牛顿法是一种在实数域和复数域上近似求解方程的方法。方法使用函数*f* (*x*)的泰勒级数的前面几项来寻找方程*f* (*x*) = 0的根。牛顿法最大的特点就在于它的收敛速度很快。

## 　　具体步骤：

　　首先，选择一个接近函数 *f* (*x*)零点的 *x*0，计算相应的 *f* (*x*0) 和切线斜率*f  '* (*x*0)（这里*f '* 表示函数 *f*  的导数）。然后我们计算穿过点(*x*0,  *f*  (*x*0)) 并且斜率为*f* '(*x*0)的直线和 *x* 轴的交点的*x*坐标，也就是求如下方程的解：

![img](https://images0.cnblogs.com/blog2015/764050/201508/222309088311820.png)

　　我们将新求得的点的 *x* 坐标命名为*x*1，通常*x*1会比*x*0更接近方程*f*  (*x*) = 0的解。因此我们现在可以利用*x*1开始下一轮迭代。迭代公式可化简为如下所示：

![img](https://images0.cnblogs.com/blog2015/764050/201508/222309221284615.png)

　　已经证明，如果*f*  ' 是连续的，并且待求的零点*x*是孤立的，那么在零点*x*周围存在一个区域，只要初始值*x*0位于这个邻近区域内，那么牛顿法必定收敛。 并且，如果*f*  ' (*x*)不为0, 那么牛顿法将具有平方收敛的性能. 粗略的说，这意味着每迭代一次，牛顿法结果的有效数字将增加一倍。下图为一个牛顿法执行过程的例子。

　　由于牛顿法是基于当前位置的切线来确定下一次的位置，所以牛顿法又被很形象地称为是"切线法"。牛顿法的搜索路径（二维情况）如下图所示：

　　牛顿法搜索动态示例图：

 ![img](https://images2017.cnblogs.com/blog/1022856/201709/1022856-20170916202719078-1588446775.gif)

**关于牛顿法和梯度下降法的效率对比：**

　　**从本质上去看，牛顿法是二阶收敛，梯度下降是一阶收敛，所以牛顿法就更快。如果更通俗地说的话，比如你想找一条最短的路径走到一个盆地的最底部，梯度下降法每次只从你当前所处位置选一个坡度最大的方向走一步，牛顿法在选择方向时，不仅会考虑坡度是否够大，还会考虑你走了一步之后，坡度是否会变得更大。所以，可以说牛顿法比梯度下降法看得更远一点，能更快地走到最底部。（牛顿法目光更加长远，所以少走弯路；相对而言，梯度下降法只考虑了局部的最优，没有全局思想。）**

　　**根据wiki上的解释，从几何上说，牛顿法就是用一个二次曲面去拟合你当前所处位置的局部曲面，而梯度下降法是用一个平面去拟合当前的局部曲面，通常情况下，二次曲面的拟合会比平面更好，所以牛顿法选择的下降路径会更符合真实的最优下降路径。**

 ![img](https://images2017.cnblogs.com/blog/1022856/201709/1022856-20170916202746985-1087770168.png)

注：红色的牛顿法的迭代路径，绿色的是梯度下降法的迭代路径。

**牛顿法的优缺点总结：**

　　**优点：二阶收敛，收敛速度快；**

　　**缺点：牛顿法是一种迭代算法，每一步都需要求解目标函数的Hessian矩阵的逆矩阵，计算比较复杂。**
##  算法的表现

下图是各个算法在等高线的表现，它们都从相同的点出发，走不同的路线达到最小值点。可以看到，Adagrad，Adadelta和RMSprop在正确的方向上很快地转移方向，并且快速地收敛，然而Momentum和NAG先被领到一个偏远的地方，然后才确定正确的方向，NAG比momentum率先更正方向。SGD则是缓缓地朝着最小值点前进。

![img](https://pic4.zhimg.com/80/v2-b271b87d441fa2ac24c3c4b72369a6b7_1440w.jpg)



![img](https://pic1.zhimg.com/v2-4a3b4a39ab8e5c556359147b882b4788_b.jpg)

存在鞍点的曲面



![img](https://pic1.zhimg.com/v2-5d5166a3d3712e7c03af74b1ccacbeac_b.jpg)



> 
