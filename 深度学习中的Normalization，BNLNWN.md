# 深度学习中的Normalization，BN/LN/WN

深度神经网络模型训练之难众所周知，其中一个重要的现象就是 Internal Covariate Shift. Batch Norm 大法自 2015 年由Google 提出之后，就成为深度学习必备之神器。自 BN 之后， Layer Norm / Weight Norm / Cosine Norm 等也横空出世。本文从 Normalization 的背景讲起，用一个公式概括 Normalization 的基本思想与通用框架，将各大主流方法一一对号入座进行深入的对比分析，并从参数和数据的伸缩不变性的角度探讨 Normalization 有效的深层原因。

## **1. 为什么需要 Normalization**

### **1.1 独立同分布与白化**



机器学习界的炼丹师们最喜欢的数据有什么特点？窃以为，莫过于“**独立同分布**”了，即*independent and identically distributed*，简称为 *i.i.d.* 独立同分布并非所有机器学习模型的必然要求（比如 Naive Bayes 模型就建立在特征彼此独立的基础之上，而Logistic Regression 和 神经网络 则在非独立的特征数据上依然可以训练出很好的模型），但独立同分布的数据可以简化常规机器学习模型的训练、提升机器学习模型的预测能力，已经是一个共识。

**独立同分布**：*独立：随机变量的“每一种可能”之间相互不影响；例如丢筛子，第一次丢的结果不会影响第二次的结果。

同分布：第一次丢筛子和第二次丢，得到任意一面的概率都是分别相同的，比如第一次丢到“一点数”的概率为1/6，第二次丢到“一点数”的概率也会是1/6，都有着相同的概率密度函数和累计分布函数，也就是说的同分布。*



因此，在把数据喂给机器学习模型之前，“**白化（whitening）**”是一个重要的数据预处理步骤。白化一般包含两个目的：

（1）*去除特征之间的相关性* —> 独立；

（2）*使得所有特征具有相同的均值和方差* —> 同分布。

白化最典型的方法就是PCA。

### **1.2 深度学习中的 Internal Covariate Shift**



深度神经网络模型的训练为什么会很困难？其中一个重要的原因是，深度神经网络涉及到很多层的叠加，而每一层的参数更新会导致上层的输入数据分布发生变化，通过层层叠加，高层的输入分布变化会非常剧烈，这就使得高层需要不断去重新适应底层的参数更新。为了训好模型，我们需要非常谨慎地去设定学习率、初始化权重、以及尽可能细致的参数更新策略。



Google 将这一现象总结为 Internal Covariate Shift，简称 ICS. 什么是 ICS 呢？

> 大家都知道在统计机器学习中的一个经典假设是“源空间（source domain）和目标空间（target domain）的数据分布（distribution）是一致的”。如果不一致，那么就出现了新的机器学习问题，如 transfer learning / domain adaptation 等。而 covariate shift 就是分布不一致假设之下的一个分支问题，它是指源空间和目标空间的条件概率是一致的，但是其边缘概率不同，即：对所有![[公式]](https://www.zhihu.com/equation?tex=x%5Cin+%5Cmathcal%7BX%7D),![[公式]](https://www.zhihu.com/equation?tex=P_s%28Y%7CX%3Dx%29%3DP_t%28Y%7CX%3Dx%29%5C%5C)但是![[公式]](https://www.zhihu.com/equation?tex=P_s%28X%29%5Cne+P_t%28X%29%5C%5C)大家细想便会发现，的确，对于神经网络的各层输出，由于它们经过了层内操作作用，其分布显然与各层对应的输入信号分布不同，而且差异会随着网络深度增大而增大，可是它们所能“指示”的样本标记（label）仍然是不变的，这便符合了covariate shift的定义。由于是对层间信号的分析，也即是“internal”的来由。



### **1.3 ICS 会导致什么问题？**



简而言之，每个神经元的输入数据不再是“独立同分布”。

其一，上层参数需要不断适应新的输入数据分布，降低学习速度。

其二，下层输入的变化可能趋向于变大或者变小，导致上层落入饱和区，使得学习过早停止。

其三，每层的更新都会影响到其它层，因此每层的参数更新策略需要尽可能的谨慎。

## **2. Normalization 的通用框架与基本思想**



我们以神经网络中的一个普通神经元为例。神经元接收一组输入向量

 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbold%7Bx%7D%3D%28x_1%2C+x_2%2C+%5Ccdots%2C+x_d%29%5C%5C) 

通过某种运算后，输出一个标量值：

![[公式]](https://www.zhihu.com/equation?tex=y%3Df%28%5Cbold%7Bx%7D%29%5C%5C)

由于 ICS 问题的存在， ![[公式]](https://www.zhihu.com/equation?tex=%5Cbold%7Bx%7D) 的分布可能相差很大。要解决独立同分布的问题，“理论正确”的方法就是对每一层的数据都进行白化操作。然而标准的白化操作代价高昂，特别是我们还希望白化操作是可微的，保证白化操作可以通过反向传播来更新梯度。



因此，以 BN 为代表的 Normalization 方法退而求其次，进行了简化的白化操作。基本思想是：在将 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbold%7Bx%7D) 送给神经元之前，先对其做**平移和伸缩变换**， 将 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbold%7Bx%7D) 的分布规范化成在固定区间范围的标准分布。



通用变换框架就如下所示：

![[公式]](https://www.zhihu.com/equation?tex=h%3Df%5Cleft%28%5Cbold%7Bg%7D%5Ccdot%5Cfrac%7B%5Cbold%7Bx%7D-%5Cbold%7B%5Cmu%7D%7D%7B%5Cbold%7B%5Csigma%7D%7D%2B%5Cbold%7Bb%7D%5Cright%29%5C%5C)

我们来看看这个公式中的各个参数。



（1） ![[公式]](https://www.zhihu.com/equation?tex=%5Cbold%7B%5Cmu%7D) 是**平移参数**（shift parameter）， ![[公式]](https://www.zhihu.com/equation?tex=%5Cbold%7B%5Csigma%7D) 是**缩放参数**（scale parameter）。通过这两个参数进行 shift 和 scale 变换：

 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbold%7B%5Chat%7Bx%7D%7D%3D%5Cfrac%7B%5Cbold%7Bx%7D-%5Cbold%7B%5Cmu%7D%7D%7B%5Cbold%7B%5Csigma%7D%7D%5C%5C) 

得到的数据符合均值为 0、方差为 1 的标准分布。



（2） ![[公式]](https://www.zhihu.com/equation?tex=%5Cbold%7Bb%7D) 是**再平移参数**（re-shift parameter）， ![[公式]](https://www.zhihu.com/equation?tex=%5Cbold%7Bg%7D) 是**再缩放参数**（re-scale parameter）。将 上一步得到的 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbold%7B%5Chat%7Bx%7D%7D) 进一步变换为： ![[公式]](https://www.zhihu.com/equation?tex=%5Cbold%7By%7D%3D%5Cbold%7Bg%7D%5Ccdot+%5Cbold%7B%5Chat%7Bx%7D%7D+%2B+%5Cbold%7Bb%7D%5C%5C)

最终得到的数据符合均值为 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbold%7Bb%7D) 、方差为 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbold%7Bg%7D%5E2) 的分布。



说好的处理 ICS，第一步都已经得到了标准分布，第二步怎么又给变走了？



答案是——**为了保证模型的表达能力不因为规范化而下降**。



我们可以看到，第一步的变换将输入数据限制到了一个全局统一的确定范围（均值为 0、方差为 1）。下层神经元可能很努力地在学习，但不论其如何变化，其输出的结果在交给上层神经元进行处理之前，将被粗暴地重新调整到这一固定范围。

所以，为了尊重底层神经网络的学习结果，我们将规范化后的数据进行再平移和再缩放，使得每个神经元对应的输入范围是针对该神经元量身定制的一个确定范围（均值为 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbold%7Bb%7D) 、方差为 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbold%7Bg%7D%5E2) ）。rescale 和 reshift 的参数都是可学习的，这就使得 Normalization 层可以学习如何去尊重底层的学习结果。



除了充分利用底层学习的能力，另一方面的重要意义在于保证获得非线性的表达能力。Sigmoid 等激活函数在神经网络中有着重要作用，通过区分饱和区和非饱和区，使得神经网络的数据变换具有了非线性计算能力。而第一步的规范化会将几乎所有数据映射到激活函数的非饱和区（线性区），仅利用到了线性变化能力，从而降低了神经网络的表达能力。而进行再变换，则可以将数据从线性区变换到非线性区，恢复模型的表达能力。



那么问题又来了——

**经过这么的变回来再变过去，会不会跟没变一样？**



不会。因为，再变换引入的两个新参数 g 和 b，可以表示旧参数作为输入的同一族函数，但是新参数有不同的学习动态。在旧参数中， ![[公式]](https://www.zhihu.com/equation?tex=%5Cbold%7Bx%7D) 的均值取决于下层神经网络的复杂关联；但在新参数中， ![[公式]](https://www.zhihu.com/equation?tex=%5Cbold%7By%7D%3D%5Cbold%7Bg%7D%5Ccdot+%5Cbold%7B%5Chat%7Bx%7D%7D+%2B+%5Cbold%7Bb%7D) 仅由 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbold%7Bb%7D) 来确定，去除了与下层计算的密切耦合。新参数很容易通过梯度下降来学习，简化了神经网络的训练。



那么还有一个问题——

**这样的 Normalization 离标准的白化还有多远？**

标准白化操作的目的是“独立同分布”。独立就不说了，暂不考虑。变换为均值为 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbold%7Bb%7D) 、方差为 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbold%7Bg%7D%5E2) 的分布，也并不是严格的同分布，只是映射到了一个确定的区间范围而已。（所以，这个坑还有得研究呢！）

## **3. 主流 Normalization 方法梳理**

### 3.1BN

![img](https://pic2.zhimg.com/80/v2-27073f09cecc1c19c6aab74896ae6d61_1440w.jpg)

![img](https://pic1.zhimg.com/80/v2-4c0aadae51d1e7b73d8b396a087e0e7c_1440w.jpg)

**BN的作用**：

1. 加快网络的训练和收敛的速度
2. 控制梯度爆炸防止梯度消失
3. 防止过拟合

- 分析：
- （1）加快收敛速度：在深度神经网络中中，如果每层的数据分布都不一样的话，将会导致网络非常难收敛和训练，而如果把 每层的数据都在转换在均值为零，方差为1 的状态下，这样每层数据的分布都是一样的训练会比较容易收敛。
- （2）防止梯度爆炸和梯度消失：

![img](https://pic4.zhimg.com/80/v2-268295f5f0ece3fd338ccd787c14e373_1440w.jpg)

以sigmoid函数为例，sigmoid函数使得输出在[0,1]之间，实际上当x道了一定的大小，经过sigmoid函数后输出范围就会变得很小

![img](https://pic4.zhimg.com/80/v2-7dfc599156734692d460fcff842052e7_1440w.jpg)

梯度消失：在深度神经网络中，如果网络的激活输出很大，其对应的梯度就会很小，导致网络的学习速率就会很慢，假设网络中每层的学习梯度都小于最大值0.25，网络中有n层，因为链式求导的原因，第一层的梯度将会小于0.25的n次方，所以学习速率相对来说会变的很慢，而对于网络的最后一层只需要对自身求导一次，梯度就大，学习速率就会比较快，这就会造成在一个很深的网络中，浅层基本不学习，权值变化小，而后面几层网络一直学习，后面的网络基本可以表征整个网络，这样失去了深度的意义。（使用BN层归一化后，网络的输出就不会很大，梯度就不会很小）

梯度爆炸：第一层偏移量的梯度=激活层斜率1x权值1x激活层斜率2x…激活层斜率(n-1)x权值(n-1)x激活层斜率n，假如激活层斜率均为最大值0.25，所有层的权值为100，这样梯度就会指数增加。（使用bn层后权值的更新也不会很大）

（3）BN算法防止过拟合：在网络的训练中，BN的使用使得一个minibatch中所有样本都被关联在了一起，因此网络不会从某一个训练样本中生成确定的结果，即同样一个样本的输出不再仅仅取决于样本的本身，也取决于跟这个样本同属一个batch的其他样本，而每次网络都是随机取batch，这样就会使得整个网络不会朝这一个方向使劲学习。一定程度上避免了过拟合。

**BN的缺陷**：

1、高度依赖于mini-batch的大小，实际使用中会对mini-Batch大小进行约束，不适合类似在线学习（mini-batch为1）。

2、不适用于RNN网络中normalize操作：BN实际使用时需要计算并且保存某一层神经网络mini-batch的均值和方差等统计信息，对于对一个固定深度的前向神经网络（DNN，CNN）使用BN，很方便；但对于RNN来说，sequence的长度是不一致的，换句话说RNN的深度不是固定的，不同的time-step需要保存不同的statics特征，可能存在一个特殊sequence比其的sequence长很多，这样training时，计算很麻烦。

### 3.2其他

**LN，IN，GN就是为了解决该问题而提出的。**

Batch Normalization 的处理对象是对一批样本，

Layer Normalization 的处理对象是单个样本。

Batch Normalization 是对这批样本的同一维度特征做归一化，

Layer Normalization 是对这单个样本的所有维度特征做归一化。

![img](https://pic1.zhimg.com/v2-0ac1060e38f8ce8914d6a600bd63f854_r.jpg)

BatchNorm这类归一化技术，**目的就是让每一层的分布稳定下来**，让后面的层可以在前面层的基础上安心学习知识。

BatchNorm就是通过对batch size这个维度归一化来让分布稳定下来。

LayerNorm则是通过对Hidden size这个维度归一化来让分布稳定下来。

先来张图直观感受下BN，LN，IN，GN的区别与联系：

![img](https://pic2.zhimg.com/80/v2-66b2a13334967dc27025e354bb448875_1440w.jpg)

这张图与我们平常看到的feature maps有些不同，立方体的3个维度为别为batch/ channel/ HW，而我们常见的feature maps中，3个维度分别为channel/ H/ W，没有batch。分析上图可知：BN计算均值和标准差时，固定channel(在一个channel内)，对HW和batch作平均；LN计算均值和标准差时，固定batch(在一个batch内)，对HW和channel作平均；IN计算均值和标准差时，同时固定channel和batch(在一个batch内中的一个channel内)，对HW作平均；GN计算均值和标准差时，固定batch且对channel作分组(在一个batch内对channel作分组)，在分组内对HW作平均。更精确的公式描述请大家自行看原论文[Group Normalization](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1803.08494)吧。



#### 2. Layer Normalization

LN的方法是对于每一个sample中的多个feature(也就是channel)进行归一化操作。把图像的尺寸表示为[N, C, H, W]的话，LN则是对于[C,H,W]进行归一化。相对于BN中所表示的同一个feature在不同的batch之间拥有同样的均值和方差。LN中所表示的则是在同一个sample中，不同的feature上有着相同的均值和方差。

与BN相比，LN也不依赖于mini-batch size的大小。这种操作通常运用在RNN的网络中。

#### 3.instance normalization

IN是针对于不同的batch, 不同的chennel进行归一化。还是把图像的尺寸表示为[N, C, H, W]的话，IN则是针对于[H,W]进行归一化。这种方式通常会用在风格迁移的训练中。

#### 4 Group Nomalization

GN是介乎于instance normal 和 layer normal 之间的一种归一化方式。也就是说当我们把所有的channel都放到同一个group中的时候就变成了layer normal， 如果我们把每个channel都归为一个不同的group，则变成了instance normal.

GN同样可以针对于mini batch size较小的情况。因为它有不受batch size的约束。

可以看到与BN不同，LN/IN和GN都没有对batch作平均，所以当batch变化时，网络的错误率不会有明显变化。但论文的实验显示：LN和IN 在时间序列模型(RNN/LSTM)和生成模型(GAN)上有很好的效果，而GN在视觉模型上表现更好。

## **4. Normalization 为什么会有效？**



我们以下面这个简化的神经网络为例来分析。

![img](https://pic1.zhimg.com/80/v2-2d76cc93f615b3401197a16bc2183e88_1440w.jpg)

**4.1 Normalization 的权重伸缩不变性**



**权重伸缩不变性（weight scale invariance）**指的是，当权重 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbold%7BW%7D) 按照常量 ![[公式]](https://www.zhihu.com/equation?tex=%5Clambda) 进行伸缩时，得到的规范化后的值保持不变，即：![[公式]](https://www.zhihu.com/equation?tex=Norm%28%5Cbold%7BW%27%7D%5Cbold%7Bx%7D%29%3DNorm%28%5Cbold%7BW%7D%5Cbold%7Bx%7D%29%5C%5C)

其中 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbold%7BW%27%7D%3D%5Clambda%5Cbold%7BW%7D) 。



**上述规范化方法均有这一性质**，这是因为，当权重 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbold%7BW%7D) 伸缩时，对应的均值和标准差均等比例伸缩，分子分母相抵。

![[公式]](https://www.zhihu.com/equation?tex=Norm%28%5Cbold%7BW%27x%7D%29%3DNorm%5Cleft%28%5Cbold%7Bg%7D%5Ccdot%5Cfrac%7B%5Cbold%7B%5Cbold%7BW%27x%7D%7D-%5Cbold%7B%5Cmu%27%7D%7D%7B%5Cbold%7B%5Csigma%27%7D%7D%2B%5Cbold%7Bb%7D%5Cright%29%5C%5C%3DNorm%5Cleft%28%5Cbold%7Bg%7D%5Ccdot%5Cfrac%7B%5Clambda%5Cbold%7BWx%7D-%5Clambda%5Cbold%7B%5Cmu%7D%7D%7B%5Clambda%5Cbold%7B%5Csigma%7D%7D%2B%5Cbold%7Bb%7D%5Cright%29%5C%5C%3DNorm%5Cleft%28%5Cbold%7Bg%7D%5Ccdot%5Cfrac%7B%5Cbold%7BWx%7D-%5Cbold%7B%5Cmu%7D%7D%7B%5Cbold%7B%5Csigma%7D%7D%2B%5Cbold%7Bb%7D%5Cright%29%3DNorm%28%5Cbold%7BWx%7D%29%5C%5C)

**权重伸缩不变性可以有效地提高反向传播的效率**。

由于

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+Norm%28%5Cbold%7BW%27x%7D%29%7D%7B%5Cpartial+%5Cbold%7Bx%7D%7D+%3D+%5Cfrac%7B%5Cpartial+Norm%28%5Cbold%7BWx%7D%29%7D%7B%5Cpartial+%5Cbold%7Bx%7D%7D+%5C%5C)

因此，权重的伸缩变化不会影响反向梯度的 Jacobian 矩阵，因此也就对反向传播没有影响，避免了反向传播时因为权重过大或过小导致的梯度消失或梯度爆炸问题，从而加速了神经网络的训练。



**权重伸缩不变性还具有参数正则化的效果，可以使用更高的学习率。**

由于 ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+Norm%28%5Cbold%7BW%27x%7D%29%7D%7B%5Cpartial+%5Cbold%7BW%27%7D%7D+%3D%5Cfrac%7B1%7D%7B+%5Clambda%7D%5Ccdot+%5Cfrac%7B%5Cpartial+Norm%28%5Cbold%7BWx%7D%29%7D%7B%5Cpartial%5Cbold%7BW%7D%7D+%5C%5C)

因此，下层的权重值越大，其梯度就越小。这样，参数的变化就越稳定，相当于实现了参数正则化的效果，避免参数的大幅震荡，提高网络的泛化性能。



**4.2 Normalization 的数据伸缩不变性**



**数据伸缩不变性（data scale invariance）**指的是，当数据 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbold%7Bx%7D) 按照常量 ![[公式]](https://www.zhihu.com/equation?tex=%5Clambda) 进行伸缩时，得到的规范化后的值保持不变，即：![[公式]](https://www.zhihu.com/equation?tex=Norm%28%5Cbold%7BW%7D%5Cbold%7Bx%27%7D%29%3DNorm%28%5Cbold%7BW%7D%5Cbold%7Bx%7D%29%5C%5C)

其中 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbold%7Bx%27%7D%3D%5Clambda%5Cbold%7Bx%7D) 。



**数据伸缩不变性仅对 BN、LN 和 CN 成立。**因为这三者对输入数据进行规范化，因此当数据进行常量伸缩时，其均值和方差都会相应变化，分子分母互相抵消。而 WN 不具有这一性质。



**数据伸缩不变性可以有效地减少梯度弥散，简化对学习率的选择**。

对于某一层神经元 ![[公式]](https://www.zhihu.com/equation?tex=h_l%3Df_%7B%5Cbold%7BW%7D_l%7D%28%5Cbold%7Bx%7D_l%29) 而言，展开可得

![[公式]](https://www.zhihu.com/equation?tex=h_l%3Df_%7B%5Cbold%7BW%7D_l%7D%28%5Cbold%7Bx%7D_l%29%3Df_%7B%5Cbold%7BW%7D_l%7D%28f_%7B%5Cbold%7BW%7D_%7Bl-1%7D%7D%28%5Cbold%7Bx%7D_%7Bl-1%7D%29%29%3D%5Ccdots%3D%5Cbold%7Bx%7D_0%5Cprod_%7Bk%3D0%7D%5El%5Cbold%7BW%7D_k%5C%5C)

每一层神经元的输出依赖于底下各层的计算结果。如果没有正则化，当下层输入发生伸缩变化时，经过层层传递，可能会导致数据发生剧烈的膨胀或者弥散，从而也导致了反向计算时的梯度爆炸或梯度弥散。



加入 Normalization 之后，不论底层的数据如何变化，对于某一层神经元 ![[公式]](https://www.zhihu.com/equation?tex=h_l%3Df_%7B%5Cbold%7BW%7D_l%7D%28%5Cbold%7Bx%7D_l%29) 而言，其输入 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbold%7Bx%7D_l) 永远保持标准的分布，这就使得高层的训练更加简单。从梯度的计算公式来看：

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+Norm%28%5Cbold%7BWx%27%7D%29%7D%7B%5Cpartial+%5Cbold%7BW%7D%7D+%3D+%5Cfrac%7B%5Cpartial+Norm%28%5Cbold%7BWx%7D%29%7D%7B%5Cpartial%5Cbold%7BW%7D%7D+%5C%5C)

数据的伸缩变化也不会影响到对该层的权重参数更新，使得训练过程更加鲁棒，简化了对学习率的选择。
