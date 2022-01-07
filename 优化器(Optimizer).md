# 优化器(Optimizer)

## 1 什么是优化器？

（1）解释

一言以蔽之，优化器就是在深度学习反向传播过程中，指引损失函数（目标函数）的各个参数往正确的方向更新合适的大小，使得更新后的各个参数让损失函数（目标函数）值不断逼近全局最小。

（2）原理解释

优化问题可以看做是我们站在山上的某个位置（当前的参数信息），想要以最佳的路线去到山下（最优点）。首先，直观的方法就是环顾四周，找到下山最快的方向走一步，然后再次环顾四周，找到最快的方向，直到下山——这样的方法便是朴素的梯度下降——当前的海拔是我们的目标函数值，而我们在每一步找到的方向便是函数梯度的反方向（梯度是函数上升最快的方向，所以梯度的反方向就是函数下降最快的方向）。

事实上，使用梯度下降进行优化，是几乎所有优化器的核心思想。当我们下山时，有两个方面是我们最关心的：

- 首先是优化方向，决定“前进的方向是否正确”，在优化器中反映为梯度或动量。
- 其次是步长，决定“每一步迈多远”，在优化器中反映为学习率。

所有优化器都在关注这两个方面，但同时也有一些其他问题，比如应该在哪里出发、路线错误如何处理……这是一些最新的优化器关注的方向。

（3）公式和定义

待优化参数： ![[公式]](https://www.zhihu.com/equation?tex=%5Comega) ，目标函数： ![[公式]](https://www.zhihu.com/equation?tex=f%28x%29) ，初始学习率： ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) ，迭代epoch： ![[公式]](https://www.zhihu.com/equation?tex=t)

参数更新步骤如下：

Ⅰ.计算目标函数关于当前参数的梯度：

> ![[公式]](https://www.zhihu.com/equation?tex=g_t%3D%5Cnabla+f%28w_t%29)

Ⅱ. 根据历史梯度计算一阶动量和二阶动量：

> ![[公式]](https://www.zhihu.com/equation?tex=m_t+%3D+%5Cphi%28g_1%2C+g_2%2C+%5Ccdots%2C+g_t%29%3B+V_t+%3D+%5Csum_%7Bi%3D0%7D%5E%7Bt%7D%7Bx_%7Bi%7D%5E%7B2%7D%7D)

Ⅲ. 计算当前时刻的下降梯度：

> ![[公式]](https://www.zhihu.com/equation?tex=%5Ceta_t+%3D+%5Calpha+%5Ccdot+m_t+%2F+%5Csqrt%7BV_t%7D)

Ⅳ. 根据下降梯度进行更新参数：

> ![[公式]](https://www.zhihu.com/equation?tex=w_%7Bt%2B1%7D+%3D+w_t+-+%5Ceta_t)



步骤Ⅲ、Ⅳ对于各个算法都是一致的，主要的差别就体现在步骤Ⅰ、Ⅱ上。

## 2 有哪些优化器？

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
>
> 

### 2.1**随机梯度下降法（Stochastic Gradient Descent，SGD）**

随机梯度下降算法每次从训练集中随机选择一个样本来进行学习，SGD没有动量的概念，因此

> ![[公式]](https://www.zhihu.com/equation?tex=m_t+%3D+g_t%3B+V_t+%3D+I%5E2)

代入步骤Ⅲ，可以得到下降梯度

> ![[公式]](https://www.zhihu.com/equation?tex=%5Ceta_t+%3D+%5Calpha+%5Ccdot+g_t+)

**SGD参数更新公式**如下，其中 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) 是学习率， ![[公式]](https://www.zhihu.com/equation?tex=g_%7Bt%7D) 是当前参数的梯度

> ![[公式]](https://www.zhihu.com/equation?tex=w_%7Bt%2B1%7D+%3D+w_t+-+%5Ceta_t%3Dw_t+-+%5Calpha+%5Ccdot+g_t+)

优点：

（1）每次只用一个样本更新模型参数，训练速度快

（2）随机梯度下降所带来的波动有利于优化的方向从当前的局部极小值点跳到另一个更好的局部极小值点，这样对于非凸函数，最终收敛于一个较好的局部极值点，甚至全局极值点。

缺点：

（1）当遇到局部最优点或鞍点时，梯度为0，无法继续更新参数

![img](https://pic1.zhimg.com/80/v2-dc7b97bdf8a712bc37b6f80bac1d2ab4_1440w.jpg)局部最优点

（2）沿陡峭方向震荡，而沿平缓维度进展缓慢，难以迅速收敛

### 2.2**SGD with Momentum**

为了抑制SGD的震荡，SGDM认为梯度下降过程可以加入惯性。下坡的时候，如果发现是陡坡，那就利用惯性跑的快一些。SGDM全称是SGD with momentum，在SGD基础上引入了一阶动量：

> ![[公式]](https://www.zhihu.com/equation?tex=m_t+%3D+%5Cbeta_1+%5Ccdot+m_%7Bt-1%7D+%2B+%281-%5Cbeta_1%29%5Ccdot+g_t)

**SGD-M参数更新公式**如下，其中 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) 是学习率， ![[公式]](https://www.zhihu.com/equation?tex=g_%7Bt%7D) 是当前参数的梯度

> ![[公式]](https://www.zhihu.com/equation?tex=w_%7Bt%2B1%7D+%3D+w_t+-%5Calpha+%5Ccdot+m_t+%3D+w_t+-%5Calpha+%5Ccdot%28%5Cbeta_1+%5Ccdot+m_%7Bt-1%7D+%2B+%281-%5Cbeta_1%29%5Ccdot+g_t%29)

一阶动量是各个时刻梯度方向的指数移动平均值，也就是说， ![[公式]](https://www.zhihu.com/equation?tex=t) 时刻的下降方向，不仅由当前点的梯度方向决定，而且由此前累积的下降方向决定。 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbeta_1) 的经验值为0.9，这就意味着下降方向主要是此前累积的下降方向，并略微偏向当前时刻的下降方向。想象高速公路上汽车转弯，在高速向前的同时略微偏向，急转弯可是要出事的。

特点：

因为加入了动量因素，SGD-M缓解了SGD在局部最优点梯度为0，无法持续更新的问题和振荡幅度过大的问题，但是并没有完全解决，当局部沟壑比较深，动量加持用完了，依然会困在局部最优里来回振荡。



### 2.3**SGD with Nesterov Acceleration**

SGD 还有一个问题是困在局部最优的沟壑里面震荡。想象一下你走到一个盆地，四周都是略高的小山，你觉得没有下坡的方向，那就只能待在这里了。可是如果你爬上高地，就会发现外面的世界还很广阔。因此，我们不能停留在当前位置去观察未来的方向，而要向前一步、多看一步、看远一些。

NAG全称Nesterov Accelerated Gradient，是在SGD、SGD-M的基础上的进一步改进，改进点在于步骤Ⅰ。我们知道在时刻 ![[公式]](https://www.zhihu.com/equation?tex=t) 的主要下降方向是由累积动量决定的，自己的梯度方向说了也不算，那与其看当前梯度方向，不如先看看如果跟着累积动量走了一步，那个时候再怎么走。因此，NAG在步骤Ⅰ，不计算当前位置的梯度方向，而是计算如果按照累积动量走了一步，那个时候的下降方向：

> ![[公式]](https://www.zhihu.com/equation?tex=g_t%3D%5Cnabla+f%28w_t-%5Calpha+%5Ccdot+m_%7Bt-1%7D+%2F+%5Csqrt%7BV_%7Bt-1%7D%7D%29)

**NAG参数更新公式**如下，其中 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) 是学习率， ![[公式]](https://www.zhihu.com/equation?tex=g_%7Bt%7D) 是当前参数的梯度

> ![[公式]](https://www.zhihu.com/equation?tex=w_%7Bt%2B1%7D+%3Dw_t+-+%5Calpha+%5Ccdot+g_t+%3Dw_t+-+%5Calpha+%5Cast%28%5Cnabla+f%28w_t-%5Calpha+%5Ccdot+m_%7Bt-1%7D+%2F+%5Csqrt%7BV_%7Bt-1%7D%7D%29%29)

然后用下一个点的梯度方向，与历史累积动量相结合，计算步骤Ⅱ中当前时刻的累积动量。

特点：

有利于跳出当前局部最优的沟壑，寻找新的最优值，但是收敛速度慢。



###  2.4**AdaGrad（自适应学习率算法）**

SGD系列的都没有用到二阶动量。二阶动量的出现，才意味着“自适应学习率”优化算法时代的到来。SGD及其变种以同样的学习率更新每个参数，但深度神经网络往往包含大量的参数，这些参数并不是总会用得到（想想大规模的embedding）。对于经常更新的参数，我们已经积累了大量关于它的知识，不希望被单个样本影响太大，希望学习速率慢一些；对于偶尔更新的参数，我们了解的信息太少，希望能从每个偶然出现的样本身上多学一些，即学习速率大一些。

怎么样去度量历史更新频率呢？

那就是二阶动量——该维度上，记录到目前为止所有梯度值的平方和：

> ![[公式]](https://www.zhihu.com/equation?tex=V_t+%3D+%5Csum_%7B%5Ctau%3D1%7D%5E%7Bt%7D+g_%5Ctau%5E2)

我们再回顾一下步骤Ⅲ中的下降梯度：

> ![[公式]](https://www.zhihu.com/equation?tex=%5Ceta_t+%3D+%5Calpha+%5Ccdot+m_t+%2F+%5Csqrt%7BV_t%7D)

**AdaGrad参数更新公式**如下，其中 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) 是学习率， ![[公式]](https://www.zhihu.com/equation?tex=g_%7Bt%7D) 是当前参数的梯度

> ![[公式]](https://www.zhihu.com/equation?tex=w_%7Bt%2B1%7D+%3Dw_t+-+%5Calpha+%5Ccdot+m_t+%2F+%5Csqrt%7BV_t%7D%3Dw_t+-+%5Calpha+%5Ccdot+m_t+%2F+%5Csqrt%7B%5Csum_%7B%5Ctau%3D1%7D%5E%7Bt%7D+g_%5Ctau%5E2%7D)

可以看出，此时实质上的学习率由 ![[公式]](https://www.zhihu.com/equation?tex=+%5Calpha) 变成了 ![[公式]](https://www.zhihu.com/equation?tex=+%5Calpha+%2F+%5Csqrt%7BV_t%7D) 。 一般为了避免分母为0，会在分母上加一个小的平滑项。因此![[公式]](https://www.zhihu.com/equation?tex=%5Csqrt%7BV_t%7D) 是恒大于0的，而且参数更新越频繁，二阶动量越大，学习率就越小。

优点：

（1）在稀疏数据场景下表现非常好

（2）此前的SGD及其变体的优化器主要聚焦在优化梯度前进的方向上，而AdaGrad首次使用二阶动量来关注学习率（步长），开启了自适应学习率算法的里程。

缺点：

（1）因为![[公式]](https://www.zhihu.com/equation?tex=%5Csqrt%7BV_t%7D) 是单调递增的，会使得学习率单调递减至0，可能会使得训练过程提前结束，即便后续还有数据也无法学到必要的知识。

### 2.5 **AdaDelta / RMSProp**

由于AdaGrad单调递减的学习率变化过于激进，考虑一个改变二阶动量计算方法的策略：不累积全部历史梯度，而只关注过去一段时间窗口的下降梯度。这也就是AdaDelta名称中Delta的来历。

修改的思路很简单。前面讲到，指数移动平均值大约就是过去一段时间的平均值，因此我们用这一方法来计算二阶累积动量：

> ![[公式]](https://www.zhihu.com/equation?tex=V_t+%3D+%5Cbeta_2+%5Ccdot+V_%7Bt-1%7D+%2B+%281-%5Cbeta_2%29+g_t%5E2)

**AdaDelta / RMSProp参数更新公式**如下，其中 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) 是学习率， ![[公式]](https://www.zhihu.com/equation?tex=g_%7Bt%7D) 是当前参数的梯度

> ![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7Dw_%7Bt%2B1%7D+%26%3Dw_t+-+%5Calpha+%5Ccdot+m_t+%2F+%5Csqrt%7BV_t%7D%5C%5C%26%3Dw_t+-+%5Calpha+%5Ccdot+m_t+%2F+%5Csqrt%7B+%5Cbeta_2+%5Ccdot+V_%7Bt-1%7D+%2B+%281-%5Cbeta_2%29+g_t%5E2%7D+%5Cend%7Balign%7D)



优点：

避免了二阶动量持续累积、导致训练过程提前结束的问题了。



### 2.6**Adam**

谈到这里，Adam和Nadam的出现就很自然而然了——它们是前述方法的集大成者。SGD-M在SGD基础上增加了一阶动量，AdaGrad和AdaDelta在SGD基础上增加了二阶动量。把一阶动量和二阶动量都用起来，就是Adam了——Adaptive + Momentum。

SGD的一阶动量：

![[公式]](https://www.zhihu.com/equation?tex=m_t+%3D+%5Cbeta_1+%5Ccdot+m_%7Bt-1%7D+%2B+%281-%5Cbeta_1%29%5Ccdot+g_t)

加上AdaDelta的二阶动量：

![[公式]](https://www.zhihu.com/equation?tex=V_t+%3D+%5Cbeta_2+%5Ccdot+V_%7Bt-1%7D+%2B+%281-%5Cbeta_2%29+g_t%5E2)

**Adam参数更新公式**如下，其中 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) 是学习率， ![[公式]](https://www.zhihu.com/equation?tex=g_%7Bt%7D) 是当前参数的梯度

> ![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7D++w_%7Bt%2B1%7D++%26%3Dw_t+-+%5Calpha+%5Ccdot+m_t+%2F+%5Csqrt%7BV_t%7D%5C%5C%26%3Dw_t+-+%5Calpha+%5Ccdot+%28%5Cbeta_1+%5Ccdot+m_%7Bt-1%7D+%2B+%281-%5Cbeta_1%29%5Ccdot+g_t%29+%2F+%5Csqrt%7B+%5Cbeta_2+%5Ccdot+V_%7Bt-1%7D+%2B+%281-%5Cbeta_2%29+g_t%5E2%7D+%5Cend%7Balign%7D)



优化算法里最常见的两个超参数 ![[公式]](https://www.zhihu.com/equation?tex=+%5Cbeta_1%2C+%5Cbeta_2) 就都在这里了，前者控制一阶动量，后者控制二阶动量。

优点：

（1）通过一阶动量和二阶动量，有效控制学习率步长和梯度方向，防止梯度的振荡和在鞍点的静止。

缺点：

（1）**可能不收敛**：

二阶动量是固定时间窗口内的累积，随着时间窗口的变化，遇到的数据可能发生巨变，使得![[公式]](https://www.zhihu.com/equation?tex=V_t)可能会时大时小，不是单调变化。这就可能在训练后期引起学习率的震荡，导致模型无法收敛。

修正的方法。由于Adam中的学习率主要是由二阶动量控制的，为了保证算法的收敛，可以对二阶动量的变化进行控制，避免上下波动。

![[公式]](https://www.zhihu.com/equation?tex=V_t+%3D+max%28%5Cbeta_2+%2A+V_%7Bt-1%7D+%2B+%281-%5Cbeta_2%29+g_t%5E2%2C+V_%7Bt-1%7D%29)

通过这样修改，就保证了 ![[公式]](https://www.zhihu.com/equation?tex=%7C%7CV_t%7C%7C+%5Cgeq+%7C%7CV_%7Bt-1%7D%7C%7C) ，从而使得学习率单调递减。


（2）**可能错过全局最优解：**

自适应学习率算法可能会对前期出现的特征过拟合，后期才出现的特征很难纠正前期的拟合效果。后期Adam的学习率太低，影响了有效的收敛。

###  2.7 Nadam

最后是Nadam。我们说Adam是集大成者，但它居然遗漏了Nesterov，这还能忍？必须给它加上，按照NAG的步骤：

> ![[公式]](https://www.zhihu.com/equation?tex=g_t%3D%5Cnabla+f%28w_t-%5Calpha+%5Ccdot+m_%7Bt-1%7D+%2F+%5Csqrt%7BV_t%7D%29)

这就是Nesterov + Adam = Nadam了。



### **2.8 总结**

**SGD参数更新公式**如下，其中 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) 是学习率， ![[公式]](https://www.zhihu.com/equation?tex=g_%7Bt%7D) 是当前参数的梯度

> ![[公式]](https://www.zhihu.com/equation?tex=w_%7Bt%2B1%7D+%3Dw_t+-+%5Calpha+%5Ccdot+g_t+)

优化器千变万化，五花八门，其实主要还是在步长（ ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) ）和梯度方向（ ![[公式]](https://www.zhihu.com/equation?tex=g_%7Bt%7D) ）两个层面进行改进，都是SGD带不同的learning rate scheduler。



## 3 优化算法的选择与使用策略

优化算法的常用tricks

（1）首先，各大算法孰优孰劣并无定论。如果是刚入门，优先考虑SGD+Nesterov Momentum或者Adam。

（2）选择你熟悉的算法——这样你可以更加熟练地利用你的经验进行调参。

（3）充分了解你的数据——如果模型是非常稀疏的，那么优先考虑自适应学习率的算法。

（4）根据你的需求来选择——在模型设计实验过程中，要快速验证新模型的效果，可以先用Adam进行快速实验优化；在模型上线或者结果发布前，可以用精调的SGD进行模型的极致优化。

（5）先用小数据集进行实验。有论文研究指出，随机梯度下降算法的收敛速度和数据集的大小的关系不大。因此可以先用一个具有代表性的小数据集进行实验，测试一下最好的优化算法，并通过参数搜索来寻找最优的训练参数。

（6）考虑不同算法的组合**。**先用Adam进行快速下降，而后再换到SGD进行充分的调优。切换策略可以参考本文介绍的方法。

（7）数据集一定要充分的打散（shuffle）**。**这样在使用自适应学习率算法的时候，可以避免某些特征集中出现，而导致的有时学习过度、有时学习不足，使得下降方向出现偏差的问题。

（8）训练过程中持续监控训练数据和验证数据上的目标函数值以及精度或者AUC等指标的变化情况。对训练数据的监控是要保证模型进行了充分的训练——下降方向正确，且学习率足够高；对验证数据的监控是为了避免出现过拟合。

（9）制定一个合适的学习率衰减策略**。**可以使用定期衰减策略，比如每过多少个epoch就衰减一次；或者利用精度或者AUC等性能指标来监控，当测试集上的指标不变或者下跌时，就降低学习率。