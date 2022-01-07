# 线性回归、岭回归和Lasso回归

## **一、线性回归**

**一、线性回归**

 假设有数据有 ![[公式]](https://www.zhihu.com/equation?tex=T%3D%5Cleft+%5C%7B+%5Cleft+%28+x%5E%7B%281%29%7D%2Cy%5E%7B%281%29%7D+%5Cright+%29+%2C...%2C%5Cleft+%28+x%5E%7B%28i%29%7D%2Cy%5E%7B%28i%29%7D+%5Cright+%29+%2C...%2C+%5Cleft+%28+x%5E%7B%28m%29%7D%2Cy%5E%7B%28m%29%7D+%5Cright+%29+%5Cright+%5C%7D) ，其中 ![[公式]](https://www.zhihu.com/equation?tex=x%5E%7B%28i%29%7D%3D%5Cleft+%5C%7B+x_%7B1%7D%5E%7B%28i%29%7D%2C..%2Cx_%7Bj%7D%5E%7B%28i%29%7D%2C...%2Cx_%7Bn%7D%5E%7B%28i%29%7D+%5Cright+%5C%7D) , ![[公式]](https://www.zhihu.com/equation?tex=y%5E%7Bi%7D%5Cin+%5Cmathbf%7BR%7D) 。其中m为训练集样本数，n为样本维度，y是样本的真实值。线性回归采用一个多维的线性函数来尽可能的拟合所有的数据点，最简单的想法就是最小化函数值与真实值误差的平方（概率解释-高斯分布加最大似然估计）。即有如下目标函数：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bequation%7D++%5Cbegin%7Bsplit%7D+J%5Cleft+%28+%5Ctheta+%5Cright+%29%26%3D%5Cfrac%7B1%7D%7B2%7D%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%5Cleft+%28+h_%7B%5Ctheta+%7D+%28+x%5E%7B%28i%29%7D%29-y%5E%7B%28i%29%7D+%5Cright+%29%5E%7B2%7D%5C%5C+%26%5Cmin_%7B%5Ctheta+%7DJ%5Cleft+%28+%5Ctheta+%5Cright+%29++%5Cend%7Bsplit%7D+%5Cend%7Bequation%7D)

其中线性函数如下：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bequation%7D++%5Cbegin%7Bsplit%7D+h_%7B%5Ctheta+%7D%5Cleft+%28+x%5E%7B%28i%29%7D%5Cright+%29%26%3D%5Ctheta+_%7B0%7D+%2B+%5Ctheta+_%7B1%7Dx_%7B1%7D%5E%7B%5Cleft+%28i++%5Cright+%29%7D%2B+%5Ctheta+_%7B2%7Dx_%7B2%7D%5E%7B%5Cleft+%28i++%5Cright+%29%7D%2B..%2B+%5Ctheta+_%7Bn%7Dx_%7Bn%7D%5E%7B%5Cleft+%28i++%5Cright+%29%7D%5C%5C+%26%3D%5Csum_%7Bj%3D1%7D%5E%7Bn%7D%5Ctheta+_%7Bj%7Dx_%7Bj%7D%5E%7B%5Cleft+%28i++%5Cright+%29%7D%5C%5C+%26%3D%5Cmathbf%7B%5Ctheta%7D%5E%7BT%7D+%5Cmathbf%7Bx%7D%5E%7B%28i%29%7D++%5Cend%7Bsplit%7D+%5Cend%7Bequation%7D)

 构建好线性回归模型的目标函数之后，接下来就是求解目标函数的最优解，即一个优化问题。常用的梯度优化方法都可以拿来用，这里以梯度下降法来求解目标函数。

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bequation%7D++%5Cbegin%7Bsplit%7D+%5Ctheta+_%7Bj%7D%EF%BC%9A%26%3D%5Ctheta+_%7Bj%7D-%5Calpha+%5Cfrac%7B%5Cpartial+%7D%7B%5Cpartial%5Ctheta+_%7Bj%7D%7DJ%28%5Ctheta+%29%5C%5C+%26%3D%5Ctheta+_%7Bj%7D-%5Calpha+%5Cfrac%7B%5Cpartial+%7D%7B%5Cpartial%5Ctheta+_%7Bj%7D%7D%5Cfrac%7B1%7D%7B2%7D%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%5Cleft+%28h_%7B%5Ctheta+%7D%5Cleft+%28+x%5E%7B%28i%29%7D%5Cright+%29-y%5E%7B%28i%29%7D++%5Cright+%29%5E2%5C%5C+%26%3D%5Ctheta+_%7Bj%7D-%5Calpha+%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%5Cleft+%28h_%7B%5Ctheta+%7D%5Cleft+%28+x%5E%7B%28i%29%7D%5Cright+%29-y%5E%7B%28i%29%7D++%5Cright+%29%5Cfrac%7B%5Cpartial+%7D%7B%5Cpartial%5Ctheta+_%7Bj%7D%7D%5C%5C+%26%3D%5Ctheta+_%7Bj%7D-%5Calpha+%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%5Cleft+%28h_%7B%5Ctheta+%7D%5Cleft+%28+x%5E%7B%28i%29%7D%5Cright+%29-y%5E%7B%28i%29%7D++%5Cright+%29x_%7Bj%7D%5E%7B%28i%29%7D++%5Cend%7Bsplit%7D+%5Cend%7Bequation%7D)

另外，线性回归也可以从最小二乘法的角度来看，下面先将样本表示向量化， ![[公式]](https://www.zhihu.com/equation?tex=X%5Cin+R%5E%7Bn+%5Ctimes+m%7D) ， ![[公式]](https://www.zhihu.com/equation?tex=Y+%5Cin+R%5E%7Bm%7D) ，构成如下数据矩阵。

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bequation%7D++%5Cbegin%7Bsplit%7D+%5Cbegin%7Bbmatrix%7D+-+%26+%28x%5E%7B1%7D%EF%BC%8Cy%5E%7B1%7D%29%5E%7BT%7D+%26+-%5C%5C+++-%26+%28x%5E%7B2%7D%EF%BC%8Cy%5E%7B2%7D%29%5E%7BT%7D+%26+-%5C%5C+++-%26+.+%26+-%5C%5C+++-%26+%28x%5E%7Bm%7D%EF%BC%8Cy%5E%7Bm%7D%29%5E%7BT%7D+%26-++%5Cend%7Bbmatrix%7D_%7B%28n%2B1%29%5Ctimes+m%7D++%5Cend%7Bsplit%7D+%5Cend%7Bequation%7D)

那么目标函数向量化形式如下：

![[公式]](https://www.zhihu.com/equation?tex=J%28%5Ctheta+%29%3D%5Cfrac%7B1%7D%7B2%7D%5Cleft+%28+%5Ctheta+%5E%7BT%7DX-y%5E%7BT%7D+%5Cright+%29%5Cleft+%28+%5Ctheta+%5E%7BT%7DX-y%5E%7BT%7D+%5Cright+%29%5E%7BT%7D)

可以看出目标函数是一个凸二次规划问题，其最优解在导数为0处取到，矩阵导数详细参考。

![[公式]](https://www.zhihu.com/equation?tex=%5Ctriangledown_%7B%5Ctheta+%7D+J%28%5Ctheta+%29%3DXX%5E%7BT%7D-XY+%3D0%5C%5C+%5CRightarrow+%5Ctheta+%3D%5Cleft+%28XX%5E%7BT%7D++%5Cright+%29%5E%7B-1%7DXY)

值得注意的上式中存在计算矩阵的逆，一般来讲当样本数大于数据维度时，矩阵可逆，可以采用最小二乘法求得目标函数的闭式解。当数据维度大于样本数时，矩阵线性相关，不可逆。此时最小化目标函数解不唯一，且非常多，出于这样一种情况，我们可以考虑奥卡姆剃刀准则来简化模型复杂度，使其不必要的特征对应的 ![[公式]](https://www.zhihu.com/equation?tex=w) 为0，可以考虑![[公式]](https://www.zhihu.com/equation?tex=0) 范数使得模型中 ![[公式]](https://www.zhihu.com/equation?tex=w) 非0个数最少（实际上采用的是 ![[公式]](https://www.zhihu.com/equation?tex=0) 范数的一个凸近似）。***当然，岭回归，lasso回归的最根本的目的不是解决不可逆问题，而是防止过拟合。***

***概率解释***

 损失函数与最小二乘法采用最小化平方和的概率解释。假设模型预测值与真实值的误差为 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon%5E%7B%28i%29%7D+) ，那么预测值 ![[公式]](https://www.zhihu.com/equation?tex=h_%7B%5Ctheta%7D%28x%5E%7B%28i%29%7D%29) 与真实值 ![[公式]](https://www.zhihu.com/equation?tex=y%5E%7B%28i%29%7D) 之间有如下关系：

![[公式]](https://www.zhihu.com/equation?tex=y%5E%7B%28i%29%7D%3Dh_%7B%5Ctheta%7D%28x%5E%7B%28i%29%7D%29%2B%5Cepsilon%5E%7B%28i%29%7D)

根据中心极限定理，当一个事件与很多独立随机变量有关，该事件服从正态分布 。一般来说，连续值我们都倾向于假设服从正态分布。假设每个样本的误差 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon%5E%7B%28i%29%7D) 独立同分布均值为 ![[公式]](https://www.zhihu.com/equation?tex=0) ，方差为$![[公式]](https://www.zhihu.com/equation?tex=%5Csigma) 的高斯分布 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon+%5E%7B%28i%29%7D-N%280%2C%5Csigma+%5E%7B2%7D%29) ,所以有：

![[公式]](https://www.zhihu.com/equation?tex=p%5Cleft+%28+%5Cepsilon+%5E%7B%28i%29%7D+%5Cright+%29%3D%5Cfrac%7B1%7D%7B%5Csqrt%7B2%5Cpi%7D+%7Dexp%5Cleft+%28+-%5Cfrac%7B%5Cleft+%28+%5Cepsilon+%5E%7B%28i%29%7D+%5Cright+%29%5E%7B2%7D%7D%7B2%5Csigma+%5E%7B2%7D%7D+%5Cright+%29)

即表示 ![[公式]](https://www.zhihu.com/equation?tex=y%5E%7B%28i%29%7D) 满足以均值为 ![[公式]](https://www.zhihu.com/equation?tex=h_%7B%5Ctheta%7D%28x%5E%7B%28i%29%7D%29) ,方差为 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon%5E%7B%28i%29%7D) 的高斯分布。

![[公式]](https://www.zhihu.com/equation?tex=p%5Cleft+%28+y+%5E%7B%28i%29%7D+%7C+x%5E%7B%28i%29%7D%3B%5Ctheta%5Cright+%29%3D%5Cfrac%7B1%7D%7B%5Csqrt%7B2%5Cpi%7D+%7Dexp%5Cleft+%28+-%5Cfrac%7B%5Cleft+%28+y+%5E%7B%28i%29%7D+-%5Ctheta%5E%7BT%7Dx%5E%7B%28i%29%7D+%5Cright+%29%5E%7B2%7D%7D%7B2%5Csigma+%5E%7B2%7D%7D+%5Cright+%29)

由最大似然估计有：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bequation%7D++%5Cbegin%7Bsplit%7D+%5Cmax+L%28%5Ctheta+%29%26%3DL%28%5Ctheta+%3Bx%5E%7B%28i%29%7D%2Cy%29%3Dp%28y%5E%7B%28i%29%7D%7Cx%5E%7B%28i%29%7D%3B%5Ctheta%29%5C%5C+L%28%5Ctheta+%3BX%2Cy%26%29%3D%5Cprod_%7Bi%3D1%7D%5E%7Bm%7Dp%28y%5E%7B%28i%29%7D%7Cx%5E%7B%28i%29%7D%3B%5Ctheta%29%5C%5C+%26%3D%5Cprod_%7Bi%3D1%7D%5E%7Bm%7D%5Cfrac%7B1%7D%7B%5Csqrt%7B2%5Cpi%7D+%7Dexp%5Cleft+%28+-%5Cfrac%7B%5Cleft+%28+y+%5E%7B%28i%29%7D+-%5Ctheta%5E%7BT%7Dx%5E%7B%28i%29%7D+%5Cright+%29%5E%7B2%7D%7D%7B2%5Csigma+%5E%7B2%7D%7D+%5Cright+%29%5C%5C+%5Cmax+logL%28%5Ctheta+%29%26%3D%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%5Clog%5Cfrac%7B1%7D%7B%5Csqrt%7B2%5Cpi%7D+%7Dexp%5Cleft+%28+-%5Cfrac%7B%5Cleft+%28+y+%5E%7B%28i%29%7D+-%5Ctheta%5E%7BT%7Dx%5E%7B%28i%29%7D+%5Cright+%29%5E%7B2%7D%7D%7B2%5Csigma+%5E%7B2%7D%7D+%5Cright+%29%5C%5C+%26%3Dmlog%5Cfrac%7B1%7D%7B%5Csqrt%7B2%5Cpi%7D+%7D-%5Cfrac%7B1%7D%7B2%5Csigma%5E%7B2%7D%7D.%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%5Cleft+%28+y+%5E%7B%28i%29%7D+-%5Ctheta%5E%7BT%7Dx%5E%7B%28i%29%7D+%5Cright+%29%5E%7B2%7D%5C%5C+%26%5CLeftrightarrow+%5Cmin+%5Cfrac%7B1%7D%7B2%5Csigma%5E%7B2%7D%7D.%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%5Cleft+%28+y+%5E%7B%28i%29%7D+-%5Ctheta%5E%7BT%7Dx%5E%7B%28i%29%7D+%5Cright+%29%5E%7B2%7D%3DJ%28%5Ctheta%29++%5Cend%7Bsplit%7D+%5Cend%7Bequation%7D)

## **二、岭回归和Lasso回归**



**过拟合问题及其解决方法**

- 问题：如下面一张图片展现过拟合问题
  ![这里写图片描述](https://resource.shangmayuan.com/droxy-blog/2020/02/11/518979dfc38f47678fd9f02e6e962750-2.JPEG)
- 解决方法：(1)：丢弃一些对咱们最终预测结果影响不大的特征，具体哪些特征须要丢弃能够经过PCA算法来实现；(2)：使用正则化技术，保留全部特征，可是减小特征前面的参数θ的大小，具体就是修改线性回归中的损失函数形式便可，岭回归以及Lasso回归就是这么作的。



 岭回归的目标函数在一般的线性回归的基础上加入了正则项，在保证最佳拟合误差的同时，使得参数尽可能的“简单”，使得模型的泛化能力强。正则项一般采用一，二范数，使得模型更具有泛化性，同时可以解决线性回归中不可逆情况，比如二范数对应的岭回归：

![[公式]](https://www.zhihu.com/equation?tex=%5Cmin_%7B%5Ctheta%7D%5Cfrac%7B1%7D%7B2%7D%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%5Cleft+%28+h_%7B%5Ctheta+%7D%5Cleft+%28+x%5E%7B%28i%29%7D+%5Cright+%29-y%5E%7B%28i%29%7D+%5Cright+%29%5E%7B2%7D+%2B+%5Clambda+%5Cleft+%5C%7C+%5Ctheta++%5Cright+%5C%7C%5E%7B2%7D)

其迭代优化函数如下：

![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta+_%7Bj%7D%EF%BC%9A%3D%5Ctheta+_%7Bj%7D-%5Calpha+%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%5Cleft+%28h_%7B%5Ctheta+%7D%5Cleft+%28+x%5E%7B%28i%29%7D%5Cright+%29-y%5E%7B%28i%29%7D++%5Cright+%29x_%7Bj%7D%5E%7B%28i%29%7D-2%5Clambda+%5Ctheta_%7Bj%7D)

另外**从最小二乘的角度来看，通过引入二范正则项，使其主对角线元素来强制矩阵可逆。**

![[公式]](https://www.zhihu.com/equation?tex=%5Ctriangledown_%7B%5Ctheta+%7D+J%28%5Ctheta+%29%3DXX%5E%7BT%7D%5Ctheta-XY+%2B%5Clambda+%5Ctheta%3D0%5C%5C+%5CRightarrow+%5Ctheta+%3D%5Cleft+%28XX%5E%7BT%7D+%2B+%5Clambda+I+%5Cright+%29%5E%7B-1%7DXY)



**Lasso回归采用一范数来约束，使参数非零个数最少**。而Lasso和岭回归的区别很好理解，在优化过程中，最优解为函数等值线与约束空间的交集，正则项可以看作是约束空间。可以看出二范的约束空间是一个球形，一范的约束空间是一个方形，这也就是二范会得到很多参数接近 ![[公式]](https://www.zhihu.com/equation?tex=0) 的值，而一范会尽可能非零参数最少。

![img](https://pic4.zhimg.com/80/v2-1cb777c2413e0c154e6e6de773a09753_1440w.jpg)

 值得注意的是线性模型的表示能力有限，但是并不一定表示线性模型只能处理线性分布的数据。这里有两种常用的线性模型非线性化。对于上面的线性函数的构造，我们可以看出模型在以 ![[公式]](https://www.zhihu.com/equation?tex=%7Bx_%7B0%7D%2Cx_%7B1%7D%2C..%2Cx_%7Bn%7D%7D) 的坐标上是线性的，但是并不表示线性的模型就一定只能用于线性分布问题上。假如我们只有一个特征 ![[公式]](https://www.zhihu.com/equation?tex=%7Bx_%7B0%7D%7D)，而实际上回归值是 ![[公式]](https://www.zhihu.com/equation?tex=y%3Dx_%7B0%7D%5E%7B2%7D) 等问题，我们同样可以采用线性模型，因为我们完全可以把输入空间映射到高维空间 ![[公式]](https://www.zhihu.com/equation?tex=%28x_%7B1%7D%5E%7B3%7D%2Cx_%7B1%7D%5E%7B2%7D%2Cx_%7B1%7D%5E%7B1%7D%29) ，其实这也是核方法以及PCA空间变换的一种思想，凡是对输入空间进行线性，非线性的变换，都是把输入空间映射到特征空间的思想，所以只需要把非线性问题转化为线性问题即可。另外一种实现线性回归非线性表示能力的是局部线性思想，即对每一个样本构建一个加权的线性模型。

岭回归与Lasso回归的出现是为了解决线性回归出现的过拟合以及在经过正规方程方法求解θ的过程当中出现的x转置乘以x不可逆这两类问题的，这两种回归均经过在损失函数中引入正则化项来达到目的，具体三者的损失函数对比见下图：
![这里写图片描述](https://resource.shangmayuan.com/droxy-blog/2020/02/11/6ef12e46cd2d411184c90df8adad2a7a-2.gif)
其中λ称为正则化参数，若是λ选取过大，会把全部参数θ均最小化，形成欠拟合，若是λ选取太小，会致使对过拟合问题解决不当，所以λ的选取是一个技术活。
岭回归与Lasso回归最大的区别在于岭回归引入的是L2范数惩罚项，**Lasso回归引入的是L1范数惩罚项，Lasso回归可以使得损失函数中的许多θ均变成0，这点要优于岭回归，由于岭回归是要全部的θ均存在的，这样计算量Lasso回归将远远小于岭回归。**

**三、局部加权线性回归**

 考虑到线性回归的表示能力有限，可能出现欠拟合现象。局部加权线性回归为每一个待预测的点构建一个加权的线性模型。其加权的方式是根据预测点与数据集中点的距离来为数据集中的点赋权重，当某点距离预测点较远时，其权重较小，反之较大。由于这种权重的机制引入使得局部加权线性回归产生了一种局部分段拟合的效果。由于该方法对于每一个预测点构建一个加权线性模型，都要重新计算与数据集中所有点的距离来确定权重值，进而确定针对该预测点的线性模型，计算成本高，同时为了实现无参估计来计算权重，需要存储整个数据集。

局部加权线性回归，在线性回归基础上引入权重，其目标函数（下面的目标函数是针对一个预测样本的）如下：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bequation%7D++%5Cbegin%7Bsplit%7D+J%5Cleft+%28+%5Ctheta+%5Cright+%29%26%3D%5Cfrac%7B1%7D%7B2%7D%5Csum_%7Bi%3D1%7D%5E%7Bm%7Dw%5E%7B%28i%29%7D%5Cleft+%28+h_%7B%5Ctheta+%7D%5Cleft+%28+x%5E%7B%28i%29%7D+%5Cright+%29-y%5E%7B%28i%29%7D+%5Cright+%29%5E%7B2%7D%5C%5C+%26%5Cmin_%7B%5Ctheta+%7DJ%5Cleft+%28+%5Ctheta+%5Cright+%29++%5Cend%7Bsplit%7D+%5Cend%7Bequation%7D)

一般选择下面的权重函数，权重函数选择一般考虑数据的分布特性。

![[公式]](https://www.zhihu.com/equation?tex=w%5E%7B%28i%29%7D%3Dexp%5Cleft+%28+-%5Cfrac%7Bx%5E%7B%28i%29%7D-x%7D%7B2%5Csigma+%5E%7B2%7D%7D+%5Cright+%29)

其中 ![[公式]](https://www.zhihu.com/equation?tex=x) 是待预测的一个数据点。

 对于上面的目标函数，我们的目标同样是使得损失函数最小化，同样局部加权线性回归可以采用梯度的方法，也可以从最小二乘法的角度给出闭式解。

![[公式]](https://www.zhihu.com/equation?tex=%5Ctriangledown_%7B%5Ctheta+%7D+J%28%5Ctheta+%29%3DXWX%5E%7BT%7D%5Ctheta-XWY+%3D0%5C%5C+%5CRightarrow+%5Ctheta+%3D%5Cleft+%28XWX%5E%7BT%7D+I+%5Cright+%29%5E%7B-1%7DXWY)

其中![[公式]](https://www.zhihu.com/equation?tex=W) 是对角矩阵，![[公式]](https://www.zhihu.com/equation?tex=W_%7Bii%7D%3Dw%5E%7B%28i%29%7D) 。

 线性回归核心思想最小化平方误差，可以从最小化损失函数和最小二乘角度来看，也有概率解释。优化过程可以采用梯度方法和闭式解。在闭式解问题中需要注意矩阵可逆问题。考虑到过拟合和欠拟合问题，有岭回归和lasso回归来防止过拟合，局部加权线性回归通过加权实现非线性表示。