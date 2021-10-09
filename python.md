# python

## Python 数据类型

- Number（数字）
- String（字符串）
- List（列表）
- Tuple（元组）
- Set（集合）
- Dictionary（字典）

Python3 的六个标准数据类型中：

- **不可变数据（3 个）：**Number（数字）、String（字符串）、Tuple（元组）；
- **可变数据（3 个）：**List（列表）、Dictionary（字典）、Set（集合）。\

| 类型      | 例子                     |
| --------- | ------------------------ |
| 整数      | `-100`                   |
| 浮点数    | `3.1416`                 |
| 字符串    | `'hello'`                |
| 列表      | `[1, 1.2, 'hello']`      |
| 字典      | `{'dogs': 5, 'pigs': 3}` |
| Numpy数组 | `array([1, 2, 3])`       |



| 类型       | 例子                      |
| ---------- | ------------------------- |
| 长整型     | `1000000000000L`          |
| 布尔型     | `True, False`             |
| 元组       | `('ring', 1000)`          |
| 集合       | `{1, 2, 3}`               |
| Pandas类型 | `DataFrame, Series`       |
| 自定义     | `Object Oriented Classes` |

### Number（数字）

Python3 支持 **int、float、bool、complex（复数）**。

在Python 3里，只有一种整数类型 int，表示为长整型，没有 python2 中的 Long。

像大多数语言一样，数值类型的赋值和计算都是很直观的。

内置的 type() 函数可以用来查询变量所指的对象类型。

```
>>> a, b, c, d = 20, 5.5, True, 4+3j
>>> print(type(a), type(b), type(c), type(d))
<class 'int'> <class 'float'> <class 'bool'> <class 'complex'>
```

此外还可以用 isinstance 来判断：

*实例*

\>>> a = 111
\>>> isinstance(a, int)
True
\>>>

isinstance 和 type 的区别在于：

- type()不会认为子类是一种父类类型。
- isinstance()会认为子类是一种父类类型。

```
>>> class A:
...     pass
... 
>>> class B(A):
...     pass
... 
>>> isinstance(A(), A)
True
>>> type(A()) == A 
True
>>> isinstance(B(), A)
True
>>> type(B()) == A
False
```

> **注意：**Python3 中，bool 是 int 的子类，True 和 False 可以和数字相加， **True==1、False==0** 会返回 **True**，但可以通过 **is** 来判断类型。
>
> ```
> >>> issubclass(bool, int) 
> True
> >>> True==1
> True
> >>> False==0
> True
> >>> True+1
> 2
> >>> False+1
> 1
> >>> 1 is True
> False
> >>> 0 is False
> False
> ```
>
> 在 Python2 中是没有布尔型的，它用数字 0 表示 False，用 1 表示 True。

当你指定一个值时，Number 对象就会被创建：

```
var1 = 1
var2 = 10
```

您也可以使用del语句删除一些对象引用。

del语句的语法是：

```
del var1[,var2[,var3[....,varN]]]
```

您可以通过使用del语句删除单个或多个对象。例如：

```
del var
del var_a, var_b
```

**注意：**

- 1、Python可以同时为多个变量赋值，如a, b = 1, 2。
- 2、一个变量可以通过赋值指向不同类型的对象。
- 3、数值的除法包含两个运算符：**/** 返回一个浮点数，**//** 返回一个整数。
- 4、在混合计算时，Python会把整型转换成为浮点数。

*数值类型实例*

| int    | float      | complex    |
| :----- | :--------- | :--------- |
| 10     | 0.0        | 3.14j      |
| 100    | 15.20      | 45.j       |
| -786   | -21.9      | 9.322e-36j |
| 080    | 32.3e+18   | .876j      |
| -0490  | -90.       | -.6545+0J  |
| -0x260 | -32.54e100 | 3e+26J     |
| 0x69   | 70.2E-12   | 4.53e-7j   |

Python还支持复数，复数由实数部分和虚数部分构成，可以用a + bj,或者complex(a,b)表示， 复数的实部a和虚部b都是浮点型

### String（字符串）

Python中的字符串用单引号 **'** 或双引号 **"** 括起来，同时使用反斜杠 **\** 转义特殊字符。

字符串的截取的语法格式如下：

```
变量[头下标:尾下标]
```

索引值以 0 为开始值，-1 为从末尾的开始位置。

![img](https://static.runoob.com/wp-content/uploads/123456-20200923-1.svg)

加号 **+** 是字符串的连接符， 星号 ***** 表示复制当前字符串，与之结合的数字为复制的次数。实例如下：

*实例*

\#!/usr/bin/python3

str = 'Runoob'

**print** (str)      # 输出字符串
**print** (str[0:-1])   # 输出第一个到倒数第二个的所有字符
**print** (str[0])    # 输出字符串第一个字符
**print** (str[2:5])   # 输出从第三个开始到第五个的字符
**print** (str[2:])    # 输出从第三个开始的后的所有字符
**print** (str * 2)    # 输出字符串两次，也可以写成 print (2 * str)
**print** (str + "TEST") # 连接字符串

执行以上程序会输出如下结果：

```
Runoob
Runoo
R
noo
noob
RunoobRunoob
RunoobTEST
```

Python 使用反斜杠 **\** 转义特殊字符，如果你不想让反斜杠发生转义，可以在字符串前面添加一个 **r**，表示原始字符串：

*实例*

\>>> **print**('Ru**\n**oob')
Ru
oob
\>>> **print**(r'Ru**\n**oob')
Ru\noob
\>>>

另外，反斜杠(\)可以作为续行符，表示下一行是上一行的延续。也可以使用 **"""..."""** 或者 **'''...'''** 跨越多行。

注意，Python 没有单独的字符类型，一个字符就是长度为1的字符串。

与 C 字符串不同的是，Python 字符串不能被改变。向一个索引位置赋值，比如word[0] = 'm'会导致错误。

**注意：**

- 1、反斜杠可以用来转义，使用r可以让反斜杠不发生转义。
- 2、字符串可以用+运算符连接在一起，用*运算符重复。
- 3、Python中的字符串有两种索引方式，从左往右以0开始，从右往左以-1开始。
- 4、Python中的字符串不能改变。、

### List（列表）

List（列表） 是 Python 中使用最频繁的数据类型。

列表可以完成大多数集合类的数据结构实现。列表中元素的类型可以不相同，它支持数字，字符串甚至可以包含列表（所谓嵌套）。

列表是写在方括号 **[]** 之间、用逗号分隔开的元素列表。

和字符串一样，列表同样可以被索引和截取，列表被截取后返回一个包含所需元素的新列表。

列表截取的语法格式如下：

```
变量[头下标:尾下标]
```

索引值以 **0** 为开始值，**-1** 为从末尾的开始位置。

![img](https://www.runoob.com/wp-content/uploads/2014/08/list_slicing1_new1.png)

加号 **+** 是列表连接运算符，星号 ***** 是重复操作。如下实例：

*实例*

\#!/usr/bin/python3

list = [ 'abcd', 786 , 2.23, 'runoob', 70.2 ]
tinylist = [123, 'runoob']

**print** (list)       # 输出完整列表
**print** (list[0])     # 输出列表第一个元素
**print** (list[1:3])    # 从第二个开始输出到第三个元素
**print** (list[2:])     # 输出从第三个元素开始的所有元素
**print** (tinylist * 2)   # 输出两次列表
**print** (list + tinylist) # 连接列表

以上实例输出结果：

```
['abcd', 786, 2.23, 'runoob', 70.2]
abcd
[786, 2.23]
[2.23, 'runoob', 70.2]
[123, 'runoob', 123, 'runoob']
['abcd', 786, 2.23, 'runoob', 70.2, 123, 'runoob']
```

与Python字符串不一样的是，列表中的元素是可以改变的：

*实例*

\>>> a = [1, 2, 3, 4, 5, 6]
\>>> a[0] = 9
\>>> a[2:5] = [13, 14, 15]
\>>> a
[9, 2, 13, 14, 15, 6]
\>>> a[2:5] = []  # 将对应的元素值设置为 []
\>>> a
[9, 2, 6]

**注意：**

- 1、List写在方括号之间，元素用逗号隔开。
- 2、和字符串一样，list可以被索引和切片。
- 3、List可以使用+操作符进行拼接。
- 4、List中的元素是可以改变的。

### Tuple（元组）

元组（tuple）与列表类似，不同之处在于元组的元素不能修改。元组写在小括号 **()** 里，元素之间用逗号隔开。

元组中的元素类型也可以不相同：

*实例*

\#!/usr/bin/python3

tuple = ( 'abcd', 786 , 2.23, 'runoob', 70.2 )
tinytuple = (123, 'runoob')

**print** (tuple)       # 输出完整元组
**print** (tuple[0])      # 输出元组的第一个元素
**print** (tuple[1:3])     # 输出从第二个元素开始到第三个元素
**print** (tuple[2:])     # 输出从第三个元素开始的所有元素
**print** (tinytuple * 2)   # 输出两次元组
**print** (tuple + tinytuple) # 连接元组

以上实例输出结果：

```
('abcd', 786, 2.23, 'runoob', 70.2)
abcd
(786, 2.23)
(2.23, 'runoob', 70.2)
(123, 'runoob', 123, 'runoob')
('abcd', 786, 2.23, 'runoob', 70.2, 123, 'runoob')
```

元组与字符串类似，可以被索引且下标索引从0开始，-1 为从末尾开始的位置。也可以进行截取（看上面，这里不再赘述）。

其实，可以把字符串看作一种特殊的元组。

*实例*

\>>> tup = (1, 2, 3, 4, 5, 6)
\>>> **print**(tup[0])
1
\>>> **print**(tup[1:5])
(2, 3, 4, 5)
\>>> tup[0] = 11 # 修改元组元素的操作是非法的
Traceback (most recent call last):
 File "<stdin>", line 1, **in** <module>
TypeError: 'tuple' object does **not** support item assignment
\>>>

虽然tuple的元素不可改变，但它可以包含可变的对象，比如list列表。

构造包含 0 个或 1 个元素的元组比较特殊，所以有一些额外的语法规则：

```
tup1 = ()    # 空元组
tup2 = (20,) # 一个元素，需要在元素后添加逗号
```

string、list 和 tuple 都属于 sequence（序列）。

**注意：**

- 1、与字符串一样，元组的元素不能修改。
- 2、元组也可以被索引和切片，方法一样。
- 3、注意构造包含 0 或 1 个元素的元组的特殊语法规则。
- 4、元组也可以使用+操作符进行拼接。

### Set（集合）

集合（set）是由一个或数个形态各异的大小整体组成的，构成集合的事物或对象称作元素或是成员。

基本功能是进行成员关系测试和删除重复元素。

可以使用大括号 **{ }** 或者 **set()** 函数创建集合，注意：创建一个空集合必须用 **set()** 而不是 **{ }**，因为 **{ }** 是用来创建一个空字典。

创建格式：

```
parame = {value01,value02,...}
或者
set(value)
```

*实例*

\#!/usr/bin/python3

sites = {'Google', 'Taobao', 'Runoob', 'Facebook', 'Zhihu', 'Baidu'}

**print**(sites)  # 输出集合，重复的元素被自动去掉

\# 成员测试
**if** 'Runoob' **in** sites :
  **print**('Runoob 在集合中')
**else** :
  **print**('Runoob 不在集合中')


\# set可以进行集合运算
a = set('abracadabra')
b = set('alacazam')

**print**(a)

**print**(a - b)   # a 和 b 的差集

**print**(a | b)   # a 和 b 的并集

**print**(a & b)   # a 和 b 的交集

**print**(a ^ b)   # a 和 b 中不同时存在的元素

以上实例输出结果：

```
{'Zhihu', 'Baidu', 'Taobao', 'Runoob', 'Google', 'Facebook'}
Runoob 在集合中
{'b', 'c', 'a', 'r', 'd'}
{'r', 'b', 'd'}
{'b', 'c', 'a', 'z', 'm', 'r', 'l', 'd'}
{'c', 'a'}
{'z', 'b', 'm', 'r', 'l', 'd'}
```

------

### Dictionary（字典）

字典（dictionary）是Python中另一个非常有用的内置数据类型。

列表是有序的对象集合，字典是无序的对象集合。两者之间的区别在于：字典当中的元素是通过键来存取的，而不是通过偏移存取。

字典是一种映射类型，字典用 **{ }** 标识，它是一个无序的 **键(key) : 值(value)** 的集合。

键(key)必须使用不可变类型。

在同一个字典中，键(key)必须是唯一的。

*实例*

\#!/usr/bin/python3

dict = {}
dict['one'] = "1 - 教程"
dict[2]   = "2 - 工具"

tinydict = {'name': 'runoob','code':1, 'site': 'www.runoob.com'}


**print** (dict['one'])    # 输出键为 'one' 的值
**print** (dict[2])      # 输出键为 2 的值
**print** (tinydict)      # 输出完整的字典
**print** (tinydict.keys())  # 输出所有键
**print** (tinydict.values()) # 输出所有值

以上实例输出结果：

```
1 - 教程
2 - 工具
{'name': 'runoob', 'code': 1, 'site': 'www.runoob.com'}
dict_keys(['name', 'code', 'site'])
dict_values(['runoob', 1, 'www.runoob.com'])
```

构造函数 dict() 可以直接从键值对序列中构建字典如下：

*实例*

\>>> dict([('Runoob', 1), ('Google', 2), ('Taobao', 3)])
{'Runoob': 1, 'Google': 2, 'Taobao': 3}
\>>> {x: x**2 **for** x **in** (2, 4, 6)}
{2: 4, 4: 16, 6: 36}
\>>> dict(Runoob=1, Google=2, Taobao=3)
{'Runoob': 1, 'Google': 2, 'Taobao': 3}
\>>>



另外，字典类型也有一些内置的函数，例如clear()、keys()、values()等。

**注意：**

- 1、字典是一种映射类型，它的元素是键值对。
- 2、字典的关键字必须为不可变类型，且不能重复。
- 3、创建空字典使用 **{ }**。





## 列表推导式

循环可以用来生成列表：

In [1]:

```py
values = [10, 21, 4, 7, 12]
squares = []
for x in values:
    squares.append(x**2)
print squares
复制ErrorOK!
[100, 441, 16, 49, 144]
复制ErrorOK!
```

列表推导式可以使用更简单的方法来创建这个列表：

In [2]:

```py
values = [10, 21, 4, 7, 12]
squares = [x**2 for x in values]
print squares
复制ErrorOK!
[100, 441, 16, 49, 144]
复制ErrorOK!
```

还可以在列表推导式中加入条件进行筛选。

例如在上面的例子中，假如只想保留列表中不大于`10`的数的平方：

In [3]:

```py
values = [10, 21, 4, 7, 12]
squares = [x**2 for x in values if x <= 10]
print squares
复制ErrorOK!
[100, 16, 49]
复制ErrorOK!
```

也可以使用推导式生成集合和字典：

In [4]:

```py
square_set = {x**2 for x in values if x <= 10}
print(square_set)
square_dict = {x: x**2 for x in values if x <= 10}
print(square_dict)
复制ErrorOK!
set([16, 49, 100])
{10: 100, 4: 16, 7: 49}
复制ErrorOK!
```

再如，计算上面例子中生成的列表中所有元素的和：

In [5]:

```py
total = sum([x**2 for x in values if x <= 10])
print(total)
复制ErrorOK!
165
复制ErrorOK!
```

但是，**Python**会生成这个列表，然后在将它放到垃圾回收机制中（因为没有变量指向它），这毫无疑问是种浪费。

为了解决这种问题，与xrange()类似，**Python**使用产生式表达式来解决这个问题：

In [6]:

```py
total = sum(x**2 for x in values if x <= 10)
print(total)
复制ErrorOK!
165
复制ErrorOK!
```

与上面相比，只是去掉了括号，但这里并不会一次性的生成这个列表。

比较一下两者的用时：

In [7]:

```py
x = range(1000000)
复制ErrorOK!
```

In [8]:

```py
%timeit total = sum([i**2 for i in x])
复制ErrorOK!
1 loops, best of 3: 3.86 s per loop
复制ErrorOK!
```

In [9]:

```py
%timeit total = sum(i**2 for i in x)
复制ErrorOK!
1 loops, best of 3: 2.58 s per loop
```





## 函数

### 定义函数

函数`function`，通常接受输入参数，并有返回值。

它负责完成某项特定任务，而且相较于其他代码，具备相对的独立性。

In [1]:

```py
def add(x, y):
    """Add two numbers"""
    a = x + y
    return a
复制ErrorOK!
```

函数通常有一下几个特征：

- 使用 `def` 关键词来定义一个函数。
- `def` 后面是函数的名称，括号中是函数的参数，不同的参数用 `,` 隔开， `def foo():` 的形式是必须要有的，参数可以为空；
- 使用缩进来划分函数的内容；
- `docstring` 用 `"""` 包含的字符串，用来解释函数的用途，可省略；
- `return` 返回特定的值，如果省略，返回 `None` 。

### 设定参数默认值

可以在函数定义的时候给参数设定默认值，例如：

In [8]:

```py
def quad(x, a=1, b=0, c=0):
    return a*x**2 + b*x + c
复制ErrorOK!
```

可以省略有默认值的参数：

In [9]:

```py
print quad(2.0)
复制ErrorOK!
4.0
复制ErrorOK!
```

可以修改参数的默认值：

In [10]:

```py
print quad(2.0, b=3)
复制ErrorOK!
10.0
复制ErrorOK!
```

In [11]:

```py
print quad(2.0, 2, c=4)
复制ErrorOK!
12.0
复制ErrorOK!
```

这里混合了位置和指定两种参数传入方式，第二个2是传给 `a` 的。

注意，在使用混合语法时，要注意不能给同一个值赋值多次，否则会报错，例如：

In [12]:

```py
print quad(2.0, 2, a=2)
复制ErrorOK!
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-12-101d0c090bbb> in <module>()
----> 1  print quad(2.0, 2, a=2)

TypeError: quad() got multiple values for keyword argument 'a'
```

### 接收不定参数

使用如下方法，可以使函数接受不定数目的参数：

In [13]:

```py
def add(x, *args):
    total = x
    for arg in args:
        total += arg
    return total
复制ErrorOK!
```

这里，`*args` 表示参数数目不定，可以看成一个元组，把第一个参数后面的参数当作元组中的元素。

In [14]:

```py
print add(1, 2, 3, 4)
print add(1, 2)
复制ErrorOK!
10
3
复制ErrorOK!
```

这样定义的函数不能使用关键词传入参数，要使用关键词，可以这样：

In [15]:

```py
def add(x, **kwargs):
    total = x
    for arg, value in kwargs.items():
        print "adding ", arg
        total += value
    return total
复制ErrorOK!
```

这里， `**kwargs` 表示参数数目不定，相当于一个字典，关键词和值对应于键值对。

In [16]:

```py
print add(10, y=11, z=12, w=13)
复制ErrorOK!
adding  y
adding  z
adding  w
46
复制ErrorOK!
```

再看这个例子，可以接收任意数目的位置参数和键值对参数：

In [17]:

```py
def foo(*args, **kwargs):
    print args, kwargs

foo(2, 3, x='bar', z=10)
复制ErrorOK!
(2, 3) {'x': 'bar', 'z': 10}
复制ErrorOK!
```

不过要按顺序传入参数，先传入位置参数 `args` ，在传入关键词参数 `kwargs` 。

### map 方法生成序列

可以通过 `map` 的方式利用函数来生成序列：

In [23]:

```py
def sqr(x): 
    return x ** 2

a = [2,3,4]
print map(sqr, a)
复制ErrorOK!
[4, 9, 16]
复制ErrorOK!
```

其用法为：

```py
map(aFun, aSeq) 复制ErrorOK!
```

将函数 `aFun` 应用到序列 `aSeq` 上的每一个元素上，返回一个列表，不管这个序列原来是什么类型。

事实上，根据函数参数的多少，`map` 可以接受多组序列，将其对应的元素作为参数传入函数：

In [24]:

```py
def add(x, y): 
    return x + y

a = (2,3,4)
b = [10,5,3]
print map(add,a,b)
复制ErrorOK!
[12, 8, 7]
```





## 模块和包

Python会将所有 `.py` 结尾的文件认定为Python代码文件，考虑下面的脚本 `ex1.py` ：

In [1]:

```py
%%writefile ex1.py

PI = 3.1416

def sum(lst):
    tot = lst[0]
    for value in lst[1:]:
        tot = tot + value
    return tot

w = [0, 1, 2, 3]
print sum(w), PI
复制ErrorOK!
Overwriting ex1.py
复制ErrorOK!
```

可以执行它：

In [2]:

```py
%run ex1.py
复制ErrorOK!
6 3.1416
复制ErrorOK!
```

这个脚本可以当作一个模块，可以使用`import`关键词加载并执行它（这里要求`ex1.py`在当前工作目录）：

In [3]:

```py
import ex1
复制ErrorOK!
6 3.1416
复制ErrorOK!
```

In [4]:

```py
ex1
复制ErrorOK!
```

Out[4]:

```py
<module 'ex1' from 'ex1.py'>复制ErrorOK!
```

在导入时，**Python**会执行一遍模块中的所有内容。

`ex1.py` 中所有的变量都被载入了当前环境中，不过要使用

```py
ex1.变量名 复制ErrorOK!
```

的方法来查看或者修改这些变量：

In [5]:

```py
print ex1.PI
复制ErrorOK!
3.1416
复制ErrorOK!
```

In [6]:

```py
ex1.PI = 3.141592653
print ex1.PI
复制ErrorOK!
3.141592653
复制ErrorOK!
```

还可以用

```py
ex1.函数名 复制ErrorOK!
```

调用模块里面的函数：

In [7]:

```py
print ex1.sum([2, 3, 4])
复制ErrorOK!
9
复制ErrorOK!
```

为了提高效率，**Python**只会载入模块一次，已经载入的模块再次载入时，Python并不会真正执行载入操作，哪怕模块的内容已经改变。

例如，这里重新导入 `ex1` 时，并不会执行 `ex1.py` 中的 `print` 语句：

In [8]:

```py
import ex1
复制ErrorOK!
```

需要重新导入模块时，可以使用`reload`强制重新载入它，例如：

In [9]:

```py
reload(ex1)
复制ErrorOK!
6 3.1416
复制ErrorOK!
```

Out[9]:

```py
<module 'ex1' from 'ex1.pyc'>复制ErrorOK!
```

删除之前生成的文件：

In [10]:

```py
import os
os.remove('ex1.py')
```

### __name__ 属性

有时候我们想将一个 `.py` 文件既当作脚本，又能当作模块用，这个时候可以使用 `__name__` 这个属性。

只有当文件被当作脚本执行的时候， `__name__`的值才会是 `'__main__'`，所以我们可以：

In [11]:

```py
%%writefile ex2.py

PI = 3.1416

def sum(lst):
    """ Sum the values in a list
    """
    tot = 0
    for value in lst:
        tot = tot + value
    return tot

def add(x, y):
    " Add two values."
    a = x + y
    return a

def test():
    w = [0,1,2,3]
    assert(sum(w) == 6)
    print 'test passed.'

if __name__ == '__main__':
    test()
复制ErrorOK!
Writing ex2.py
复制ErrorOK!
```

运行文件：

In [12]:

```py
%run ex2.py
复制ErrorOK!
test passed.
复制ErrorOK!
```

当作模块导入， `test()` 不会执行：

In [13]:

```py
import ex2
复制ErrorOK!
```

但是可以使用其中的变量：

In [14]:

```py
ex2.PI
复制ErrorOK!
```

Out[14]:

```py
3.1416复制ErrorOK!
```

使用别名：

In [15]:

```py
import ex2 as e2
e2.PI
复制ErrorOK!
```

Out[15]:

```py
3.1416
```

### 其他导入方法

可以从模块中导入变量：

In [16]:

```py
from ex2 import add, PI
复制ErrorOK!
```

使用 `from` 后，可以直接使用 `add` ， `PI`：

In [17]:

```py
add(2, 3)
复制ErrorOK!
```

Out[17]:

```py
5复制ErrorOK!
```

或者使用 `*` 导入所有变量：

In [18]:

```py
from ex2 import *
add(3, 4.5)
复制ErrorOK!
```

Out[18]:

```py
7.5复制ErrorOK!
```

这种导入方法不是很提倡，因为如果你不确定导入的都有哪些，可能覆盖一些已有的函数。

删除文件：

In [19]:

```py
import os
os.remove('ex2.py')
复制ErrorOK!
```

### [包](https://apachecn.gitee.io/ailearning/#/docs/da/024?id=包)

假设我们有这样的一个文件夹：

foo/

- `__init__.py`
- `bar.py` (defines func)
- `baz.py` (defines zap)

这意味着 foo 是一个包，我们可以这样导入其中的内容：

```py
from foo.bar import func
from foo.baz import zap
复制ErrorOK!
```

`bar` 和 `baz` 都是 `foo` 文件夹下的 `.py` 文件。

导入包要求：

- 文件夹 `foo` 在**Python**的搜索路径中
- `__init__.py` 表示 `foo` 是一个包，它可以是个空文件。

### [常用的标准库](https://apachecn.gitee.io/ailearning/#/docs/da/024?id=常用的标准库)

- re 正则表达式
- copy 复制
- math, cmath 数学
- decimal, fraction
- sqlite3 数据库
- os, os.path 文件系统
- gzip, bz2, zipfile, tarfile 压缩文件
- csv, netrc 各种文件格式
- xml
- htmllib
- ftplib, socket
- cmd 命令行
- pdb
- profile, cProfile, timeit
- collections, heapq, bisect 数据结构
- mmap
- threading, Queue 并行
- multiprocessing
- subprocess
- pickle, cPickle
- struct

### [PYTHONPATH设置](https://apachecn.gitee.io/ailearning/#/docs/da/024?id=pythonpath设置)

Python的搜索路径可以通过环境变量PYTHONPATH设置，环境变量的设置方法依操作系统的不同而不同，具体方法可以网上搜索。