# 目录
- [1.Python中迭代器的概念？](#1.python中迭代器的概念？)
- [2.Python中生成器的相关知识](#2.python中生成器的相关知识)
- [3.Python中装饰器的相关知识](#3.python中装饰器的相关知识)
- [4.Python的深拷贝与浅拷贝？](#4.python的深拷贝与浅拷贝？)
- [5.Python是解释语言还是编译语言？](#5.python是解释语言还是编译语言？)
- [6.Python的垃圾回收机制](#6.python的垃圾回收机制)
- [7.Python里有多线程吗？](#7.python里有多线程吗？)
- [8.Python中range和xrange的区别？](#8.python中range和xrange的区别？)
- [9.Python中列表和元组的区别？](#9.python中列表和元组的区别？)
- [10.Python中dict（字典）的底层结构？](#10.python中dict（字典）的底层结构？)
- [11.常用的深度学习框架有哪些，都是哪家公司开发的？](#11.常用的深度学习框架有哪些，都是哪家公司开发的？)
- [12.PyTorch动态图和TensorFlow静态图的区别？](#12.pytorch动态图和tensorflow静态图的区别？)
- [13.Python中assert的作用？](#13.python中assert的作用？)
- [14.Python中互换变量有不用创建临时变量的方法吗？](#14.python中互换变量有不用创建临时变量的方法吗？)
- [15.Python中的主要数据结构都有哪些？](#15.python中的主要数据结构都有哪些？)
- [16.Python中的可变对象和不可变对象？](#16.python中的可变对象和不可变对象？)
- [17.Python中的None代表什么？](#17.python中的none代表什么？)
- [18.Python中 $ *args $ 和 $ **kwargs $ 的区别？](#18.python中args和kwargs的区别？)
- [19.Python中Numpy的broadcasting机制？](#19.python中numpy的broadcasting机制？)
- [20.Python中的实例方法、静态方法和类方法三者区别？](#20.python中的实例方法、静态方法和类方法三者区别？)
- [21.Python中常见的切片操作](#21.python中常见的切片操作)
- [22.Python中如何进行异常处理？](#22.python中如何进行异常处理？)
- [23.Python中remove，del以及pop之间的区别？](#23.python中remove，del以及pop之间的区别？)
- [24.Python中PIL和OpenCV处理图像的区别？](#24.Python中PIL和OpenCV处理图像的区别？)
- [25.Python中全局变量与局部变量之间的区别？](#25.Python中全局变量与局部变量之间的区别？)
- [26.Python中`if "__name__" == __main__'`的作用?](#26.Python中name==main?)
- [27.Python中assert的作用?](#27.Python中assert的作用?)
- [28.python中如何无损打开图像，并无损保存图像?](#28.python中如何无损打开图像，并无损保存图像?)
- [29.PyTorch中张量操作Clone与Detach的区别?（腾讯实习二面）](#29.PyTorch中张量操作Clone与Detach的区别?（腾讯实习二面）)
- [30.Python多进程中的fork和spawn模式有什么区别？](#30.Python多进程中的fork和spawn模式有什么区别？)
- [31.什么是Python中的推导式？Python的推导式一共有多少种？](#31.什么是Python中的推导式？Python的推导式一共有多少种？)
- [32.python中一共都有哪些数据结构？](#32.python中一共都有哪些数据结构？)
- [33.python中index使用注意事项](#33.python中index使用注意事项)
- [34.python中@staticmethod和@classmethod使用注意事项](#34.python中@staticmethod和@classmethod使用注意事项)
- [35.Python中函数传参时会改变参数本身吗？](#35.Python中函数传参时会改变参数本身吗？)
- [36.什么是python的全局解释器锁GIL？](#36.什么是python的全局解释器锁GIL？)
- [37.什么是python的字符串格式化技术？](#37.什么是python的字符串格式化技术？)


<h2 id="1.python中迭代器的概念？">1.Python中迭代器的概念？</h2>

<font color=DeepSkyBlue>可迭代对象是迭代器、生成器和装饰器的基础。</font>简单来说，可以使用for来循环遍历的对象就是可迭代对象。比如常见的list、set和dict。

我们来看一个🌰：
```
from collections import Iterable
print(isinstance('abcddddd', Iterable))     # str是否可迭代

print(isinstance([1,2,3,4,5,6], Iterable))   # list是否可迭代

print(isinstance(12345678, Iterable))       # 整数是否可迭代

-------------结果如下----------------
True
True
False
```

当对所有的可迭代对象调用 dir() 方法时，会发现他们都实现了 iter 方法。这样就可以通过 iter(object) 来返回一个迭代器。
  
```
x = [1, 2, 3]
y = iter(x)
print(type(x))

print(type(y))

------------结果如下------------
<class 'list'>
<class 'list_iterator'>
```

可以看到调用iter()之后，变成了一个list_iterator的对象。可以发现增加了一个__next__方法。<font color=DeepSkyBlue>所有实现了__iter__和__next__两个方法的对象，都是迭代器</font>。

<font color=DeepSkyBlue>迭代器是带状态的对象，它会记录当前迭代所在的位置，以方便下次迭代的时候获取正确的元素。</font>__iter__返回迭代器自身，__next__返回容器中的下一个值，如果容器中没有更多元素了，则抛出Stoplteration异常。

```
x = [1, 2, 3]
y = iter(x)
print(next(y))
print(next(y))
print(next(y))
print(next(y))

----------结果如下----------
1
2
3
Traceback (most recent call last):
  File "/Users/Desktop/test.py", line 6, in <module>
    print(next(y))
StopIteration
```

如何判断对象是否是迭代器，和判断是否是可迭代对象的方法差不多，只要把 Iterable 换成 Iterator。

Python的for循环本质上就是通过不断调用next()函数实现的，举个栗子，下面的代码先将可迭代对象转化为Iterator，再去迭代。<font color=DeepSkyBlue>这样可以节省对内存，因为迭代器只有在我们调用 next() 才会实际计算下一个值</font>。

```
x = [1, 2, 3]
for elem in x:
    ...
```

![](https://files.mdnice.com/user/33499/88b65d43-fc23-4324-95c0-bc4002523cdc.png)

itertools 库提供了很多常见迭代器的使用。 

```
>>> from itertools import count     # 计数器
>>> counter = count(start=13)
>>> next(counter)
13
>>> next(counter)
14
```


<h2 id="2.python中生成器的相关知识">2.Python中生成器的相关知识</h2>
  
我们创建列表的时候，受到内存限制，容量肯定是有限的，而且不可能全部给他一次枚举出来。Python常用的列表生成式有一个致命的缺点就是定义即生成，非常的浪费空间和效率。

如果列表元素可以按照某种算法推算出来，那我们可以在循环的过程中不断推算出后续的元素，这样就不必创建完整的list，从而节省大量的空间。在Python中，这种一边循环一边计算的机制，称为生成器：generator。

要创建一个generator，最简单的方法是改造列表生成式：
  
```
a = [x * x for x in range(10)]
print(a)
b = (x * x for x in range(10))
print(b)

--------结果如下--------------
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
<generator object <genexpr> at 0x10557da50>
```
还有一个方法是生成器函数，通过def定义，然后使用yield来支持迭代器协议，比迭代器写起来更简单。

```
def spam():
    yield"first"
    yield"second"
    yield"third"

for x in spam():
    print(x)

-------结果如下---------
first
second
third
```
进行函数调用的时候，返回一个生成器对象。在使用next()调用的时候，遇到yield就返回，记录此时的函数调用位置，下次调用next()时，从断点处开始。

我们完全可以像使用迭代器一样使用 generator ，当然除了定义。定义一个迭代器，需要分别实现 iter() 方法和 next() 方法，但 generator 只需要一个小小的yield。

generator还有 send() 和 close() 方法，都是只能在next()调用之后，生成器处于挂起状态时才能使用的。

python是支持协程的，也就是微线程，就是通过generator来实现的。配合generator我们可以自定义函数的调用层次关系从而自己来调度线程。

<h2 id="3.python中装饰器的相关知识">3.Python中装饰器的相关知识</h2>
  
装饰器允许通过将现有函数传递给装饰器，从而<font color=DeepSkyBlue>向现有函数添加一些额外的功能</font>，该装饰器将执行现有函数的功能和添加的额外功能。

装饰器本质上还是一个函数，它可以让已有的函数不做任何改动的情况下增加功能。

接下来我们使用一些例子来具体说明装饰器的作用：

如果我们不使用装饰器，我们通常会这样来实现在函数执行前插入日志：

```
def foo():
    print('i am foo')

def foo():
    print('foo is running')
    print('i am foo')
```

虽然这样写是满足了需求，但是改动了原有的代码，如果有其他的函数也需要插入日志的话，就需要改写所有的函数，这样不能复用代码。

我们可以进行如下改写：

```
import logging

def use_log(func):
    logging.warning("%s is running" % func.__name__)
    func()

def bar():
    print('i am bar')

use_log(bar)    #将函数作为参数传入

-------------运行结果如下--------------
WARNING:root:bar is running
i am bar
```
这样写的确可以复用插入的日志，缺点就是显式的封装原来的函数，我们希望能隐式的做这件事。

我们可以用装饰器来写：

```
import logging

def use_log(func):
    def wrapper(*args, **kwargs):
        logging.warning('%s is running' % func.__name__)
        return func(*args, **kwargs)

    return wrapper


def bar():
    print('I am bar')


bar = use_log(bar)
bar()

------------结果如下------------
WARNING:root:bar is running
I am bar
```

其中，use_log函数就是装饰器，它把我们真正想要执行的函数bar()封装在里面，返回一个封装了加入代码的新函数，看起来就像是bar()被装饰了一样。

但是这样写还是不够隐式，我们可以通过@语法糖来起到bar = use_log(bar)的作用。

```
import logging

def use_log(func):
    def wrapper(*args, **kwargs):
        logging.warning('%s is running' % func.__name__)
        return func(*args, **kwargs)

    return wrapper


@use_log
def bar():
    print('I am bar')


@use_log
def haha():
    print('I am haha')


bar()
haha()

------------结果如下------------
WARNING:root:bar is running
I am bar
WARNING:root:haha is running
I am haha
```

这样子看起来就非常简洁，而且代码很容易复用。可以看成是一种智能的高级封装。


<h2 id="4.python的深拷贝与浅拷贝？">4.Python的深拷贝与浅拷贝？</h2>
  
在Python中，用一个变量给另一个变量赋值，其实就是给当前内存中的对象增加一个“标签”而已。

```
>>> a = [6, 6, 6, 6]
>>> b = a
>>> print(id(a), id(b), sep = '\n')
66668888
66668888

>>> a is b
True（可以看出，其实a和b指向内存中同一个对象。）
```

<font color=DeepSkyBlue>浅拷贝</font>是指创建一个新的对象，其内容是原对象中元素的引用（新对象与原对象共享内存中的子对象）。

注：浅拷贝和深拷贝的不同仅仅是对组合对象来说，所谓的组合对象就是包含了其他对象的对象，如列表，类实例等等。而对于数字、字符串以及其他“原子”类型，没有拷贝一说，产生的都是原对象的引用。

常见的浅拷贝有：切片操作、工厂函数、对象的copy()方法，copy模块中的copy函数。

```
>>> a = [6, 8, 9]
>>> b = list(a)
>>> print(id(a), id(b))
4493469248 4493592128    #a和b的地址不同

>>> for x, y in zip(a, b):
...     print(id(x), id(y))
... 
4489786672 4489786672
4489786736 4489786736
4489786768 4489786768
# 但是他们的子对象地址相同
```
  
从上面的例子中可以看出，a浅拷贝得到b，a和b指向内存中不同的list对象，但是他们的元素指向相同的int对象，这就是浅拷贝。

<font color=DeepSkyBlue>深拷贝</font>是指创建一个新的对象，然后递归的拷贝原对象所包含的子对象。深拷贝出来的对象与原对象没有任何关联。

深拷贝只有一种方式：copy模块中的deepcopy函数。

我们接下来用一个包含可变对象的列表来确切地展示浅拷贝和深拷贝的区别：
  
```
>>> a = [[6, 6], [8, 8], [9, 9]]
>>> b = copy.copy(a)   # 浅拷贝
>>> c = copy.deepcopy(a) # 深拷贝
>>> print(id(a), id(b)) # a和b地址不同
4493780304 4494523680
>>> for x, y in zip(a, b):   # a和b的子对象地址相同
...     print(id(x), id(y))
... 
4493592128 4493592128
4494528592 4494528592
4493779024 4493779024
>>> print(id(a), id(c))   # a和c不同
4493780304 4493469248
>>> for x, y in zip(a, c): # a和c的子对象地址也不同
...     print(id(x), id(y))
... 
4493592128 4493687696
4494528592 4493686336
4493779024 4493684896
```

<h2 id="5.python是解释语言还是编译语言？">5.Python是解释语言还是编译语言？</h2>
  
Python是解释语言。

<font color=DeepSkyBlue>解释语言</font>的优点是可移植性好，缺点是运行需要解释环境，运行起来比编译语言要慢，占用的资源也要多一些，代码效率低。

<font color=DeepSkyBlue>编译语言</font>的优点是运行速度快，代码效率高，编译后程序不可以修改，保密性好。缺点是代码需要经过编译才能运行，可移植性较差，只能在兼容的操作系统上运行。

![解释语言和编译语言的区别](https://files.mdnice.com/user/33499/aa37f783-c7da-45b8-9639-d97e1b74d017.png)


<h2 id="6.python的垃圾回收机制">6.Python的垃圾回收机制</h2>
  
在Python中，使用<font color=DeepSkyBlue>引用计数</font>进行垃圾回收；同时通过<font color=DeepSkyBlue>标记-清除算法</font>解决容器对象可能产生的循环引用问题；最后通过<font color=DeepSkyBlue>分代回收算法</font>提高垃圾回收效率。

<h2 id="7.python里有多线程吗？">7.Python里有多线程吗？</h2>
  
<font color=DeepSkyBlue>Python里的多线程是假的多线程</font>。

Python解释器由于设计时有GIL全局锁，导致了多线程无法利用多核，只有一个线程在解释器中运行。

对于I/O密集型任务，Python的多线程能起到作用，但对于CPU密集型任务，Python的多线程几乎占不到任何优势，还有可能因为争夺资源而变慢。

对所有面向I/O的（会调用内建的操作系统C代码的）程序来说，GIL会在这个I/O调用之前被释放，以允许其它的线程在这个线程等待I/O的时候运行。

如果是纯计算的程序，没有 I/O 操作，解释器会每隔 100 次操作就释放这把锁，让别的线程有机会执行（这个次数可以通过 sys.setcheckinterval 来调整）如果某线程并未使用很多I/O 操作，它会在自己的时间片内一直占用处理器和GIL。

缓解GIL锁的方法：多进程和协程（协程也只是单CPU，但是能减小切换代价提升性能）

<h2 id="8.python中range和xrange的区别？">8.Python中range和xrange的区别？</h2>
  
首先，xrange函数和range函数的用法完全相同，不同的地方是xrange函数生成的不是一个list对象，而是一个生成器。

要生成很大的数字序列时，使用xrange会比range的性能优很多，因为其不需要一上来就开辟很大的内存空间。
  
```
Python 2.7.15 | packaged by conda-forge | (default, Jul  2 2019, 00:42:22) 
[GCC 4.2.1 Compatible Clang 4.0.1 (tags/RELEASE_401/final)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> range(10)
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
>>> xrange(10)
xrange(10)
>>> list(xrange(10))
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```
xrange函数和range函数一般都用在循环的时候。具体例子如下所示：

```
>>> for i in range(0,7):
...     print(i)
... 
0
1
2
3
4
5
6

>>> for i in xrange(0,7):
...     print(i)
... 
0
1
2
3
4
5
6
```
在Python3中，xrange函数被移除了，只保留了range函数的实现，但是此时range函数的功能结合了xrange和range。并且range函数的类型也发生了变化，在Python2中是list类型，但是在Python3中是range序列的对象。

<h2 id="9.python中列表和元组的区别？">9.Python中列表和元组的区别？</h2>

1. <font color=DeepSkyBlue>列表是可变的</font>，在创建之后可以对其进行任意的修改。

2. <font color=DeepSkyBlue>元组是不可变的</font>，元组一旦创建，便不能对其进行更改，可以元组当作一个只读版本的列表。

3. 元组无法复制。

4. Python将低开销的较大的块分配给元组，因为它们是不可变的。对于列表则分配小内存块。与列表相比，元组的内存更小。当你拥有大量元素时，元组比列表快。

<h2 id="10.python中dict（字典）的底层结构？">10.Python中dict（字典）的底层结构？</h2>
  
Python的dict（字典）为了支持快速查找使用了哈希表作为底层结构，哈希表平均查找时间复杂度为O(1)。CPython 解释器使用二次探查解决哈希冲突问题。

<h2 id="11.常用的深度学习框架有哪些，都是哪家公司开发的？">11.常用的深度学习框架有哪些，都是哪家公司开发的？</h2>
  
1. PyTorch：Facebook

2. TensorFlow：Google

3. Keras：Google

4. MxNet：Dmlc社区

5. Caffe：UC Berkeley

6. PaddlePaddle：百度

<h2 id="12.pytorch动态图和tensorflow静态图的区别？">12.PyTorch动态图和TensorFlow静态图的区别？</h2>
  
PyTorch动态图：计算图的运算与搭建同时进行；其较灵活，易调节。
  
TensorFlow静态图：计算图先搭建图，后运算；其较高效，不灵活。

![](https://files.mdnice.com/user/33499/877c601c-9522-4caa-9b67-2f3591bee071.png)

<h2 id="13.python中assert的作用？">13.Python中assert的作用？</h2>

Python中assert（断言）用于判断一个表达式，在表达式条件为$false$的时候触发异常。

断言可以在条件不满足程序运行的情况下直接返回错误，而不必等待程序运行后出现崩溃的情况。

Rocky直接举一些例子：

```python
>>> assert True 
>>> assert False
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AssertionError
>>> assert 1 == 1
>>> assert 1 == 2
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AssertionError
>>> assert 1 != 2
```
  
<h2 id="14.python中互换变量有不用创建临时变量的方法吗？">14.Python中互换变量有不用创建临时变量的方法吗？</h2>

在Python中，当我们想要互换两个变量的值或将列表中的两个值交换时，我们可以使用如下的格式进行，不需要创建临时变量：

```python
x, y = y, x
```

这么做的原理是什么呢？

首先一般情况下Python是从左到右解析一个语句的，但在赋值操作的时候，因为是右值具有更高的计算优先级，所以需要从右向左解析。

对于上面的代码，它的执行顺序如下：

先计算右值$y , x$(这里是简单的原值，但可能会有表达式或者函数调用的计算过程)， 在内存中创建元组(tuple)，存储$y, x$分别对应的值；计算左边的标识符，元组被分别分配给左值，通过解包(unpacking)，元组中第一个标示符对应的值$(y)$，分配给左边第一个标示符$(x)$，元组中第二个标示符对应的值$(x)$，分配给左边第二个标示符$(y)$，完成了$x$和$y$的值交换。

<h2 id="15.python中的主要数据结构都有哪些？">15.Python中的主要数据结构都有哪些？</h2>

1. 列表（list）
2. 元组（tuple）
3. 字典（dict）
4. 集合（set）
  
<h2 id="16.python中的可变对象和不可变对象？">16.Python中的可变对象和不可变对象？</h2>

可变对象与不可变对象的区别在于对象本身是否可变。

可变对象：list（列表） dict（字典） set（集合）

不可变对象：tuple（元组） string（字符串） int（整型） float（浮点型） bool（布尔型）

<h2 id="17.python中的none代表什么？">17.Python中的None代表什么？</h2>

None是一个特殊的常量，表示空值，其和False，0以及空字符串不同，它是一个特殊Python对象, None的类型是NoneType。

None和任何其他的数据类型比较返回False。

```
>>> None == 0
False
>>> None == ' '
False
>>> None == None
True
>>> None == False
False
```

我们可以将None复制给任何变量，也可以给None赋值。
  
<h2 id="18.python中args和kwargs的区别？">18.Python中$*args$和$**kwargs$的区别？</h2>

$*args$和$**kwargs$主要用于函数定义。我们可以将不定数量的参数传递给一个函数。

<font color=DeepSkyBlue>这里的不定的意思是</font>：预先并不知道函数使用者会传递多少个参数, 所以在这个场景下使用这两个关键字。
<h3 id="args">$*args$</h3>

$*args$是用来发送<font color=DeepSkyBlue>一个非键值对的可变数量的参数列表</font>给一个函数。

我们直接看一个例子：

```python
def test_var_args(f_arg, *argv):
    print("first normal arg:", f_arg)
    for arg in argv:
        print("another arg through *argv:", arg)

test_var_args('hello', 'python', 'ddd', 'test')

-----------------结果如下-----------------------
first normal arg: hello
another arg through *argv: python
another arg through *argv: ddd
another arg through *argv: test
```

<h3 id="kwargs">$**kwargs$</h3>

$**kwargs$允许我们<font color=DeepSkyBlue>将不定长度的键值对, 作为参数传递给一个函数</font>。如果我们想要在一个函数里处理带名字的参数, 我们可以使用$**kwargs$。

我们同样举一个例子：

```python
def greet_me(**kwargs):
    for key, value in kwargs.items():
        print("{0} == {1}".format(key, value))

greet_me(name="yasoob")

-----------结果如下-------------
name == yasoob
```

<h2 id="19.python中numpy的broadcasting机制？">19.Python中Numpy的broadcasting机制？</h2>

Python的Numpy库是一个非常实用的数学计算库，其broadcasting机制给我们的矩阵运算带来了极大地方便。

我们先看下面的一个例子：

```python
>>> import numpy as np
>>> a = np.array([1,2,3])
>>> a
array([1, 2, 3])
>>> b = np.array([6,6,6])
>>> b
array([6, 6, 6])
>>> c = a + b
>>> c
array([7, 8, 9])
```

上面的代码其实就是把数组$a$和数组$b$中同样位置的每对元素相加。这里$a$和$b$是相同长度的数组。

如果两个数组的长度不一致，这时候broadcasting就可以发挥作用了。

比如下面的代码：

```python
>>> d = a + 5
>>> d
array([6, 7, 8])
```

broadcasting会把$5$扩展成$[5,5,5]$，然后上面的代码就变成了对两个同样长度的数组相加。示意图如下（broadcasting不会分配额外的内存来存取被复制的数据，这里只是方面描述）：

![](https://img-blog.csdnimg.cn/20200902094314838.png#pic_center)

我们接下来看看多维数组的情况：

```python
>>> e
array([[1., 1., 1.],
       [1., 1., 1.],
       [1., 1., 1.]])
>>> e + a
array([[2., 3., 4.],
       [2., 3., 4.],
       [2., 3., 4.]])
```

在这里一维数组被扩展成了二维数组，和$e$的尺寸相同。示意图如下所示：

![](https://img-blog.csdnimg.cn/2020090209463749.png#pic_center)

我们再来看一个需要对两个数组都做broadcasting的例子：

```python
>>> b = np.arange(3).reshape((3,1))
>>> b
array([[0],
       [1],
       [2]])
>>> b + a
array([[1, 2, 3],
       [2, 3, 4],
       [3, 4, 5]])
```

在这里$a$和$b$都被扩展成相同的尺寸的二维数组。示意图如下所示：

![](https://img-blog.csdnimg.cn/20200902094859308.png#pic_center)

**总结broadcasting的一些规则：**

1. 如果两个数组维数不相等，维数较低的数组的shape进行填充，直到和高维数组的维数匹配。
2. 如果两个数组维数相同，但某些维度的长度不同，那么长度为1的维度会被扩展，和另一数组的同维度的长度匹配。
3. 如果两个数组维数相同，但有任一维度的长度不同且不为1，则报错。

```python
>>> a = np.arange(3)
>>> a
array([0, 1, 2])
>>> b = np.ones((2,3))
>>> b
array([[1., 1., 1.],
       [1., 1., 1.]])
>>> a.shape
(3,)
>>> a + b
array([[1., 2., 3.],
       [1., 2., 3.]])
```

接下来我们看看报错的例子：

```python
>>> a = np.arange(3)
>>> a
array([0, 1, 2])
>>> b = np.ones((3,2))
>>> b
array([[1., 1.],
       [1., 1.],
       [1., 1.]])
>>> a + b
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: operands could not be broadcast together with shapes (3,) (3,2)
```
  
<h2 id="20.python中的实例方法、静态方法和类方法三者区别？">20.Python中的实例方法、静态方法和类方法三者区别？</h2>

<font color=DeepSkyBlue>不用@classmethod和@staticmethod修饰的方法为实例方法</font>。在类中定义的方法默认都是实例方法。实例方法最大的特点是它至少要包含一个self参数，用于绑定调用此方法的实例对象，实例方法通常可以用类对象直接调用。

<font color=DeepSkyBlue>采用@classmethod修饰的方法为类方法</font>。类方法和实例方法相似，它至少也要包含一个参数，只不过类方法中通常将其命名为cls，Python会自动将类本身绑定给cls参数。我们在调用类方法时，无需显式为cls参数传参。

<font color=DeepSkyBlue>采用@staticmethod修饰的方法为静态方法</font>。静态方法没有类似self、cls这样的特殊参数，因此Python的解释器不会对它包含的参数做任何类或对象的绑定。也正因为如此，类的静态方法中无法调用任何类属性和类方法。

<h2 id="21.python中常见的切片操作">21.Python中常见的切片操作</h2>

[:n]代表列表中的第一项到第n项。我们看一个例子：

```python
example = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

print(example[:6])

---------结果---------
[1, 2, 3, 4, 5, 6]
```

[n:]代表列表中第n+1项到最后一项：

```python
example = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

print(example[6:])

---------结果---------
[7, 8, 9, 10]
```

[-1]代表取列表的最后一个元素：

```python
example = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

print(example[-1])

---------结果---------
10
```

[:-1]代表取除了最后一个元素的所有元素：

```python
example = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

print(example[:-1])

---------结果---------
[1, 2, 3, 4, 5, 6, 7, 8, 9]
```

[::-1]代表取整个列表的相反列表：

```python
example = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

print(example[::-1])

---------结果---------
[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
```

[1:]代表从第二个元素意指读取到最后一个元素：

```python
example = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

print(example[1:])

---------结果---------
[2, 3, 4, 5, 6, 7, 8, 9, 10]
```

[4::-1]代表取下标为4（即第五个元素）的元素和之前的元素反转读取：

```python
example = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

print(example[4::-1])

---------结果---------
[5, 4, 3, 2, 1]
```

<h2 id="22.python中如何进行异常处理？">22.Python中如何进行异常处理？</h2>

一般情况下，在Python无法正常处理程序时就会发生一个异常。<font color=DeepSkyBlue>异常在Python中是一个对象，表示一个错误</font>。当Python脚本发生异常时我们需要捕获处理它，否则程序会终止执行。

捕捉异常可以使用try，except和finally语句。

try和except语句用来检测try语句块中的错误，从而让except语句捕获异常信息并处理。

```
try:
    6688 / 0
except:
    '''异常的父类，可以捕获所有的异常'''
    print "0不能被除"
else:
    '''保护不抛出异常的代码'''
    print "没有异常"
finally:
    print "最后总是要执行我"
```

<h2 id="23.python中remove，del以及pop之间的区别？">23.Python中remove，del以及pop之间的区别？</h2>

remove，del以及pop都可以用于删除列表、字符串等里面的元素，但是具体用法并不相同。

1. remove是剔除第一个匹配的值。
2. del是通过索引来删除当中的元素。
3. pop是通过索引来删除当中的元素，并且返回该元素；若括号内不添加索引值，则默认删除最后一个元素。

```
>>> a = [0, 1, 2, 1, 3] 
>>> a.remove(1) 
>>> a 
[0, 2, 1, 3] 

>>> a = [0, 1, 2, 1, 3] 
>>> del a[1] 
[0, 2, 1, 3] 

>>> a = [0, 1, 2, 1, 3] 
>>> a.pop(1) 
1 
>>> a 
[0, 2, 1, 3] 
```

<h2 id="24.Python中PIL和OpenCV处理图像的区别？">24.Python中PIL和OpenCV处理图像的区别？</h2>

Python中的Pillow/PIL（Python Imaging Library）和OpenCV（Open Source Computer Vision Library）都是强大的图像处理库，但它们各有特点和优势，适用于不同的应用场景。

### 以下是两者之间的详细对比：

1. 数据类型：Pillow读取图像时返回的是PIL.Image.Image类型的对象。而OpenCV读取图像时返回的是NumPy数组。
2. 颜色格式：OpenCV默认使用BGR颜色格式读取图像。而Pillow使用的是更常见的RGB格式。
3. 应用专长：Pillow主要专注于图像的加载、处理和保存，提供了广泛的图像处理能力，如图像缩放、裁剪、过滤、图像合成等，更加侧重于图像处理的基本操作和图像的IO操作。OpenCV是一个专注于实时计算机视觉的库，它的功能不仅限于图像处理，还包括视频处理、人脸识别、对象检测、复杂图像分析等，适用于需要快速有效处理大量数据的应用。
4. 性能和效率：对于高性能的实时计算机视觉任务，OpenCV通常比Pillow表现得更好，因为OpenCV底层使用了优化的C++代码，同时支持多线程和GPU加速。在常规图像处理任务中，Pillow更易于使用，且对于基本的图像处理任务已经足够快，并且易于集成到Python应用中。

### 总结

Pillow适合于需要处理图像文件、执行基本图像处理任务的应用，而OpenCV适合于需要实施复杂的图像分析、计算机视觉处理或实时视频处理的项目。选择哪个库取决于你的具体需求、项目复杂度以及性能要求。

<h2 id="25.Python中全局变量与局部变量之间的区别？">25.Python中全局变量与局部变量之间的区别？</h2>

在Python中，**全局变量和局部变量的区别主要体现在变量的作用域、声明位置以及在程序中的可访问性上**。理解这些差异有助于我们更好地管理数据的流向和变量的生命周期，防止不必要的编程错误。

### 全局变量

1. **定义与作用域**：
   - 全局变量是在函数外部定义的，其作用域覆盖了整个代码文件/定义它的模块。
   - 在任何函数内部和外部都可以访问全局变量（除非被局部作用域的同名变量遮蔽）。

2. **使用场景**：
   - 当多个函数需要访问同一数据时，可以使用全局变量。
   - 用于定义整个应用程序可能需要的配置信息或共享数据。

### 局部变量

1. **定义与作用域**：
   - 局部变量是在函数内部/代码块中定义的，它只在定义它的函数或代码块内部有效。
   - 函数或代码块执行完毕后，局部变量的生命周期结束，它们所占用的内存也随之释放。

2. **使用场景**：
   - 当变量的用途仅限于特定函数或代码块时，应使用局部变量。
   - 局部变量有助于保持函数的独立性，使函数更易理解和重用。

3. **优势**：
   - 局部变量避免了函数间的数据交互问题，减少了代码的耦合度。
   - 使用局部变量可以提高代码的可读性和维护性。

### 访问和修改全局变量

在Python中，如果你需要在一个函数内部修改全局变量，你必须使用`global`这个全局关键字来进行声明：

```python
x = 10  # 全局变量

def update():
    global x
    x = 20  # 修改全局变量

def print_x():
    print(x)  # 访问全局变量

update()
print_x()  # 输出: 20
```

如果不使用`global`全局关键字，对全局变量的修改实际上会创建一个同名的新的局部变量，而不会改变全局变量的值。


<h2 id="26.Python中name==main?">26.Python中`if '__name__ == '__main__'`的作用?</h2>

在Python中，`if __name__ == '__main__'`用于确定当前Python脚本是需要被直接运行还是被另一个Python脚本导入。

每个Python脚本中都有`__name__`这个内置变量，它在脚本被直接运行时被设置为`'__main__'`。而当其被另一个脚本导入时，`__name__` 被设置为脚本的名字。


例如，假设有一个模块 `example.py`：

```python
def my_function():
    print("This function is defined in the module.")

if __name__ == '__main__':
    my_function()
```

当你直接运行 `example.py` 时，`__name__` 会是 `'__main__'`，所以 `my_function()` 会被执行。但如果你从另一个模块导入 `example.py`：

```python
import example

example.my_function()  # 调用函数
```

在这种情况下，`__name__` 会是 `'example'`，所以 `if __name__ == '__main__'` 块下的代码不会被执行，你只会得到 `my_function()` 的定义和功能。


<h2 id="27.Python中assert的作用?">27.Python中assert的作用?</h2>

在Python中，`assert` 语句用于断言某个条件是真的。如果条件为真，程序会继续运行；如果条件为假，则会触发一个 `AssertionError` 异常。这种方式主要用于调试阶段，确保代码在特定条件下正确运行，或用来作为程序内部的自检手段。

`assert` 语句的基本语法如下：
```python
assert condition, error_message
```
- `condition`：这是需要检查的表达式，结果应为布尔值。
- `error_message`：这是可选的，如果提供，当条件不满足（即为`False`）时，将显示此错误消息。

### 使用示例

假设你正在开发一个函数，该函数必须接受正整数作为输入。你可以使用 `assert` 语句来确保输入是正整数：

```python
def print_inverse(number):
    assert number > 0, "The number must be positive"
    print(1 / number)

print_inverse(5)  # 正常运行
print_inverse(-1)  # 抛出 AssertionError: The number must be positive
```

### 注意事项
- `assert` 语句可以被全局解释器选项 `-O` （优化模式）和其他方法禁用。当Python解释器以优化模式运行时，所有的`assert`语句都会被忽略。因此，通常不建议在生产代码中使用`assert`来做输入验证或处理程序的核心逻辑。
- 适合使用`assert`的场景包括：测试代码的健壮性、检查函数的输入参数是否符合预期。

总之，`assert`是一个有用的调试工具，可以帮助开发者在开发过程中快速定位问题。

<h2 id="28.python中如何无损打开图像，并无损保存图像?">28.python中如何无损打开图像，并无损保存图像?</h2>

在Python中，如果我们想无损地打开和保存图像，关键是选择支持无损保存的图像格式，如PNG、TIFF或BMP。使用Python的Pillow库可以方便地处理这种任务。下面Rocky将详细说明如何使用Pillow无损地打开和保存图像。

### 使用Pillow无损打开和保存图像

1. **打开图像**：使用`Image.open()`函数打开图像文件。这个函数不会修改图像数据，因此打开图像本身是无损的。

2. **保存图像**：使用`Image.save()`函数，并确保选择无损格式，如PNG。

以下是一个具体的示例：

```python
from PIL import Image

# 打开图像文件
img = Image.open('example.jpg')  # 假设原始文件是JPEG格式

# 执行一些图像处理操作（可选）
# 注意，确保这些操作是无损的，如大小调整、裁剪等

# 保存图像为PNG格式，这是无损压缩
img.save('output.png', 'PNG')

# 或者用这种方式进行无损保存
img.save('path_to_save_image.png', format='PNG', optimize=True)
```

### 注意事项
- PNG使用的是无损压缩算法，这意味着图像的所有信息在压缩和解压过程中都被完整保留，不会有任何质量损失。
- 当从有损压缩格式（如JPEG）转换为无损格式（如PNG）时，虽然保存过程是无损的，但原始从JPEG格式读取的图像可能已经丢失了一些信息。因此，最佳实践是始终从无损格式源文件开始操作，以保持最高质量。
- 如果我们的图像已经在一个无损格式（如PNG），你只需重新保存它，同样选择无损格式，即可保持图像质量不变。
- 无损保存与 `optimize=True` 选项：`optimize=True`选项会尝试找到更为压缩的存储方式来保存文件，但不会影响图像的质量。具体来说，它在保存文件时会尝试优化图像数据的存储方式，比如重新排列文件中的色块和使用更有效的编码方法，但仍然保持图像数据的完整性和质量。因此，即使使用了 `optimize=True` 选项，保存的PNG文件也是无损的，不会有质量损失。这使得PNG格式非常适合需要无损压缩的应用，如需要频繁编辑和保存的图像处理任务。

### 处理其他图像格式
对于其他格式如TIFF或BMP，Pillow同样支持无损操作，保存方法类似，只需改变保存格式即可：

```python
# 保存为TIFF格式
img.save('output.tiff', 'TIFF')

# 保存为BMP格式
img.save('output.bmp', 'BMP')
```

使用Pillow库，我们可以轻松地在Python中进行无损图像处理。它提供了广泛的功能，能够处理几乎所有常见的图像格式，并支持复杂的图像处理任务。

<h2 id="29.PyTorch中张量操作Clone与Detach的区别?（腾讯实习二面）">29.PyTorch中张量操作Clone与Detach的区别？</h2>

clone方法:

创建了张量的一个新的副本，这个副本具有与原始张量相同的数据和形状，但它们是存储在内存中的两个独立实体。对克隆张量的修改不会影响到原始张量。clone通常用于需要保留原始数据不变时创建张量的独立副本。

    import torch
    x = torch.tensor([1, 2, 3])
    y = x.clone()  # y 是x的一个副本，对y的修改不会影响到x

detach方法:

从当前的计算图中分离出一个张量，使其成为一个不需要梯度的张量。这意味着，经过detach操作的张量将不会参与梯度传播，不会在反向传播中更新。detach通常用于阻止梯度传播，或者在进行推断时将张量从模型的参数中分离出来。

    import torch
    x = torch.tensor([1, 2, 3], requires_grad=True)
    y = x.detach()  # y 不需要梯度，对y的操作不会引起梯度计算
主要区别

目的：clone的目的是创建数据的副本，而detach的目的是分离张量以停止梯度计算。

内存使用：clone会创建数据的一个完整副本，因此会使用更多的内存。detach则不会复制数据，只是创建了一个新的视图，通常不会增加额外的内存使用。

梯度：clone得到的副本可以有与原始张量相同的requires_grad属性，而detach得到的张量总是没有梯度的。

数据一致性：clone创建的是数据的一致副本，对副本的修改不会反映到原始张量上。detach操作的张量与原始张量数据上是一致的，对副本修改会反映原始张量，但不参与梯度计算。


<h2 id="30.Python多进程中的fork和spawn模式有什么区别？">30.Python多进程中的fork和spawn模式有什么区别？</h2>

1. windows和MacOS中默认为spawn模式，unix系统默认为fork模式，其中windows只支持spawn，unix同时支持两者；
2. spawn模式不会继承父进程的资源，而是从头创建一个全新的进程，启动较慢；
3. fork模式会继承父进程的资源，即通过复制父进程资源来快速创建子进程，启动较快；


<h2 id="31.什么是Python中的推导式？Python的推导式一共有多少种？">31.什么是Python中的推导式？Python的推导式一共有多少种？</h2>

Python中的推导式（comprehensions）是一种简洁、灵活且高效的构建Python数据结构的方法，包括列表、字典、集合和生成器。推导式允许以表达式的形式快速生成新的数据结构，同时在创建过程中可以直接应用条件筛选或操作。下面详细介绍Python中四种主要的推导式：

### 1. 列表推导式（List Comprehensions）

**功能**：用于创建列表，可以通过应用表达式自动处理并生成新列表。

**基本语法**：
```python
[expression for item in iterable if condition]
```

- `expression` 是对 `item` 的操作或者应用表达式。
- `item` 是从 `iterable` 中逐个取出的元素。
- `condition` 是一个可选的条件语句，用于过滤。

**示例**：
```python
# 生成0-9每个数字的平方的列表
squares = [x**2 for x in range(10)]

# 结果
{0, 1, 64, 4, 36, 9, 16, 49, 81, 25}
```

### 2. 字典推导式（Dictionary Comprehensions）

**功能**：用于创建字典，允许通过迭代可迭代对象来生成键值对。

**基本语法**：
```python
{key_expression : value_expression for item in iterable if condition}
```

- `key_expression` 表示字典的键的表达式。
- `value_expression` 表示字典的值的表达式。
- `item` 是从 `iterable` 中逐个取出的元素。
- `condition` 是一个可选的条件语句，用于过滤。

**示例**：
```python
# 使用数字作为键，其平方作为值
squares_dict = {x: x**2 for x in range(5)}

# 结果
{0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
```

### 3. 集合推导式（Set Comprehensions）

**功能**：用于创建集合，类似于列表推导式，但结果是一个集合，自动去重。

**基本语法**：
```python
{expression for item in iterable if condition}
```

- `expression` 是对 `item` 的操作或者应用表达式。
- `item` 是从 `iterable` 中逐个取出的元素。
- `condition` 是一个可选的条件语句，用于过滤。

**示例**：
```python
# 创建一个包含0-9每个数字平方的集合
square_set = {x**2 for x in range(10)}

# 结果
{0, 1, 64, 4, 36, 9, 16, 49, 81, 25}
```

### 4. 生成器推导式（Generator Expressions）

**功能**：生成器推导式是一种类似于列表推导式的结构，用于创建生成器（一种迭代器），不会一次性生成所有元素，而是按需产生，节约内存。

**基本语法**：
```python
(expression for item in iterable if condition)
```

- `expression` 是对 `item` 的操作或者应用表达式。
- `item` 是从 `iterable` 中逐个取出的元素。
- `condition` 是一个可选的条件语句，用于过滤。

**示例**：
```python
# 创建一个生成器，包含0-9每个数字的平方
square_gen = (x**2 for x in range(10))

# 结果
<generator object <genexpr> at 0x7f0827a98660>
```

推导式提供了一种高效和直观的方式来创建数据结构，使代码更加简洁易读。在合适的情况下，使用推导式可以有效提升编程效率和执行性能。


<h2 id="32.python中一共都有哪些数据结构？">32.python中一共都有哪些数据结构？</h2>

Python提供了一系列内置的数据结构，这些数据结构非常强大和灵活，可以用来处理各种不同类型的数据。这些数据结构包括列表、元组、字典、集合，以及通过标准库可用的更多高级数据结构如队列和堆。下面是这些主要数据结构的详细介绍：

### 1. 列表（List）
列表是Python中最常用的数据结构之一，它是一个有序的集合，可以包含任何类型的对象：数字、字符串、甚至其他列表。列表是可变的，这意味着它们可以被修改。

**基本操作**：
- 创建列表：`my_list = [1, 2, 3]`
- 添加元素：`my_list.append(4)`
- 删除元素：`del my_list[0]`
- 切片操作：`my_list[1:3]`

### 2. 元组（Tuple）
元组与列表类似，但它们是不可变的。这意味着一旦创建了元组，就不能修改其内容。元组通常用于保护数据不被更改，并且可以作为字典键使用，而列表则不能。

**基本操作**：
- 创建元组：`my_tuple = (1, 2, 3)`
- 访问元素：`my_tuple[1]`
- 切片操作：`my_tuple[1:2]`

### 3. 字典（Dictionary）
字典是一种关联数组或哈希表，它由键值对组成。字典中的键必须是唯一的，并且必须是不可变类型，如字符串或元组。字典在查找、添加和删除操作上非常高效。

**基本操作**：
- 创建字典：`my_dict = {'key': 'value'}`
- 访问元素：`my_dict['key']`
- 添加或修改元素：`my_dict['new_key'] = 'new_value'`
- 删除元素：`del my_dict['key']`

### 4. 集合（Set）
集合是一个无序的元素集，提供了强大的成员测试和删除重复元素的功能。集合中的元素必须是不可变类型，并且集合本身是可变的。

**基本操作**：
- 创建集合：`my_set = {1, 2, 3}`
- 添加元素：`my_set.add(4)`
- 删除元素：`my_set.remove(2)`
- 成员测试：`1 in my_set`

### 高级数据结构
Python的标准库还提供了一些高级数据结构，这些结构在`collections`模块和其他模块中定义。

#### 队列（Queue）
队列是一种先进先出的数据结构，标准库中的`queue.Queue`用于多线程编程中的线程安全的队列操作。

#### 双端队列（Deque）
`collections.deque`提供了一个双端队列，支持从任一端添加或删除元素的高效操作。

#### 计数器（Counter）
`collections.Counter`是一个简单的计数器，用于计数可哈希对象。

#### 有序字典（OrderedDict）
`collections.OrderedDict`是一个保持元素插入顺序的字典。

#### 堆（Heap）
模块`heapq`提供了堆队列算法，特别是优先级队列的实现。

这些数据结构使Python在处理各种数据时变得非常灵活和强大。选择正确的数据结构可以显著提高程序的效率和性能。

<h2 id="33.python中index使用注意事项">33.python中index使用注意事项</h2>

### 作用
Python list.index方法返回某一个值的元素位于列表中的索引。
### 语法和参数
```
list.index(element, start, end)
```
element:要查询的元素值，不可省略的参数，可以是任意类型的对象实例<br>
start:可选整型参数，查找的起始位置。<br>
end:可选整型参数，查找的结束位置。<br>
返回值：int类型，参数在list中的索引（*靠近表头的索引）

注意事项：
当查询参数在列表中存在多个，index方法只会返回靠近表头的索引，即第一次查到的参数，并不会将所有匹配值的索引全部返回。
在无法确定列表中元素是否重复的情况下，不建议使用此方法查询索引，建议使用range(len(list))的方法。

<h2 id="34.python中@staticmethod和@classmethod使用注意事项">34.python中@staticmethod和@classmethod使用注意事项</h2>

### @staticmethod

1) 静态方法：staticmethod将一个普通函数嵌入到类中，使其成为类的静态方法。静态方法不需要一个类实例即可被调用，同时它也不需要访问类实例的状态。
2) 参数：静态方法可以接受任何参数，但通常不使用self或cls作为第一个参数。
3) 访问：由于静态方法不依赖于类实例的状态，因此它们不能修改类或实例的状态。
4) 用途：当函数与类相关，但其操作不依赖于类状态时，适合使用静态方法。
### @classmethod
1) 类方法：classmethod将一个方法绑定到类而非类的实例。类方法通常用于操作类级别的属性。
2) 参数：类方法至少有一个参数，通常命名为cls，它指向类本身。
3) 访问：类方法可以修改类的状态，但不能修改实例的状态。
4) 用途：当方法需要访问或修改类属性，或者需要通过类来创建实例时，适合使用类方法。

### 使用场景
- 当方法不需要访问任何属性时，使用staticmethod。
- 当方法操作的是类属性而不是实例属性时，使用classmethod。

### 代码示例
```python
class MyClass:
    class_variable = "I'm a class variable."

    def __init__(self, value):
        self.instance_variable = value

    @staticmethod
    def static_method():
        return "Static method called."

    @classmethod
    def class_method(cls):
        return f"Class method called. Class variable: {cls.class_variable}"

# 调用静态方法
MyClass.static_method()

# 调用类方法
MyClass.class_method()

```
### 问题
在使用falsk-restful这个框架进行模型部署调用时，发现模型推理时间很快，但是完整的一次请求过程非常耗时。在debug的过程中发现，每次请求调用api接口时，模型的推理类都会被实例化，推理类在构造的时候，会在初始化中加载模型，加载模型的过程是耗时较长的。
### fixbug
```python
classs Infer(object):
    def __init__(self, cfg: dict)->None:
        self.cfg = cfg
        self.load_model(self.cfg)

    @classmethod
    def load_model(cls, cfg: dict):
        cls.cfg = cfg
        if not hasattr(cls, "model"):
            cls.model = torch.load("xxx.pt")
```
通过@classmethod方法初始化模型的加载，相当于创建了一个全局变量，在后续的请求调用中，不会一直重复加载


<h2 id="35.Python中函数传参时会改变参数本身吗？">35.Python中函数传参时会改变参数本身吗？</h2>

**在Python中，函数传参是否会改变参数本身取决于参数的数据类型和传递方式**。这涉及到Python的参数传递机制，通常被称为“传引用的值”（pass-by-reference value）或者“传对象引用”（pass-by-object-reference）。这里的一些基本规则和示例将帮助大家理解这一概念：

### 可变与不可变对象
1. **不可变对象**：包括整数、浮点数、字符串、元组等。这些类型的数据不允许被修改。
   - 当我们传递一个不可变对象给函数时，虽然函数内部可以使用这个对象的值，但任何试图改变该对象的操作都将在本地创建一个新对象。外部原始对象不会被改变。

   ```python
   def modify(x):
       x = 10
       return x
   
   a = 5
   modify(a)
   print(a)  # 输出 5，原始值未改变
   ```

2. **可变对象**：包括列表、字典、集合等。这些类型的数据可以被修改。
   - 当你传递一个可变对象给函数时，函数内部对这个对象的任何修改都会反映到原始对象上。

   ```python
   def modify(lst):
       lst.append(3)
   
   my_list = [1, 2]
   modify(my_list)
   print(my_list)  # 输出 [1, 2, 3]，原始列表被修改
   ```

### 函数参数的工作方式
- 在Python中，所有的函数参数都是按“引用传递”的。但实际上，这意味着当对象传递给函数时，传递的是对象的引用（内存地址），而不是对象的实际拷贝。对于不可变对象，由于不能被改变，所以任何修改都会导致创建一个新的本地对象；而对于可变对象，则可以在原地址上进行修改。
- 对于列表和字典这样的可变对象，如果你不希望函数中的操作影响到原始数据，你可以传递一个拷贝给函数，而不是原始对象本身。

```python
import copy

def modify(lst):
    lst.append(3)

original_list = [1, 2]
new_list = copy.deepcopy(original_list)
modify(new_list)
print(original_list)  # 输出 [1, 2]
print(new_list)       # 输出 [1, 2, 3]
```

### 总结
在Python中，函数的行为取决于传入参数的类型。不可变对象（如整数、字符串和元组）不会在函数调用中被修改，而可变对象（如列表和字典）可以被修改。了解这些差异有助于我们更好地管理函数中数据的状态和行为。


<h2 id="36.什么是python的全局解释器锁GIL？">36.什么是python的全局解释器锁GIL？</h2>

在Python中，全局解释器锁（Global Interpreter Lock，简称GIL）是一个重要的概念，特别是在涉及多线程执行时。GIL 是一个互斥锁，保证同一时间内只有一个线程可以执行Python字节码。简而言之，尽管在多核处理器上运行，Python 的标准实现 CPython 在执行多线程应用时，并不能有效地利用多核处理器的优势。

### GIL 的目的
1. **简化内存管理**：CPython 使用引用计数来管理内存，这种方法在多线程环境中容易产生问题。GIL 通过确保一次只有一个线程运行，避免了常见的并发访问问题，如竞态条件。
2. **保护CPython的内部数据结构**：没有GIL，程序员必须采用其他并发控制技术，如细粒度锁，这可能会使CPython的实现更复杂。

### GIL 的影响
尽管GIL简化了内存管理和内部数据结构的保护，但它也限制了Python程序在多核处理器上的并行执行能力：
- **多线程局限性**：在CPU密集型程序中，GIL成为性能瓶颈，因为线程不能在多个CPU核心上同时执行计算任务。
- **I/O密集型应用的表现更好**：I/O操作不需要大量CPU计算，线程可能会在等待I/O操作完成时释放GIL，从而让其他线程有机会执行。

### 绕过GIL
虽然GIL在多线程编程中存在局限性，但Python社区提供了多种方法来绕过这些限制：
1. **使用多进程**：通过`multiprocessing`模块，可以创建多个进程，每个进程拥有自己的Python解释器和内存空间，从而不受GIL的限制。
2. **使用其他实现**：如Jython和IronPython，这些Python实现没有GIL，可以更好地利用多核处理器。
3. **使用特定库**：一些库设计可以在底层进行多线程或多进程操作，从而绕过GIL的限制，例如NumPy和其他与C语言库交互的扩展。


<h2 id="37.什么是python的字符串格式化技术？">37.什么是python的字符串格式化技术？</h2>

在Python中，字符串格式化是一项重要的技能，能帮助我们高效地生成和处理字符串。Python提供了多种字符串格式化的方法，包括旧式的百分号（`%`）格式化、新式的`str.format()`方法以及最新的f-string（格式化字符串字面量）。下面Rocky将详细讲解这些方法，让大家更好地理解。

### 1. 百分号（%）格式化

这是Python中最古老的字符串格式化方法，使用`%`符号进行占位符替换。

#### 基本用法：
```python
name = "Alice"
age = 30
formatted_string = "Name: %s, Age: %d" % (name, age)
print(formatted_string)  # 输出: Name: Alice, Age: 30
```
- `%s` 用于字符串
- `%d` 用于整数
- `%f` 用于浮点数

#### 控制浮点数精度：
```python
pi = 3.14159
formatted_string = "Pi: %.2f" % pi
print(formatted_string)  # 输出: Pi: 3.14
```

### 2. `str.format()` 方法

`str.format()`方法更加灵活和强大，允许指定占位符的位置和格式。

#### 基本用法：
```python
name = "Alice"
age = 30
formatted_string = "Name: {}, Age: {}".format(name, age)
print(formatted_string)  # 输出: Name: Alice, Age: 30
```

#### 使用索引指定占位符的位置：
```python
formatted_string = "Name: {0}, Age: {1}".format(name, age)
print(formatted_string)  # 输出: Name: Alice, Age: 30
```

#### 使用命名参数：
```python
formatted_string = "Name: {name}, Age: {age}".format(name=name, age=age)
print(formatted_string)  # 输出: Name: Alice, Age: 30
```

#### 控制浮点数精度：
```python
pi = 3.14159
formatted_string = "Pi: {:.2f}".format(pi)
print(formatted_string)  # 输出: Pi: 3.14
```

### 3. f-string（格式化字符串字面量）

f-string是Python 3.6引入的一种新的格式化方法，提供了简洁和高效的方式。

#### 基本用法：
```python
name = "Alice"
age = 30
formatted_string = f"Name: {name}, Age: {age}"
print(formatted_string)  # 输出: Name: Alice, Age: 30
```

#### 控制浮点数精度：
```python
pi = 3.14159
formatted_string = f"Pi: {pi:.2f}"
print(formatted_string)  # 输出: Pi: 3.14
```

### 字符串格式化技术的应用场景

- **日志记录**：格式化日志信息以便调试和分析。
- **数据输出**：生成报告或导出数据。
- **用户界面**：动态显示信息。
  
