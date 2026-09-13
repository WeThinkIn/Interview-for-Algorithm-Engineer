# 目录

- [1.Python中迭代器的概念？](#1.python中迭代器的概念？)
- [2.Python中生成器的相关知识](#2.python中生成器的相关知识)
- [3.Python中装饰器的相关知识](#3.python中装饰器的相关知识)
- [4.Python的深拷贝与浅拷贝？](#4.python的深拷贝与浅拷贝？)
- [5.Python的垃圾回收机制](#5.python的垃圾回收机制)
- [6.Python中 $ *args $ 和 $ **kwargs $ 的区别？](#6.python中args和kwargs的区别？)
- [7.Python中Numpy的broadcasting机制？](#7.python中numpy的broadcasting机制？)
- [8.python中@staticmethod和@classmethod使用注意事项](#8.python中@staticmethod和@classmethod使用注意事项)
- [9.Python中有哪些常用的设计模式？](#9.Python中有哪些常用的设计模式？)
- [10.Python中的lambda表达式？](#10.Python中的lambda表达式？)
- [11.介绍一下Python中的引用计数原理，如何消除一个变量上的所有引用计数?](#11.介绍一下Python中的引用计数原理，如何消除一个变量上的所有引用计数?)
- [12.有哪些提高python运行效率的方法?](#12.有哪些提高python运行效率的方法?)
- [13.线程池与进程池的区别是什么?](#13.线程池与进程池的区别是什么?)
- [14.multiprocessing模块怎么使用?](#14.multiprocessing模块怎么使用?)
- [15.ProcessPoolExecutor怎么使用?](#15.ProcessPoolExecutor怎么使用?)
- [16.Python中什么情况下会产生内存泄漏?](#16.Python中什么情况下会产生内存泄漏?)
- [17.介绍一下Python中的封装(Encapsulation)思想](#17.介绍一下Python中的封装(Encapsulation)思想)
- [18.介绍一下Python中的继承（Inheritance）思想](#18.介绍一下Python中的继承（Inheritance）思想)
- [19.介绍一下Python中的多态（Polymorphism）思想](#19.介绍一下Python中的多态（Polymorphism）思想)
- [20.介绍一下Python的自省特性](#20.介绍一下Python的自省特性)
- [21.介绍一下Python中的sequence和mapping代表的数据结构](#21.介绍一下Python中的sequence和mapping代表的数据结构)
- [22.Python中使用async def定义函数有什么作用？](#22.Python中使用async-def定义函数有什么作用？)
- [23.Python中布尔索引有哪些用法？](#23.Python中布尔索引有哪些用法？)
- [24.Python中有哪些高级的逐元素矩阵级计算操作？](#24.Python中有哪些高级的逐元素矩阵级计算操作？)
- [25.Python中使用迭代器遍历和非迭代器遍历有什么区别？](#25.Python中使用迭代器遍历和非迭代器遍历有什么区别？)
- [26.介绍一下Python中map与reduce函数的用法](#26.介绍一下Python中map与reduce函数的用法)
- [27.介绍一下Python中高阶函数的原理](#27.介绍一下Python中高阶函数的原理)
- [28.Python与C++有哪些区别？](#28.Python与C++有哪些区别？)
- [29.Python与C语言有哪些区别？](#29.Python与C语言有哪些区别？)
- [30.在AI行业中，Python编程中的动态库和静态库的含义是什么？两者之间什么差异？](#30.在AI行业中，Python编程中的动态库和静态库的含义是什么？两者之间什么差异？)
- [31.Python中的闭包是什么？在AI工程中有什么用？](#31.Python中的闭包是什么在AI工程中有什么用)
- [32.Python中的元类metaclass是什么？](#32.Python中的元类metaclass是什么)
- [33.Python中的上下文管理器with和__enter__/__exit__有什么价值？](#33.Python中的上下文管理器with和__enter____exit__有什么价值)
- [34.Python协程、asyncio和异步IO在AI Agent中如何使用？](#34.Python协程asyncio和异步IO在AI-Agent中如何使用)
- [35.Python设计模式在AI Agent系统中如何落地？](#35.Python设计模式在AI-Agent系统中如何落地)
- [36.Python对象池、连接池和模型池在AI服务中有什么区别？](#36.Python对象池连接池和模型池在AI服务中有什么区别)
- [37.Python中如何设计插件化工具注册机制？](#37.Python中如何设计插件化工具注册机制)


<h2 id="1.python中迭代器的概念？">1.Python中迭代器的概念？</h2>

<font color=DeepSkyBlue>可迭代对象是迭代器、生成器和装饰器的基础。</font>简单来说，可以使用for来循环遍历的对象就是可迭代对象。比如常见的list、set和dict。

注意：Python 3.10 起应从 `collections.abc` 导入抽象基类，`from collections import Iterable` 已废弃：

```python
from collections.abc import Iterable, Iterator

print(isinstance('abcddddd', Iterable))     # str是否可迭代
print(isinstance([1, 2, 3, 4, 5, 6], Iterable))  # list是否可迭代
print(isinstance(12345678, Iterable))       # 整数是否可迭代

-------------结果如下----------------
True
True
False
```

当对所有的可迭代对象调用 dir() 方法时，会发现他们都实现了 `__iter__` 方法。这样就可以通过 `iter(object)` 来返回一个迭代器。

```python
x = [1, 2, 3]
y = iter(x)
print(type(x))
print(type(y))

------------结果如下------------
<class 'list'>
<class 'list_iterator'>
```

可以看到调用 `iter()` 之后，变成了一个 `list_iterator` 的对象，并且多了 `__next__` 方法。<font color=DeepSkyBlue>所有实现了 `__iter__` 和 `__next__` 两个方法的对象，都是迭代器</font>。

<font color=DeepSkyBlue>迭代器是带状态的对象，它会记录当前迭代所在的位置</font>。`__iter__` 返回迭代器自身，`__next__` 返回容器中的下一个值，如果容器中没有更多元素了，则抛出 `StopIteration` 异常。

```python
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
Traceback (most recent last):
  File "test.py", line 6, in <module>
    print(next(y))
StopIteration
```

Python的for循环本质上就是通过不断调用 `next()` 函数实现的。<font color=DeepSkyBlue>迭代器只有在调用 `next()` 时才实际计算下一个值，因此可显著节省内存</font>。

```python
x = [1, 2, 3]
for elem in x:
    ...
```

### 一线实践：为什么要关心迭代器？

1. **流式读取大数据/日志**：按行读取 GB 级日志或数据集，避免一次性加载进内存。
2. **DataLoader 与 batch 拼装**：训练/推理时按 batch 拉取数据，本质就是自定义迭代器。
3. **LLM 流式输出**：`stream=True` 时，SDK 返回的就是 token 级迭代器。

```python
def read_jsonl(path):
    """按行读取 jsonl，内存占用 O(1)"""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield __import__("json").loads(line)

# 训练数据批量迭代
def batch_iter(items, batch_size):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
```

itertools 库提供了很多常见迭代器的使用：

```python
from itertools import count, islice, chain

counter = count(start=13)
print(next(counter))  # 13
print(next(counter))  # 14

# 无限迭代器用 islice 截断
first_5 = list(islice(count(100), 5))  # [100, 101, 102, 103, 104]

# 多数据源拼接成一个迭代流
stream = chain(train_files, val_files)
```


<h2 id="2.python中生成器的相关知识">2.Python中生成器的相关知识</h2>

创建列表时受内存限制，容量有限。列表生成式「定义即生成」，对大体量数据非常浪费。

如果列表元素可以按某种算法推算出来，就可以在循环过程中不断推算后续元素，不必创建完整 list。这种一边循环一边计算的机制称为生成器（generator）。

最简单的创建方式是把列表生成式的 `[]` 改成 `()`：

```python
a = [x * x for x in range(10)]
print(a)
b = (x * x for x in range(10))
print(b)

--------结果如下--------------
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
<generator object <genexpr> at 0x10557da50>
```

另一种方式是生成器函数：用 `def` 定义，函数体里用 `yield`：

```python
def spam():
    yield "first"
    yield "second"
    yield "third"

for x in spam():
    print(x)

-------结果如下---------
first
second
third
```

调用生成器函数时不会立刻执行函数体，而是返回一个生成器对象。执行到 `yield` 就暂停并返回，下次 `next()` 时从断点继续。

generator 还有 `send()`、`throw()`、`close()` 方法，只能在生成器处于挂起状态时使用。`send()` 可以把值注入到 `yield` 表达式的结果，这是实现简单协程的基础。

### 一线实践：生成器的典型场景

**1. 流式处理 LLM 返回**

```python
import openai

client = openai.OpenAI()

def stream_chat(prompt: str):
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta  # 边收边用，不等全部生成完

for token in stream_chat("介绍一下生成器"):
    print(token, end="", flush=True)
```

**2. 大文件 ETL / 预处理**

```python
import json

def transform_logs(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("level") == "ERROR":
                yield {"ts": rec["ts"], "msg": rec["msg"][:200]}
```

**3. 管道式组合（惰性求值）**

```python
def numbers(n):
    for i in range(n):
        yield i

def square(seq):
    for x in seq:
        yield x * x

def is_even(seq):
    for x in seq:
        if x % 2 == 0:
            yield x

# 全程不落地中间 list
result = list(is_even(square(numbers(100))))
```

<font color=DeepSkyBlue>生成器适合「流式、大体量、不需要反复访问」的数据；需要随机访问或多次遍历时，应转成 list</font>。


<h2 id="3.python中装饰器的相关知识">3.Python中装饰器的相关知识</h2>

装饰器允许在不修改原函数代码的前提下，向函数添加额外功能。本质是一个接收函数、返回新函数的高阶函数。

### 从手动封装到 @ 语法糖

```python
import logging

def use_log(func):
    def wrapper(*args, **kwargs):
        logging.warning("%s is running", func.__name__)
        return func(*args, **kwargs)
    return wrapper

@use_log
def bar():
    print("I am bar")

bar()
------------结果如下------------
WARNING:root:bar is running
I am bar
```

`@use_log` 等价于 `bar = use_log(bar)`。

### 带参数的装饰器

```python
import time
from functools import wraps

def retry(max_retries: int = 3, delay: float = 1.0):
    def decorator(func):
        @wraps(func)  # 保留原函数元信息，避免被 IDE/文档/监控搞混
        def wrapper(*args, **kwargs):
            last_exc = None
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    time.sleep(delay * (2 ** i))  # 简单指数退避
            raise last_exc
        return wrapper
    return decorator

@retry(max_retries=3, delay=0.5)
def call_llm(prompt: str):
    ...
```

### 一线实践中装饰器几乎无处不在

| 场景 | 典型装饰器 | 作用 |
|------|-----------|------|
| Web 框架路由 | `@app.get("/health")` | 注册路由 |
| 缓存 | `@lru_cache` / `@cache` | 避免重复计算（embedding、配置解析） |
| 重试/超时 | `@retry`、`tenacity` | 外部 API 不稳定时的容错 |
| 鉴权 | `@login_required` | FastAPI 依赖注入 / Flask before |
| 日志与追踪 | `@trace`、OpenTelemetry | 记录耗时、入参、异常 |
| 权限与配额 | 自定义 `@rate_limit` | LLM 服务限流 |
| torch.no_grad | `@torch.no_grad()` | 推理时关闭梯度 |

### 类装饰器与装饰器类

```python
# 用类实现装饰器（可带状态，适合统计调用次数、耗时）
class CountCalls:
    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        return self.func(*args, **kwargs)

@CountCalls
def inference(x):
    return x * 2

inference(1)
print(inference.count)  # 1
```

<font color=DeepSkyBlue>工程建议：装饰器里务必用 `functools.wraps` 保留原函数的 `__name__`/`__doc__`/签名，否则类型检查、文档生成和 APM 链路都会出问题。</font>


<h2 id="4.python的深拷贝与浅拷贝？">4.Python的深拷贝与浅拷贝？</h2>

在Python中，用一个变量给另一个变量赋值，其实就是给当前内存中的对象增加一个「标签」。

```python
>>> a = [6, 6, 6, 6]
>>> b = a
>>> print(id(a), id(b), sep='\n')
66668888
66668888
>>> a is b
True  # a和b指向内存中同一个对象
```

<font color=DeepSkyBlue>浅拷贝</font>创建一个新对象，其内容是原对象中元素的引用（新对象与原对象共享子对象）。

注：浅拷贝和深拷贝的差异只体现在组合对象（列表、dict、类实例等）上。数字、字符串等原子类型没有拷贝一说，赋值都是引用。

常见浅拷贝：切片 `a[:]`、`list(a)`、`dict(d)`、`.copy()`、`copy.copy()`。

```python
>>> import copy
>>> a = [[6, 6], [8, 8], [9, 9]]
>>> b = a[:]          # 浅拷贝
>>> c = copy.deepcopy(a)  # 深拷贝
>>> a[0] is b[0]      # True，共享子对象
True
>>> a[0] is c[0]      # False，完全独立
False
>>> a[0][0] = 999
>>> print(a[0], b[0], c[0])
[999, 6] [999, 6] [6, 6]
```

<font color=DeepSkyBlue>深拷贝</font>递归拷贝所有子对象，与原对象完全独立。实现方式只有 `copy.deepcopy()`。

### 一线实践：什么时候必须分清？

**1. 配置/超参对象被多个组件共享时**

```python
base_cfg = {"lr": 1e-4, "layers": [12, 24, 36]}

# 错误：改 trainer 的 layers 会同时改掉 eval 的
trainer_cfg = base_cfg
eval_cfg = base_cfg

# 正确：需要独立可变副本时用深拷贝
import copy
trainer_cfg = copy.deepcopy(base_cfg)
eval_cfg = copy.deepcopy(base_cfg)
trainer_cfg["layers"].append(48)
# eval_cfg["layers"] 不受影响
```

**2. 缓存历史消息 / 对话上下文**

```python
history = [{"role": "user", "content": "hi"}]
snapshot = history.copy()       # 浅拷贝：append 历史不影响 snapshot 长度
# 但若原地改 history[0]["content"]，snapshot 也会变
snapshot = copy.deepcopy(history)  # 彻底隔离
```

**3. Torch 张量与自定义对象**

```python
import torch

a = torch.randn(3, 3)
b = a.clone()              # 推荐：张量自己的拷贝语义
c = copy.deepcopy(a)       # 也可用，但 clone/ detach 语义更清晰
d = a.detach().clone()     # 断梯度 + 拷贝
```

<font color=DeepSkyBlue>经验法则：默认优先浅拷贝；只有当子对象会被原地修改且不希望波及副本时，才用 deepcopy（深拷贝大对象成本高）。</font>


<h2 id="5.python的垃圾回收机制">5.Python的垃圾回收机制</h2>

在Python中，使用<font color=DeepSkyBlue>引用计数</font>进行主路径回收；通过<font color=DeepSkyBlue>标记-清除</font>解决容器对象的循环引用；再通过<font color=DeepSkyBlue>分代回收</font>提高扫描效率。

### 1. 引用计数（主机制）

对象被引用时 `ob_refcnt+1`，引用消失时 `-1`，降为 0 立即销毁。触发点包括赋值、传参、放入容器等。

```python
import sys
a = [1, 2, 3]
print(sys.getrefcount(a))  # 通常比你数出来的多 1（getrefcount 自己的临时引用）
```

### 2. 标记-清除（处理循环引用）

两个对象互相引用时，引用计数永远不会到 0。GC 会定期扫描容器对象（list/dict/实例等），找出不可达的环并回收。

### 3. 分代回收

新创建对象大多朝生夕死，所以把对象分成 0/1/2 代，年轻代回收更频繁，老年代更少打扰。

### 一线实践：和 AI 服务相关的关键点

```python
import gc
import torch

# 1) 大对象用完及时断开，别指望 GC 立刻帮你收 GPU 显存
model = None
gc.collect()
torch.cuda.empty_cache()

# 2) 循环引用常见于：回调闭包、双向链表、缓存对象图
# 可用 weakref 打破环
import weakref
class Node:
    def __init__(self):
        self.parent = None
        self.children = []

parent = Node()
child = Node()
parent.children.append(child)
child.parent = weakref.ref(parent)  # 弱引用，不增加引用计数

# 3) 长驻服务里不要滥用 gc.disable()；必要时用 gc.freeze() 减少全量扫描
```

<font color=DeepSkyBlue>注意：Python GC 只管 CPU 侧对象；CUDA 显存、C 扩展里的原生内存需要显式释放（`.detach()`、`del`、`empty_cache`、上下文退出）。</font>


<h2 id="6.python中args和kwargs的区别？">6.Python中$*args$和$**kwargs$的区别？</h2>

$*args$ 和 $**kwargs$ 主要用于函数定义，把不定数量的参数传给函数。

### $*args$

$*args$ 接收<font color=DeepSkyBlue>非键值对的可变位置参数</font>，在函数内是 tuple。

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

### $**kwargs$

$**kwargs$ 接收<font color=DeepSkyBlue>不定长度的键值对参数</font>，在函数内是 dict。

```python
def greet_me(**kwargs):
    for key, value in kwargs.items():
        print("{0} == {1}".format(key, value))

greet_me(name="yasoob")

-----------结果如下-------------
name == yasoob
```

### 一线实践：包装、透传与签名对齐

**1. 写通用 wrapper 时必须透传**

```python
import time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.perf_counter()-t0:.3f}s")
        return result
    return wrapper
```

**2. 调用第三方 SDK 时组装参数**

```python
def chat(prompt: str, **kwargs):
    defaults = {"model": "gpt-4o-mini", "temperature": 0.7}
    defaults.update(kwargs)
    return client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        **defaults,
    )

chat("hi", temperature=0.2, max_tokens=64)
```

**3. 强制关键字参数与位置限定（Python 3）**

```python
def train(data, *, lr: float = 1e-4, batch_size: int = 32):
    ...

# train(data, 1e-3)          # TypeError
train(data, lr=1e-3)          # 正确，意图更清晰
```

<font color=DeepSkyBlue>面试常问：函数调用时 `f(*seq, **mapping)` 是「解包」；定义时是「收集」。二者方向相反。</font>


<h2 id="7.python中numpy的broadcasting机制？">7.Python中Numpy的broadcasting机制？</h2>

NumPy 的 broadcasting 让不同形状的数组在算术运算时自动对齐，避免显式循环或 `tile`/`repeat`。

```python
>>> import numpy as np
>>> a = np.array([1, 2, 3])
>>> b = np.array([6, 6, 6])
>>> c = a + b
>>> c
array([7, 8, 9])
```

形状不一致时，broadcasting 会把标量/低维数组扩展到兼容形状（<font color=DeepSkyBlue>不真正分配复制内存</font>）：

```python
>>> d = a + 5
>>> d
array([6, 7, 8])
```

多维示例：

```python
>>> e = np.ones((3, 3))
>>> e + a   # a.shape=(3,) 被广播成 (3,3)
array([[2., 3., 4.],
       [2., 3., 4.],
       [2., 3., 4.]])

>>> b = np.arange(3).reshape((3, 1))
>>> b + a
array([[1, 2, 3],
       [2, 3, 4],
       [3, 4, 5]])
```

### 规则总结

1. 从右向左对齐 shape；缺的维度用 1 补齐。
2. 长度为 1 的维度被扩展成另一数组对应维度的长度。
3. 两边对应维度既不相等、也不为 1，则报错。

```python
>>> a = np.arange(3)          # (3,)
>>> b = np.ones((3, 2))       # (3, 2)
>>> a + b
ValueError: operands could not be broadcast together with shapes (3,) (3,2)
```

### 一线实践：broadcasting 在 AI 里的真实用法

```python
import numpy as np

# 1) 批量归一化：减均值除标准差（mean/std 形状 (C,) 或 (C,1,1)）
x = np.random.randn(32, 3, 224, 224)  # NCHW
mean = x.mean(axis=(0, 2, 3), keepdims=True)
std = x.std(axis=(0, 2, 3), keepdims=True) + 1e-6
x_norm = (x - mean) / std

# 2) 注意力 mask：把 (B,1,1,T) 广播到 (B,H,T,T)
scores = np.random.randn(2, 8, 16, 16)  # B,H,T,T
mask = np.ones((2, 1, 1, 16)) * -1e9
masked = scores + mask

# 3) 余弦相似度批量计算
emb = np.random.randn(1000, 512)
emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)  # keepdims 才能广播
query = np.random.randn(1, 512)
query = query / np.linalg.norm(query, axis=1, keepdims=True)
sims = emb @ query.T  # (1000, 1)
```

<font color=DeepSkyBlue>Torch 中 broadcasting 规则相同，是写自定义层、loss、mask 时必须吃透的基础。</font>


<h2 id="8.python中@staticmethod和@classmethod使用注意事项">8.python中@staticmethod和@classmethod使用注意事项</h2>

### @staticmethod

1. 静态方法：把普通函数放进类命名空间，不接收 `self`/`cls`。
2. 不能修改类或实例状态，只是「归类放在一起」的工具函数。
3. 适合与类相关、但不依赖类/实例数据的纯逻辑。

### @classmethod

1. 类方法：第一个参数是 `cls`，指向类本身。
2. 可以读写类属性，常用于替代构造器（alternate constructor）。
3. 适合需要访问/修改类级状态，或按类创建实例的场景。

### 对比

| | 实例方法 | classmethod | staticmethod |
|--|---------|-------------|--------------|
| 第一参数 | `self` | `cls` | 无 |
| 能否访问实例状态 | 能 | 否 | 否 |
| 能否访问类状态 | 能 | 能 | 否 |
| 常见用途 | 业务逻辑 | 工厂、缓存、多态构造 | 纯工具函数 |

```python
import torch

class ModelRegistry:
    _cache: dict = {}

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = self._load()

    def _load(self):
        key = self.cfg["name"]
        # 类级缓存，全局只加载一次
        if key not in ModelRegistry._cache:
            ModelRegistry._cache[key] = torch.load(self.cfg["path"])
        return ModelRegistry._cache[key]

    @classmethod
    def from_pretrained(cls, name: str):
        # 多态构造：不同配置来源统一入口
        return cls({"name": name, "path": f"ckpt/{name}.pt"})

    @staticmethod
    def validate_cfg(cfg: dict) -> bool:
        return "name" in cfg and "path" in cfg
```

### 真实踩坑：模型被重复加载

在 Flask/FastAPI 中，若每次请求都 `Infer()`，`__init__` 会反复加载权重，导致接口「推理很快、整请求很慢」。

```python
class Infer:
    _model = None

    def __init__(self, cfg: dict):
        self.cfg = cfg
        if Infer._model is None:
            Infer._model = self.load_model(cfg)
        self.model = Infer._model

    @classmethod
    def load_model(cls, cfg: dict):
        # 类属性做单例缓存
        if not hasattr(cls, "model") or cls.model is None:
            cls.model = torch.load(cfg["path"])
        return cls.model
```

更干净的一线做法是在应用生命周期里只加载一次：

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = load_engine()   # 启动时加载
    yield
    app.state.engine = None            # 关闭时释放

app = FastAPI(lifespan=lifespan)
```


<h2 id="9.Python中有哪些常用的设计模式？">9.Python中有哪些常用的设计模式？</h2>

Python 支持多种设计模式。下面按创建型 / 结构型 / 行为型给出在 AI 工程中真正常用的几类。

### 创建型模式

**1. 单例模式（Singleton）**

保证全局唯一实例。AI 服务里常见于：全局配置、tokenizer、模型缓存、客户端连接。

```python
class Settings:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def __init__(self):
        if self._loaded:
            return
        self.api_key = "..."
        self._loaded = True
```

更推荐用模块级变量、`lru_cache` 或 FastAPI 的 `lifespan`/依赖注入，而不是手写 `__new__` 单例。

**2. 工厂方法（Factory Method）**

按类型创建对象。多模型路由、多数据源适配时非常常见。

```python
def build_model(cfg: dict):
    kind = cfg["type"]
    if kind == "hf":
        from transformers import AutoModel
        return AutoModel.from_pretrained(cfg["name"])
    if kind == "torch":
        return TorchModel(cfg)
    raise ValueError(f"unknown model type: {kind}")
```

**3. 抽象工厂（Abstract Factory）**

创建一组相关对象。例如：不同后端（OpenAI / 本地 vLLM）各自提供 client + tokenizer + encoder。

### 结构型模式

**1. 适配器（Adapter）**

统一不同 SDK/接口。把各家 LLM API 适配成同一个 `chat(messages) -> str`。

```python
class OpenAIAdapter:
    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    def chat(self, messages: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages
        )
        return resp.choices[0].message.content

class VLLMAdapter:
    def __init__(self, base_url: str, model: str):
        ...
    def chat(self, messages: list[dict]) -> str:
        ...

def get_adapter(cfg) -> object:
    return OpenAIAdapter(...) if cfg["provider"] == "openai" else VLLMAdapter(...)
```

**2. 装饰器模式（Decorator）**

动态叠加能力：重试、缓存、打点、脱敏，而不改核心调用链。

**3. 代理（Proxy）**

控制访问：懒加载模型、权限校验、远程调用封装。

```python
class LazyModelProxy:
    def __init__(self, factory):
        self._factory = factory
        self._model = None

    def __getattr__(self, name):
        if self._model is None:
            self._model = self._factory()
        return getattr(self._model, name)
```

### 行为型模式

**1. 观察者（Observer）**

事件总线：训练完成、评测结束、webhook 回调。

```python
class EventBus:
    def __init__(self):
        self._subs: dict[str, list] = {}

    def on(self, event: str, fn):
        self._subs.setdefault(event, []).append(fn)

    def emit(self, event: str, payload):
        for fn in self._subs.get(event, []):
            fn(payload)

bus = EventBus()
bus.on("train_done", lambda p: print("save ckpt", p["step"]))
bus.emit("train_done", {"step": 1000})
```

**2. 策略（Strategy）**

可替换算法：不同重试策略、不同采样策略、不同 chunk 切分策略。

```python
class ChunkStrategy:
    def split(self, text: str) -> list[str]:
        raise NotImplementedError

class FixedSizeChunk(ChunkStrategy):
    def __init__(self, size: int = 512):
        self.size = size
    def split(self, text: str):
        return [text[i:i+self.size] for i in range(0, len(text), self.size)]

class RecursiveChunk(ChunkStrategy):
    def split(self, text: str):
        # 按段落/句子优先切分
        ...

def chunk(text: str, strategy: ChunkStrategy):
    return strategy.split(text)
```

**3. 模板方法（Template Method）**

训练/评测骨架固定，步骤由子类实现：`load_data` → `forward` → `loss` → `optimize`。


<h2 id="10.Python中的lambda表达式？">10.Python中的lambda表达式？</h2>

Lambda 表达式即匿名函数，语法为：

```python
lambda 参数: 表达式
```

### 主要特征

1. 可有任意数量参数，但只能有一个表达式（不能写语句）。
2. 适合简短、一次性的逻辑。
3. 调用时求值并返回表达式结果。

### 示例

```python
f = lambda x: x * 2
print(f(3))  # 6

g = lambda x, y: x + y
print(g(2, 3))  # 5

words = ["apple", "hi", "banana"]
print(sorted(words, key=lambda w: len(w)))
# ['hi', 'apple', 'banana']
```

### 一线实践

```python
# 1) pandas / 结果排序
df["score"] = df["text"].apply(lambda t: len(t.split()))

# 2) 按置信度排序评测结果
results.sort(key=lambda r: r["confidence"], reverse=True)

# 3) 简单默认工厂（注意：默认参数求值时机）
handlers = {
    "sum": lambda xs: sum(xs),
    "mean": lambda xs: sum(xs) / len(xs),
}
```

### 优点与局限

- 优点：简洁、便于和 `sorted`/`map`/`filter` 组合。
- 局限：只能单表达式；复杂逻辑请用 `def`，否则可读性变差。

<font color=DeepSkyBlue>不要写过长的 lambda；需要复用、调试或加文档时，升级成命名函数。</font>


<h2 id="11.介绍一下Python中的引用计数原理，如何消除一个变量上的所有引用计数?">11.介绍一下Python中的引用计数原理，如何消除一个变量上的所有引用计数?</h2>

引用计数是 CPython 垃圾回收的主路径，用来跟踪对象被多少地方引用。

### 原理

1. **创建**：对象出生时引用计数为 1。
2. **增加**：新引用指向它时 +1（赋值、传参、放入容器）。
3. **减少**：引用消失时 -1（重新赋值、del、离开作用域）。
4. **销毁**：计数降到 0 时立即释放内存。

```python
a = [1, 2, 3]   # 1
b = a           # 2
c = a           # 3
del b           # 2
c = None        # 1
del a           # 0 → 释放
```

```python
import sys
a = [1, 2, 3]
print(sys.getrefcount(a))  # 多 1：getrefcount 自己的临时引用
```

### 如何消除一个变量上的所有引用

1. 删除所有指向它的变量（`del`）。
2. 从容器中移除（list/dict/set/自定义缓存）。
3. 打破循环引用（`weakref` 或等 GC 扫描）。

```python
import gc

a = [1, 2, 3]
b = a
c = {"key": a}

del a, b
del c["key"]
gc.collect()
```

### 循环引用

```python
class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

node1 = Node(1)
node2 = Node(2)
node1.next = node2
node2.next = node1  # 环

del node1, node2
gc.collect()  # 需要 GC 兜底
```

一线更推荐主动避免环：

```python
import weakref

class Session:
    def __init__(self):
        self._tools = []

class Tool:
    def __init__(self, session: Session):
        self.session = weakref.ref(session)  # 不阻止 session 回收
```

<font color=DeepSkyBlue>长生命周期缓存（tokenizer、模型、连接）最容易造成「内存泄漏式」引用堆积；用 weakref.WeakValueDictionary 或显式失效策略。</font>


<h2 id="12.有哪些提高python运行效率的方法?">12.有哪些提高python运行效率的方法?</h2>

## 一、优化代码结构

### 1. 合适的数据结构与算法

- 查找用 `set`/`dict`，不要在 list 上做 `in`（O(n)）。
- 频繁两端操作用 `collections.deque`。
- 计数用 `Counter`，去重保持顺序用 `dict.fromkeys`。

### 2. 减少重复计算

```python
from functools import lru_cache

@lru_cache(maxsize=1024)
def embed_text(text: str):
    ...  # 调 embedding API 前先查缓存
```

### 3. 向量化优先于 Python 循环

```python
# 慢
result = [x * 2 for x in huge_list]

# 快：NumPy / Torch 向量化
import numpy as np
arr = np.asarray(huge_list)
result = arr * 2
```

### 4. 生成器流式处理

大文件、大结果集用 yield，避免一次性 materialize。

## 二、高性能库与工具

| 工具 | 适用场景 |
|------|----------|
| NumPy / Pandas | 数值计算、表格处理 |
| PyTorch | GPU 张量与自动微分 |
| `torch.compile` | 编译加速热点模型 |
| Numba | 数值热循环 JIT |
| Cython / C 扩展 | 真正的计算瓶颈 |
| PyPy | 纯 CPU 的 Python 脚本（兼容性需验证） |
| polars / duckdb | 大规模列式数据分析 |

## 三、并发模型选择

- **I/O 密集**：`asyncio` + 异步 HTTP（调 LLM API、爬取、写库）优先。
- **CPU 密集**：多进程（`ProcessPoolExecutor`）或把计算下沉到 NumPy/原生库/GPU。
- **多线程**：受 GIL 限制，适合 I/O，不适合纯 Python CPU 并行。

```python
import asyncio
import httpx

async def fetch(client, url):
    r = await client.get(url)
    return r.json()

async def main(urls):
    async with httpx.AsyncClient(timeout=30) as client:
        tasks = [fetch(client, u) for u in urls]
        return await asyncio.gather(*tasks)
```

## 四、性能分析先行

```bash
python -m cProfile -o out.prof train.py
python -m pstats out.prof  # 交互查看热点

# 逐行分析
pip install line_profiler
kernprof -l -v hot_script.py
```

```python
# 内存
# pip install memory_profiler scalene
```

## 五、代码级微优化

1. 局部变量快于全局/属性查找：`range_ = range`、缓存 `self.x` 到局部。
2. 字符串拼接用 `"".join(parts)`，不要在循环里 `s += t`。
3. 异常控制流有成本，不要用异常做普通分支。
4. 批量 I/O：一次读 1MB，不要一次读 1 字节。

## 六、核心思想

1. **先 profiling，再优化**，避免猜热点。
2. **算法与数据结构优先**，然后向量化，最后才是并行/编译。
3. **I/O 用异步，计算用原生/GPU**。
4. **可读性与正确性优先于过早的微优化**。


<h2 id="13.线程池与进程池的区别是什么?">13.线程池与进程池的区别是什么?</h2>

#### 1. 线程池

维护一组可复用线程，降低创建/销毁开销。线程共享内存，切换轻量。

- **适用**：I/O 密集（网络请求、磁盘、数据库）。
- **限制**：受 GIL，纯 Python CPU 代码无法真正并行。

#### 2. 进程池

维护一组进程，各自独立地址空间，不受 GIL，可吃满多核。

- **适用**：CPU 密集（图像处理、特征计算、大量 Python 逻辑）。
- **代价**：进程创建贵，进程间通信要 pickle，共享状态复杂。

### 对比表

| 特性 | 线程池 | 进程池 |
|------|--------|--------|
| 任务类型 | I/O 密集 | CPU 密集 |
| GIL | 受限 | 不受限 |
| 内存 | 共享 | 隔离 |
| 切换开销 | 小 | 大 |
| 通信 | 直接共享变量（需锁） | Queue/Pipe/Manager |
| 典型场景 | 调 API、读写文件 | 解码图像、数值计算 |

### 如何选择

1. 任务大部分时间在等网络/磁盘 → 线程池或 asyncio。
2. 任务大部分时间在算 → 进程池，或把计算下沉到 C/GPU。
3. 混合负载：异步调度 I/O，进程池处理 CPU 段。

### 延伸：asyncio 何时优于线程池？

高并发网络调用（几百路 LLM API）时，单线程事件循环 + 异步客户端通常比开几百个线程更省内存、更易控超时与重试。线程池更适合「第三方库只有同步接口」的场景。


-----

Q: ProcessPoolExecutor 是线程还是进程？

A: `ProcessPoolExecutor` 属于**进程**。它创建进程池，每个任务在独立进程中执行，不受 GIL 限制，适合 CPU 密集型任务。相比 `ThreadPoolExecutor`，创建与切换更贵，但能真正并行执行 Python 字节码。


<h2 id="14.multiprocessing模块怎么使用?">14.multiprocessing模块怎么使用?</h2>

在需要绕过 GIL 做 CPU 并行时，`multiprocessing` 提供完整工具链。注意：Windows/macOS 默认 spawn，主模块必须放在 `if __name__ == "__main__":` 下，否则会递归创建进程。

### 1. Process：直接起进程

```python
from multiprocessing import Process
import time

def worker(name):
    print(f"Worker {name} started")
    time.sleep(1)
    print(f"Worker {name} finished")

if __name__ == "__main__":
    p = Process(target=worker, args=("A",))
    p.start()
    p.join()
```

### 2. Pool：进程池 map

```python
from multiprocessing import Pool

def square(x):
    return x * x

if __name__ == "__main__":
    with Pool(4) as p:
        print(p.map(square, [1, 2, 3, 4]))  # [1, 4, 9, 16]
```

### 3. Queue / Pipe：进程间通信

```python
from multiprocessing import Process, Queue

def worker(q):
    q.put("hello")

if __name__ == "__main__":
    q = Queue()
    p = Process(target=worker, args=(q,))
    p.start()
    print(q.get())
    p.join()
```

### 4. Lock：保护共享资源

```python
from multiprocessing import Process, Lock

def worker(lock, num):
    with lock:
        print(f"Process {num} working")

if __name__ == "__main__":
    lock = Lock()
    ps = [Process(target=worker, args=(lock, i)) for i in range(5)]
    for p in ps: p.start()
    for p in ps: p.join()
```

### 5. Value / Array / Manager：共享数据

```python
from multiprocessing import Manager, Process

def worker(shared_dict, key, value):
    shared_dict[key] = value

if __name__ == "__main__":
    with Manager() as manager:
        shared = manager.dict()
        ps = [Process(target=worker, args=(shared, i, i * i)) for i in range(5)]
        for p in ps: p.start()
        for p in ps: p.join()
        print(dict(shared))
```

### 6. Event / Semaphore：同步与限流

```python
from multiprocessing import Event, Semaphore, Process
import time

def worker(sem, num):
    with sem:
        print(f"Worker {num} access")
        time.sleep(1)

if __name__ == "__main__":
    sem = Semaphore(2)
    ps = [Process(target=worker, args=(sem, i)) for i in range(4)]
    for p in ps: p.start()
    for p in ps: p.join()
```

### 一线实践建议

1. **现代代码优先 `concurrent.futures.ProcessPoolExecutor`**，接口更干净，和线程池统一。
2. **传给子进程的参数必须可 pickle**；大数组考虑共享内存 `shared_memory` 或先落盘再传路径。
3. **CUDA 张量不要跨进程直接传**；用独立进程加载模型，或走共享显存/文件。
4. **与 async 混用时**，用 `loop.run_in_executor` 把 CPU 任务丢进进程池。


<h2 id="15.ProcessPoolExecutor怎么使用?">15.ProcessPoolExecutor怎么使用?</h2>

`ProcessPoolExecutor` 是 `concurrent.futures` 中的进程池封装，比 `multiprocessing.Pool` 更高层，和 `ThreadPoolExecutor` 接口几乎一致，适合快速把 CPU 密集任务并行化。

### 基本用法

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

def cpu_task(n: int) -> int:
    time.sleep(0.5)  # 模拟计算
    return n * n

if __name__ == "__main__":
    numbers = list(range(8))

    with ProcessPoolExecutor(max_workers=4) as ex:
        # 方式 1：map，保持输入顺序
        print(list(ex.map(cpu_task, numbers)))

    with ProcessPoolExecutor(max_workers=4) as ex:
        # 方式 2：submit + as_completed，谁先完成谁先处理
        futs = [ex.submit(cpu_task, n) for n in numbers]
        for fut in as_completed(futs):
            print(fut.result())
```

### 主要 API

- `submit(fn, *args, **kwargs)` → 返回 `Future`
- `map(func, iterable)` → 惰性结果迭代器
- `shutdown(wait=True)` → `with` 退出时自动调用
- `Future.result(timeout=...)` / `Future.add_done_callback(...)`

### 与 multiprocessing.Pool 的比较

| | ProcessPoolExecutor | multiprocessing.Pool |
|--|---------------------|---------------------|
| 接口风格 | Future / concurrent.futures | map/apply_async |
| 与线程池一致性 | 高 | 低 |
| 取消任务 | Future.cancel（已运行的不可取消） | 终止较麻烦 |
| 推荐度（新代码） | 更推荐 | 维护旧代码时仍常见 |

### 一线场景

**1. 批量图像预处理 / 解码**

```python
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

def process_image(path: str):
    img = Image.open(path).convert("RGB").resize((224, 224))
    return path, list(img.getdata())  # 或返回 numpy bytes

if __name__ == "__main__":
    paths = [...]  # 上千张
    with ProcessPoolExecutor(max_workers=8) as ex:
        for path, pixels in ex.map(process_image, paths, chunksize=16):
            ...
```

**2. 与 asyncio 组合**

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor

def heavy(x):
    return sum(i * i for i in range(x))

async def main():
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor(max_workers=4) as pool:
        tasks = [loop.run_in_executor(pool, heavy, 10**6) for _ in range(4)]
        print(await asyncio.gather(*tasks))

asyncio.run(main())
```

<font color=DeepSkyBlue>注意：worker 函数必须定义在可导入模块顶层；Windows 下主模块加 `if __name__ == "__main__":`。`chunksize` 对大量小任务能显著降低调度开销。</font>


<h2 id="16.Python中什么情况下会产生内存泄漏?">16.Python中什么情况下会产生内存泄漏?</h2>

CPython 有引用计数兜底，但仍会「逻辑泄漏」或原生资源泄漏。AI 服务中最常见的是缓存无限增长和 GPU 显存未释放。

### 常见原因

**1. 无界缓存 / 全局字典只增不减**

```python
# 危险：每个请求都往里塞
_embedding_cache = {}

def embed(text: str):
    if text not in _embedding_cache:
        _embedding_cache[text] = model.encode(text)
    return _embedding_cache[text]

# 修复：限制大小
from functools import lru_cache

@lru_cache(maxsize=4096)
def embed(text: str):
    return model.encode(text)
```

**2. 循环引用 + 带 `__del__` 的对象**

自定义 `__del__` 的循环引用对象可能无法被 GC 回收。尽量用 `weakref`、上下文管理器，而不是 `__del__` 做资源释放。

**3. 闭包捕获大对象**

```python
def make_handler(huge_array):
    def handler(x):
        return huge_array[x]  # 闭包一直持有 huge_array
    return handler

# 长生命周期注册表里挂满 handler → 内存下不去
```

修复：只捕获需要的切片，或改用弱引用/显式释放。

**4. 全局 list 只 append**

```python
LOGS = []
def on_request(rec):
    LOGS.append(rec)  # 服务跑一周就炸

# 修复：deque(maxlen=10000)
from collections import deque
LOGS = deque(maxlen=10_000)
```

**5. GPU 显存：引用未断开**

```python
# 推理后
outputs = model(batch)
loss = outputs.loss
# 若 outputs / 中间激活被缓存进 list 或日志系统，显存一直占着

# 实践
with torch.no_grad():
    outputs = model(batch)
result = outputs.cpu()
del outputs
torch.cuda.empty_cache()
```

**6. 未关闭的原生资源**

文件、socket、数据库连接、mmap、C 扩展句柄。用 `with` 或显式 `close()`。

**7. 线程/进程对象未 join，队列引用堆积**

生产者往 `Queue` 塞、消费者挂了，队列和 payload 全留住。

### 排查手段

```python
import gc, tracemalloc

tracemalloc.start()
# ... 跑一段业务 ...
snapshot = tracemalloc.take_snapshot()
for stat in snapshot.statistics("lineno")[:10]:
    print(stat)

# 对象数量突增
print(len(gc.get_objects()))
gc.collect()
```

生产上还可以用 `filprofiler`、`memray`、`pympler`，以及 cgroup/PSS 监控容器内存。

<font color=DeepSkyBlue>经验：80% 的「Python 内存泄漏」其实是无界缓存、日志列表和未释放的 GPU 张量，而不是 GC bug。</font>


<h2 id="17.介绍一下Python中的封装(Encapsulation)思想">17.介绍一下Python中的封装(Encapsulation)思想</h2>

封装把数据和操作数据的方法绑在一起，并对外隐藏实现细节，只暴露稳定接口。

### Python 中的「约定式封装」

- `name`：公开
- `_name`：受保护（约定内部使用）
- `__name`：类私有，触发 name mangling（`_ClassName__name`）
- 没有 C++/Java 那种强制访问控制

```python
class Tokenizer:
    def __init__(self, vocab: dict):
        self._vocab = vocab          # 内部实现，外部不应直接改
        self.__version = "1.0"       # 名称改写

    def encode(self, text: str) -> list[int]:
        return [self._vocab.get(t, 0) for t in text.split()]

    def vocab_size(self) -> int:
        return len(self._vocab)
```

### 属性与描述符

```python
from pydantic import BaseModel, Field

class TrainConfig(BaseModel):
    lr: float = Field(1e-4, gt=0, le=1)
    batch_size: int = Field(32, ge=1)

# Pydantic 用类型注解 + 校验做封装，比手写 setter 更省事
cfg = TrainConfig(lr=0.001, batch_size=64)
```

### 一线价值

1. **模型服务对外只暴露 `predict`/`chat`**，内部权重、设备、批处理逻辑可随意改。
2. **配置对象不可随意改字段**，避免运行中被旁路修改导致不可复现。
3. **测试与替换实现更容易**：只要接口不变，内部可以重写。


<h2 id="18.介绍一下Python中的继承（Inheritance）思想">18.介绍一下Python中的继承（Inheritance）思想</h2>

继承让子类复用父类的属性和方法，并可扩展或覆盖。

### 基本语法

```python
class BaseModel:
    def __init__(self, name: str):
        self.name = name

    def forward(self, x):
        raise NotImplementedError

class MLP(BaseModel):
    def __init__(self, name: str, dims: list[int]):
        super().__init__(name)   # 推荐 super()，支持多继承
        self.dims = dims

    def forward(self, x):
        return x  # 实现
```

### 方法解析顺序（MRO）

多重继承时用 C3 线性化决定查找顺序：

```python
class A: ...
class B(A): ...
class C(A): ...
class D(B, C): ...

print(D.mro())
# [D, B, C, A, object]
```

`super()` 按 MRO 前进，而不是简单地「调父类」。

### 私有属性不被子类直接访问

```python
class Parent:
    def __init__(self):
        self.__private = 1  # _Parent__private

class Child(Parent):
    def get(self):
        # return self.__private  # AttributeError
        return self._Parent__private  # 能绕，但别这么写
```

### 一线建议

1. **优先组合而不是深继承**：`class Trainer: def __init__(self, model, data, optim)` 往往比三层继承更清晰。
2. **用 ABC/Protocol 定义接口**，而不是靠「继承某个具体类」来表达能力。
3. **框架扩展点**（HuggingFace Trainer、LightningModule）常用模板方法 + 继承，这是合法且常见的用法。


<h2 id="19.介绍一下Python中的多态（Polymorphism）思想">19.介绍一下Python中的多态（Polymorphism）思想</h2>

多态指同一操作作用于不同对象时产生不同行为。Python 通过动态类型和鸭子类型天然支持多态。

### 鸭子类型

```python
class OpenAIEngine:
    def chat(self, messages):
        return "openai reply"

class LocalEngine:
    def chat(self, messages):
        return "local reply"

def ask(engine, messages):
    return engine.chat(messages)  # 不关心具体类型，有 chat 即可

ask(OpenAIEngine(), [...])
ask(LocalEngine(), [...])
```

### 继承 + 方法重写

```python
from abc import ABC, abstractmethod

class Chunker(ABC):
    @abstractmethod
    def split(self, text: str) -> list[str]:
        ...

class FixedChunker(Chunker):
    def split(self, text: str) -> list[str]:
        return [text[i:i+512] for i in range(0, len(text), 512)]

class SentenceChunker(Chunker):
    def split(self, text: str) -> list[str]:
        return [s.strip() for s in text.split("。") if s.strip()]
```

### 结构化类型：Protocol（typing）

```python
from typing import Protocol

class SupportsChat(Protocol):
    def chat(self, messages: list[dict]) -> str: ...

def run_pipeline(engine: SupportsChat):
    return engine.chat([{"role": "user", "content": "hi"}])
```

类型检查器能做静态校验，运行时仍是鸭子类型。

### 运算符重载

```python
class TensorLike:
    def __init__(self, data):
        self.data = data
    def __add__(self, other):
        return TensorLike(self.data + other.data)
```

### 价值

1. 换模型供应商、换存储后端时，上层业务代码不用改。
2. 测试时注入 Fake/Mock 对象即可。
3. 插件系统天然依赖多态。


<h2 id="20.介绍一下Python的自省特性">20.介绍一下Python的自省特性</h2>

自省（Introspection）指程序在运行时检查自身对象结构的能力：类型、属性、方法、签名、源码等。

### 常用 API

```python
x = 10
print(type(x), isinstance(x, int))  # <class 'int'> True

class Person:
    def __init__(self, name):
        self.name = name
    def greet(self):
        print(f"Hello, {self.name}")

p = Person("Alice")
print(hasattr(p, "name"))           # True
print(getattr(p, "name"))           # Alice
setattr(p, "age", 25)
print(p.__dict__)                   # {'name': 'Alice', 'age': 25}
print(callable(p.greet))            # True
print(dir(p)[:5])                   # 属性与方法列表
```

### inspect 模块

```python
import inspect

def my_function(x: int, y: int = 0) -> int:
    return x + y

print(inspect.signature(my_function))  # (x: int, y: int = 0) -> int
print(inspect.getsource(my_function))
print(inspect.iscoroutinefunction(asyncio.sleep))  # 可判断是否协程
```

### 一线应用

**1. 插件/工具自动发现与注册**

```python
import importlib, pkgutil

def load_tools(package_name: str):
    pkg = importlib.import_module(package_name)
    tools = {}
    for _, mod_name, _ in pkgutil.iter_modules(pkg.__path__):
        mod = importlib.import_module(f"{package_name}.{mod_name}")
        if hasattr(mod, "register"):
            tools.update(mod.register())
    return tools
```

**2. 从函数签名生成 JSON Schema（LLM Function Calling）**

```python
import inspect
from typing import get_type_hints

def fn_schema(fn) -> dict:
    sig = inspect.signature(fn)
    hints = get_type_hints(fn)
    props = {}
    required = []
    for name, param in sig.parameters.items():
        ann = hints.get(name, str)
        props[name] = {"type": {int: "integer", float: "number", str: "string"}.get(ann, "string")}
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {"name": fn.__name__, "parameters": {"type": "object", "properties": props, "required": required}}
```

**3. 序列化、DI、ORM、ORM 映射、调试器**都依赖自省。

<font color=DeepSkyBlue>自省是 Python 元编程与框架能力的底层原语；写 SDK/Agent 框架时几乎必用。</font>


<h2 id="21.介绍一下Python中的sequence和mapping代表的数据结构">21.介绍一下Python中的sequence和mapping代表的数据结构</h2>

`Sequence` 与 `Mapping` 定义在 `collections.abc`，是两类核心容器抽象。

- **Sequence**：按位置有序，支持索引/切片。
- **Mapping**：键值对，按键快速查找。

### Sequence

常见类型：`list`、`tuple`、`str`、`range`、`deque`。

| 操作 | 描述 | 示例 |
|------|------|------|
| `obj[i]` | 索引 | `my_list[0]` |
| `obj[a:b]` | 切片 | `my_list[1:3]` |
| `len(obj)` | 长度 | `len(my_list)` |
| `in` | 成员判断 | `3 in my_list` |
| `for x in obj` | 迭代 | 遍历 |

```python
from collections.abc import Sequence

print(isinstance([1, 2], Sequence))   # True
print(isinstance((1, 2), Sequence))   # True
print(isinstance("hi", Sequence))     # True
print(isinstance({"a": 1}, Sequence)) # False
```

### Mapping

常见类型：`dict`、`defaultdict`、`OrderedDict`、`Counter`、`ChainMap`。

| 操作 | 描述 |
|------|------|
| `obj[k]` / `obj[k]=v` | 读写 |
| `k in obj` | 键存在性 |
| `keys()/values()/items()` | 视图 |
| `obj.get(k, default)` | 安全读取 |

```python
from collections import Counter, defaultdict, ChainMap

tokens = ["a", "b", "a", "c", "a"]
print(Counter(tokens))  # Counter({'a': 3, 'b': 1, 'c': 1})

cfg = ChainMap({"lr": 1e-4}, {"lr": 1e-3, "epochs": 3})  # 前者优先
print(cfg["lr"], cfg["epochs"])
```

### 对比

| | Sequence | Mapping |
|--|----------|---------|
| 组织方式 | 有序位置 | 键值 |
| 访问 | 索引 | 键 |
| 典型 | list/tuple/str | dict/Counter |

### 一线注意

1. **list 不适合频繁头部插入**，用 `deque`。
2. **成员判断 O(1) 用 set/dict**，不要用 list。
3. **Python 3.7+ dict 保序**；需要 LRU 用 `functools.lru_cache` 或 `OrderedDict`。
4. **消息结构、tool call payload、配置**本质上都是 Mapping；序列化前先校验 schema（pydantic）。


<h2 id="22.Python中使用async-def定义函数有什么作用？">22.Python中使用async def定义函数有什么作用？</h2>

`async def` 定义协程函数。调用它得到 coroutine 对象，需要 `await` 或事件循环驱动。

### 作用

- **非阻塞并发**：在等待 I/O 时让出控制权，其他任务继续跑。
- **高吞吐低开销**：单线程可挂起成千上万个等待网络的任务。
- **结构化并发**：`asyncio.gather` / `TaskGroup` 管理一组任务。

```python
import asyncio
import httpx

async def fetch(client: httpx.AsyncClient, url: str):
    r = await client.get(url, timeout=10)
    return r.status_code, len(r.content)

async def main(urls: list[str]):
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*(fetch(client, u) for u in urls), return_exceptions=True)
    return results

asyncio.run(main(["https://example.com", "https://example.org"]))
```

### 与同步的区别

| | 同步 | 异步 |
|--|------|------|
| 等待 I/O | 阻塞整线程 | `await` 让出事件循环 |
| 并发模型 | 多线程/多进程 | 单线程多协程 |
| 适合 | 脚本、CPU | API 调用、高并发网关 |

### 一线实践：并发调用 LLM

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def one(prompt: str):
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content

async def batch(prompts: list[str], concurrency: int = 8):
    sem = asyncio.Semaphore(concurrency)  # 限流，防止打爆 API

    async def guarded(p):
        async with sem:
            return await one(p)

    return await asyncio.gather(*(guarded(p) for p in prompts), return_exceptions=True)

# results = asyncio.run(batch(["解释一下协程"] * 20))
```

### 注意点

1. 事件循环里不要跑重 CPU，会卡住所有任务；CPU 丢线程/进程池。
2. `async` 函数只能被 `await`，直接调用只创建协程对象，不执行。
3. 超时用 `asyncio.wait_for` / `asyncio.timeout`（3.11+）。
4. 取消用 `asyncio.CancelledError` 与 `task.cancel()`。


<h2 id="23.Python中布尔索引有哪些用法？">23.Python中布尔索引有哪些用法？</h2>

布尔索引用布尔数组/掩码筛选数据，是 NumPy/Pandas/Torch 向量化操作的核心。

### 基本用法

```python
import numpy as np

x = np.array([1, 5, 3, 8, 2])
mask = x > 3
print(mask)        # [False  True False  True False]
print(x[mask])     # [5 8]
print(x[x > 3])    # [5 8]
```

### 修改与多条件

```python
x[x < 3] = 0
# [1 5 3 8 2] -> [0 5 3 8 0]

# 与 / 或 / 非
print(x[(x > 2) & (x < 7)])   # 5 3
print(x[(x < 2) | (x > 7)])   # 0 8
print(x[~(x > 3)])            # 取反
```

### 二维与 Torch

```python
import torch

a = torch.tensor([[1, 2], [3, 4]])
print(a[a > 2])  # tensor([3, 4])

# 按行筛选
scores = torch.tensor([0.1, 0.9, 0.4])
keep = scores > 0.5
print(scores[keep])  # tensor([0.9000])

# 置零、掩码填充
logits = torch.randn(4, 10)
mask = torch.zeros(4, 10, dtype=torch.bool)
mask[0, 3] = True
logits = logits.masked_fill(~mask, float("-inf"))
```

### 一线场景

```python
# 1) 过滤低置信度检测框
keep = confidences > 0.5
boxes, scores = boxes[keep], scores[keep]

# 2) 过滤损坏样本
valid = (losses < threshold) & np.isfinite(losses)
dataset = dataset[valid]

# 3) 按标签抽取子集
idx = (labels == target_class)
samples = images[idx]
```

<font color=DeepSkyBlue>规则：掩码形状必须与被索引数组的前缀维度兼容；`& | ~` 不能写成 `and or not`（那是 Python 逻辑运算）。</font>


<h2 id="24.Python中有哪些高级的逐元素矩阵级计算操作？">24.Python中有哪些高级的逐元素矩阵级计算操作？</h2>

在 NumPy/Torch 中，应尽量用向量化逐元素/矩阵级运算代替 Python 循环。

### 常用操作

```python
import numpy as np

a = np.random.randn(4, 8)
b = np.random.randn(4, 8)

# 逐元素
a + b
a * b
np.maximum(a, 0)          # ReLU
np.exp(a)
np.log(np.abs(a) + 1e-8)
np.clip(a, -1, 1)

# 归约
a.sum(axis=-1, keepdims=True)
a.mean(axis=0)
np.max(a, axis=-1)
np.linalg.norm(a, axis=-1, keepdims=True)

# 矩阵级
a @ b.T                   # 矩阵乘
np.einsum("ij,jk->ik", a, b.T)

# 条件
np.where(a > 0, a, 0)     # 类似 mask
```

### Torch 对应

```python
import torch
import torch.nn.functional as F

x = torch.randn(2, 8, 16)
y = torch.randn(2, 8, 16)

F.relu(x)
F.softmax(x, dim=-1)
F.mse_loss(x, y)
torch.einsum("bhd,bmd->bhm", x, y)
x.masked_fill(y < 0, 0.0)
```

### 一线高频：注意力打分

```python
def attention_scores(q, k, scale=None):
    # q: (B,H,T,D)  k: (B,H,S,D)
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return torch.matmul(q, k.transpose(-2, -1)) * scale
```

### 性能建议

1. 用 `keepdims=True` 方便广播，避免 `reshape` 错误。
2. `einsum` 可读性与性能的折中，复杂收缩优先考虑。
3. 能 `matmul`/`conv` 就不要写双层 for。
4. 注意 dtype/设备一致；混合精度时显式 cast。


<h2 id="25.Python中使用迭代器遍历和非迭代器遍历有什么区别？">25.Python中使用迭代器遍历和非迭代器遍历有什么区别？</h2>

| | 迭代器/生成器 | 非迭代器（list 等） |
|--|---------------|---------------------|
| 内存 | 惰性，O(1) 附近 | 全量在内存 |
| 遍历次数 | 通常一次 | 可多次 |
| 随机访问 | 不支持 | 支持 |
| 何时取值 | `next()` 时计算 | 已经算好 |

```python
# 迭代器：耗尽即空
it = iter([1, 2, 3])
print(list(it))  # [1, 2, 3]
print(list(it))  # []

# list 可反复
xs = [1, 2, 3]
print(list(xs), list(xs))
```

### 一线选择

```python
# 适合迭代器
def iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

# 需要 shuffle / 多 epoch / 索引 → 先物化或用可重入数据集类
```

```python
# LLM 流式：必须用迭代器语义
for chunk in client.chat.completions.create(..., stream=True):
    print(chunk.choices[0].delta.content or "", end="")
```

<font color=DeepSkyBlue>口诀：大数据、单次扫、流式 → 迭代器；小数据、反复用、随机访问 → list。</font>


<h2 id="26.介绍一下Python中map与reduce函数的用法">26.介绍一下Python中map与reduce函数的用法</h2>

### map

把函数映射到可迭代对象每个元素，返回惰性迭代器。

```python
nums = [1, 2, 3, 4]
print(list(map(lambda x: x * 2, nums)))  # [2, 4, 6, 8]

# 多序列并行
print(list(map(lambda a, b: a + b, [1, 2], [10, 20])))  # [11, 22]
```

现代写法通常用列表推导更清晰：

```python
result = [x * 2 for x in nums]
```

### reduce

把序列归约成一个值。

```python
from functools import reduce

print(reduce(lambda a, b: a + b, [1, 2, 3, 4]))  # 10

# 带初始值
print(reduce(lambda a, b: a + b, [1, 2, 3, 4], 100))  # 110
```

### 一线实践

```python
# 合并多个 dict 配置层
layers = [{"a": 1}, {"b": 2}, {"a": 3}]
merged = reduce(lambda acc, d: {**acc, **d}, layers, {})
# {'a': 3, 'b': 2}

# 累积评测指标
from operator import mul
print(reduce(mul, [0.9, 0.8, 0.95], 1.0))  # 连乘召回率链
```

### 何时不用 reduce

- 简单求和/求积：直接 `sum()` / `math.prod()` 更快更清晰。
- 复杂归约：写具名函数或用 pandas/numpy 聚合，可读性更好。


<h2 id="27.介绍一下Python中高阶函数的原理">27.介绍一下Python中高阶函数的原理</h2>

高阶函数（Higher-Order Function）指：**接收函数作为参数**，或**返回函数**。

Python 中函数是一等公民：

```python
def add(x, y):
    return x + y

def logger(fn):
    def wrapper(*args, **kwargs):
        print("call", fn.__name__)
        return fn(*args, **kwargs)
    return wrapper

print(logger(add)(1, 2))
```

### 常见形态

1. **函数作为参数**：`map`、`sorted(key=...)`、`filter`、装饰器。
2. **函数作为返回值**：闭包、装饰器工厂、策略工厂。
3. **偏函数**：固定部分参数。

```python
from functools import partial, wraps

def chat(model: str, prompt: str, temperature: float = 0.7):
    ...

mini_chat = partial(chat, "gpt-4o-mini", temperature=0.2)
mini_chat("你好")
```

### 一线场景

```python
# 1) 统一重试包装
def with_retry(fn, times=3):
    @wraps(fn)
    def wrapper(*a, **k):
        for i in range(times):
            try:
                return fn(*a, **k)
            except Exception:
                if i == times - 1:
                    raise
    return wrapper

# 2) 排序/分组的 key 函数
docs.sort(key=lambda d: d["score"], reverse=True)

# 3) 回调与钩子
def run_training(hooks: dict):
    hooks.get("on_epoch_end", lambda *_: None)(epoch)
```

<font color=DeepSkyBlue>理解高阶函数，才能真正理解装饰器、闭包、策略模式和大部分 Python 框架的扩展机制。</font>


<h2 id="28.Python与C++有哪些区别？">28.Python与C++有哪些区别？</h2>

| 维度 | Python | C++ |
|------|--------|-----|
| 类型 | 动态、运行时 | 静态、编译期 |
| 执行 | 解释 + 字节码 | 编译为机器码 |
| 性能 | 纯 Python 慢 | 可极致优化 |
| 内存 | 自动 GC | 手动/智能指针 |
| 并发 | 受 GIL，原生并行弱 | 真线程/多核友好 |
| 生态 | AI/数据/脚本极强 | 游戏/引擎/底层/HPC |
| 开发速度 | 快 | 慢，但运行时控制强 |

### AI 行业的真实分工

- **Python**：训练脚本、数据处理、服务编排、Agent、实验迭代。
- **C++**：推理引擎（TensorRT/部分 vLLM 后端）、CUDA 算子、游戏/嵌入式、性能内核。

### 互操作常见方式

```python
# 1) pybind11 / nanobind：把 C++ 模块导出成 Python 包
# 2) ctypes / cffi：调用动态库
# 3) Torch 的 C++ 扩展 / Triton 写算子
```

面试结论：<font color=DeepSkyBlue>Python 是胶水与上层，C++ 是内核与极致性能；一线 AI 工程师至少要能读懂并调用 C++/CUDA 扩展的接口。</font>


<h2 id="29.Python与C语言有哪些区别？">29.Python与C语言有哪些区别？</h2>

| 维度 | Python | C |
|------|--------|---|
| 抽象层级 | 高级 | 低级 |
| 指针 | 无直接指针 | 指针核心 |
| 内存 | 自动 | malloc/free |
| 编译 | 解释执行 | 编译为机器码 |
| 典型用途 | 业务/数据/AI 应用 | OS、驱动、嵌入式、内核库 |

### 关键理解

1. CPython 本身是 C 写的；很多「Python 库」性能来自 C/Fortran/CUDA 内核。
2. 写热路径时的升级路径通常是：纯 Python → NumPy/Torch 向量化 → Cython/C 扩展 → CUDA。
3. C 的心智模型（内存布局、指针、缓存）有助于理解张量连续性、零拷贝、共享内存。

```python
# Python 侧感知连续性
import torch
x = torch.randn(4, 4)
print(x.is_contiguous())
y = x.t()
print(y.is_contiguous())  # False，转置后步长变了
```


<h2 id="30.在AI行业中，Python编程中的动态库和静态库的含义是什么？两者之间什么差异？">30.在AI行业中，Python编程中的动态库和静态库的含义是什么？两者之间什么差异？</h2>

在 Python 生态里，底层扩展多以原生库形式存在：

- **静态库**：编译期把目标代码链接进可执行文件/扩展模块（`.a` / `.lib`）。产物自包含，体积大，更新需重编。
- **动态库**：运行时加载共享库（`.so` / `.dll` / `.dylib`）。多个程序可共享，可单独升级，但依赖管理更复杂。

### 对照

| | 静态库 | 动态库 |
|--|--------|--------|
| 链接时机 | 编译期 | 运行期 |
| 产物 | 并入二进制 | 独立 .so/.dll |
| 升级 | 整体重编 | 可替换库文件 |
| 部署 | 简单、偏大 | 需保证依赖路径正确 |
| AI 场景 | 某些完全静态链接的推理二进制 | CUDA/cuDNN/TensorRT/torch 扩展的常态 |

### Python 中的实际意义

```python
import torch
print(torch.__version__)
# torch 依赖 CUDA 动态库：libcudart、libcublas、自定义 .so 算子
```

常见问题：

1. **找不到 .so/.dll**：`PATH` / `LD_LIBRARY_PATH` / `CUDA_HOME` 没配好。
2. **ABI 不匹配**：编译器、CUDA、Python 版本不一致导致 `undefined symbol`。
3. **Windows/Linux 混用**：不能把 Linux 的 `.so` 丢到 Windows 上。

<font color=DeepSkyBlue>部署 AI 服务时，动态库版本矩阵（Python、CUDA、cuDNN、驱动、torch）是最常见的环境类故障源。</font>


<h2 id="31.Python中的闭包是什么？在AI工程中有什么用？">31.Python中的闭包是什么？在AI工程中有什么用？</h2>

闭包：内层函数引用了外层函数的变量，且外层函数已返回，内层函数仍「捕获」这些变量。

```python
def make_multiplier(n):
    def multiplier(x):
        return x * n   # n 来自外层作用域
    return multiplier

double = make_multiplier(2)
print(double(5))  # 10
print(double.__closure__)  # 闭包单元
```

### 机制要点

1. 捕获的是变量本身（cell），不是某一时刻的值拷贝（对可变对象尤其要注意共享）。
2. `nonlocal` 可在内层修改外层绑定。
3. 闭包是装饰器的底层结构。

### AI 工程用法

**1. 装饰器工厂**

```python
def retry(times: int):
    def deco(fn):
        def wrapper(*a, **k):
            for _ in range(times):
                try:
                    return fn(*a, **k)
                except Exception:
                    last = True
            raise RuntimeError("failed")
        return wrapper
    return deco
```

**2. 配置偏函数 / 绑定客户端**

```python
def make_client(base_url: str, api_key: str):
    session = httpx.Client(base_url=base_url, headers={"Authorization": f"Bearer {api_key}"})
    def chat(messages):
        return session.post("/chat", json={"messages": messages})
    return chat
```

**3. 回调与钩子**

```python
def on_epoch(epoch: int):
    def hook(state):
        if state["val_loss"] < best[0]:
            best[0] = state["val_loss"]
            save(state)
    return hook
```

### 坑

```python
funcs = []
for i in range(3):
    funcs.append(lambda: i)  # 全部捕获同一 i

print([f() for f in funcs])  # [3, 3, 3]

# 修复：默认参数绑定
funcs = [lambda i=i: i for i in range(3)]
```

<font color=DeepSkyBlue>闭包用好了能写出很干净的可配置组件；用不好会造成意外的大对象捕获和延迟绑定 bug。</font>


<h2 id="32.Python中的元类metaclass是什么？">32.Python中的元类metaclass是什么？</h2>

元类是「类的类」。普通对象由类创建，类本身由元类创建。默认元类是 `type`。

```python
class Foo:
    pass

print(type(Foo))  # <class 'type'>
```

`type(name, bases, dict)` 可动态建类：

```python
def speak(self):
    return "hi"

Bar = type("Bar", (), {"speak": speak})
print(Bar().speak())
```

### 自定义元类

```python
class RegistryMeta(type):
    registry: dict = {}

    def __new__(mcs, name, bases, ns):
        cls = super().__new__(mcs, name, bases, ns)
        if name != "BaseTool":
            RegistryMeta.registry[name] = cls
        return cls

class BaseTool(metaclass=RegistryMeta):
    pass

class SearchTool(BaseTool):
    def run(self, q: str):
        return f"search {q}"

print(RegistryMeta.registry.keys())
```

### 一线何时用元类？

多数场景用装饰器/`__init_subclass__` 就够了：

```python
class BaseTool:
    _registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        BaseTool._registry[cls.__name__] = cls
```

真正用元类的场合：

1. ORM/校验框架在类创建时注入字段（pydantic 内部）。
2. 强制子类实现接口、自动注册插件。
3. 拦截类创建过程做审计/约束。

<font color=DeepSkyBlue>原则：能不用元类就不用；它是框架作者工具，不是业务代码首选。</font>


<h2 id="33.Python中的上下文管理器with和__enter__/__exit__有什么价值？">33.Python中的上下文管理器with和__enter__/__exit__有什么价值？</h2>

上下文管理器保证「进入时获取资源，退出时必定释放」，即使发生异常。

### 协议

```python
class Managed:
    def __enter__(self):
        print("enter")
        return self

    def __exit__(self, exc_type, exc, tb):
        print("exit", exc_type)
        return False  # 不吞异常

with Managed() as m:
    print("body")
```

### contextlib 更简洁

```python
from contextlib import contextmanager

@contextmanager
def timer(tag: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        print(tag, time.perf_counter() - t0)

with timer("infer"):
    model(x)
```

### AI 工程中的高价值用法

**1. GPU/设备上下文**

```python
import torch

with torch.no_grad():
    out = model(x)

with torch.autocast("cuda", dtype=torch.float16):
    out = model(x)
```

**2. 分布式与并行**

```python
with torch.device("cuda:0"):
    w = torch.randn(1024, 1024)

# 多卡
from accelerate import Accelerator
accelerator = Accelerator()
model, loader = accelerator.prepare(model, loader)
```

**3. 连接/会话/锁**

```python
with engine.begin() as conn:      # SQLAlchemy 事务
    conn.execute(stmt)

with lock, cache_slot as slot:    # 资源槽
    ...
```

**4. 临时配置/环境变量**

```python
from contextlib import contextmanager

@contextmanager
def temp_seed(seed: int):
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(state)
```

### 价值总结

1. 异常安全的资源释放（RAII 风格）。
2. 代码作用域清晰，减少「忘记 close」类 bug。
3. 可组合：`with A() as a, B() as b:`。


<h2 id="34.Python协程、asyncio和异步IO在AI Agent中如何使用？">34.Python协程、asyncio和异步IO在AI Agent中如何使用？</h2>

AI Agent 的主循环几乎全是 I/O：调 LLM、查工具、读写检索库、等待人工确认。`asyncio` 是一线最主流的并发骨架。

### 为什么 Agent 需要异步

1. 一次用户请求可能触发多次模型调用 + 多个工具调用。
2. 工具耗时不可控（搜索、代码执行、文件读写）。
3. 需要超时、取消、重试、并行 fan-out。

### 典型 Agent 异步骨架

```python
import asyncio
from dataclasses import dataclass

@dataclass
class ToolResult:
    name: str
    ok: bool
    data: str

async def call_llm(messages: list[dict]) -> dict:
    ...

async def run_tool(name: str, args: dict) -> ToolResult:
    await asyncio.sleep(0.1)  # 模拟 I/O
    return ToolResult(name, True, "ok")

async def agent_loop(user_input: str, max_steps: int = 8):
    messages = [{"role": "user", "content": user_input}]
    for _ in range(max_steps):
        action = await call_llm(messages)
        if action.get("type") == "final":
            return action["content"]

        # 并行执行工具
        tasks = [
            asyncio.wait_for(run_tool(t["name"], t["args"]), timeout=30)
            for t in action.get("tools", [])
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        messages.append({
            "role": "tool",
            "content": [
                r if isinstance(r, ToolResult) else ToolResult("error", False, str(r))
                for r in results
            ],
        })
    return "max steps exceeded"

# asyncio.run(agent_loop("查一下今天的新闻并总结"))
```

### 关键原语

| 原语 | 用途 |
|------|------|
| `asyncio.gather` | 并行一组协程 |
| `asyncio.wait_for` | 超时 |
| `asyncio.Semaphore` | 并发限流 |
| `asyncio.Queue` | 生产者消费者流水线 |
| `asyncio.TaskGroup`（3.11+） | 结构化并发 |
| `loop.run_in_executor` | 把同步/CPU 任务丢线程/进程池 |

### 与同步 SDK 混用

```python
# 第三方只有同步客户端时
result = await asyncio.to_thread(sync_client.chat, messages)
```

### 工程注意

1. 异步函数内部不要再调会阻塞事件循环的同步重接口。
2. 取消要传播：`except asyncio.CancelledError` 时清理资源后 re-raise。
3. 生产服务用 FastAPI/Starlette，天然 async；WSGI 同步服务需线程池。


<h2 id="35.Python设计模式在AI Agent系统中如何落地？">35.Python设计模式在AI Agent系统中如何落地？</h2>

Agent 系统是设计模式最能体现价值的地方之一。

### 1. 策略：不同的规划/工具选择策略

```python
class Planner:
    def plan(self, state) -> list[dict]:
        raise NotImplementedError

class ReActPlanner(Planner):
    def plan(self, state):
        return [{"tool": "search", "args": {"q": state["query"]}}]

class OneShotPlanner(Planner):
    def plan(self, state):
        return [{"tool": "final", "args": {"content": state["query"]}}]

class Agent:
    def __init__(self, planner: Planner):
        self.planner = planner

    def step(self, state):
        return self.planner.plan(state)
```

### 2. 观察者：轨迹与监控

```python
class Tracer:
    def __init__(self):
        self.events = []
    def on(self, event: str, payload: dict):
        self.events.append((event, payload))

agent_tracer = Tracer()
# 每个 step/tool_call/llm_request 都 emit，便于回放与评测
```

### 3. 责任链：中间件（鉴权、脱敏、限流、审计）

```python
class Middleware:
    async def __call__(self, request, call_next):
        raise NotImplementedError

class RateLimit(Middleware):
    def __init__(self, rpm: int):
        self.sem = asyncio.Semaphore(rpm)
    async def __call__(self, request, call_next):
        async with self.sem:
            return await call_next(request)
```

### 4. 备忘录：会话状态快照

```python
@dataclass
class AgentState:
    messages: list
    scratchpad: dict

    def snapshot(self):
        return copy.deepcopy(self)

    def restore(self, snap):
        self.messages = snap.messages
        self.scratchpad = snap.scratchpad
```

### 5. 代理：工具网关

```python
class ToolGateway:
    def __init__(self, tools: dict, policy):
        self.tools = tools
        self.policy = policy

    async def invoke(self, name, args):
        if not self.policy.allow(name, args):
            raise PermissionError(name)
        return await self.tools[name](**args)
```

### 6. 工厂：按配置构建 Agent 拓扑

```python
def build_agent(cfg: dict):
    planner = ReActPlanner() if cfg["mode"] == "react" else OneShotPlanner()
    tools = load_tools(cfg["tools"])
    return Agent(planner=planner, tools=ToolGateway(tools, DefaultPolicy()))
```

<font color=DeepSkyBlue>落地原则：模式服务于可替换、可观测、可测试；不要为了「用了模式」而过度抽象。</font>


<h2 id="36.Python对象池、连接池和模型池在AI服务中有什么区别？">36.Python对象池、连接池和模型池在AI服务中有什么区别？</h2>

三者都是「复用昂贵资源」，但池化的对象不同。

### 对象池（Object Pool）

- **池化对象**：任意创建成本高的对象（解析器、tokenizer、临时大缓冲）。
- **目标**：降低反复构造/析构开销。
- **注意**：对象重置逻辑必须正确，否则串状态。

```python
import queue

class ObjectPool:
    def __init__(self, factory, size: int):
        self.q = queue.Queue()
        for _ in range(size):
            self.q.put(factory())

    def acquire(self):
        return self.q.get()

    def release(self, obj):
        self.q.put(obj)
```

### 连接池（Connection Pool）

- **池化对象**：数据库/Redis/HTTP 连接。
- **目标**：复用 TCP/TLS 握手，控并发。
- **常见实现**：SQLAlchemy engine、`asyncpg` pool、`httpx` 连接池、Redis client。

```python
# SQLAlchemy
engine = create_engine(url, pool_size=10, max_overflow=20, pool_pre_ping=True)

# asyncpg
pool = await asyncpg.create_pool(dsn, min_size=5, max_size=20)
```

### 模型池（Model Pool）

- **池化对象**：已加载权重的模型实例/推理进程（GPU 上尤其关键）。
- **目标**：避免每次请求 `load_state_dict`；多实例吃满多卡；隔离显存。
- **形态**：
  1. 进程内单例 + 批处理（单卡服务）
  2. 多进程/多实例（每进程一张卡）
  3. 外部推理服务（vLLM/Triton/TGI）当远程池

```python
class ModelPool:
    def __init__(self, devices: list[str], factory):
        self.devices = devices
        self.factory = factory
        self._models = {}
        self._lock = asyncio.Lock()

    async def get(self, device: str):
        async with self._lock:
            if device not in self._models:
                self._models[device] = self.factory(device)
            return self._models[device]
```

### 对比

| | 对象池 | 连接池 | 模型池 |
|--|--------|--------|--------|
| 资源成本 | 中 | 中（网络） | 极高（权重/显存） |
| 典型问题 | 状态污染 | 连接失效/超时 | 显存碎片、OOM、多卡调度 |
| 归还语义 | reset | 保活/校验 | 常不归还，而是请求级借用 |
| 是否跨进程 | 常进程内 | 可 | 常多进程/独立服务 |

<font color=DeepSkyBlue>生产建议：模型推理不要写在请求路径里现载；要么进程内预热单例，要么下沉到 vLLM/Triton 这类专用服务。</font>


<h2 id="37.Python中如何设计插件化工具注册机制？">37.Python中如何设计插件化工具注册机制？</h2>

Agent/RAG/评测框架几乎都需要「可插拔工具」。目标：新增工具不改核心调度代码。

### 1. 装饰器注册（最常用）

```python
# registry.py
_TOOL_REGISTRY: dict[str, dict] = {}

def register_tool(name: str | None = None, description: str = "", schema: dict | None = None):
    def deco(fn):
        key = name or fn.__name__
        _TOOL_REGISTRY[key] = {
            "fn": fn,
            "description": description or (fn.__doc__ or "").strip(),
            "schema": schema or {},
        }
        return fn
    return deco

def get_tool(name: str):
    return _TOOL_REGISTRY[name]

def list_tools():
    return [
        {"name": k, "description": v["description"], "schema": v["schema"]}
        for k, v in _TOOL_REGISTRY.items()
    ]
```

```python
# tools/search.py
from registry import register_tool

@register_tool(
    name="web_search",
    description="Search the web",
    schema={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
)
async def web_search(query: str) -> str:
    return f"results for {query}"
```

### 2. 入口点自动发现（pkgutil / importlib）

```python
import importlib, pkgutil

def autodiscover(package_name: str = "app.tools"):
    pkg = importlib.import_module(package_name)
    for _, mod_name, _ in pkgutil.iter_modules(pkg.__path__):
        importlib.import_module(f"{package_name}.{mod_name}")
    # import 时装饰器已注册
```

### 3. 基于 ABC / Protocol 的插件类

```python
from typing import Protocol

class Tool(Protocol):
    name: str
    async def run(self, **kwargs) -> str: ...

class EchoTool:
    name = "echo"
    async def run(self, text: str) -> str:
        return text
```

### 4. 执行器与安全边界

```python
import asyncio

async def invoke_tool(call: dict) -> str:
    tool = get_tool(call["name"])
    try:
        result = await asyncio.wait_for(tool["fn"](**call["args"]), timeout=30)
        return str(result)
    except asyncio.TimeoutError:
        return f"tool {call['name']} timeout"
    except Exception as e:
        return f"tool {call['name']} failed: {e}"
```

### 5. 工程增强

1. **JSON Schema 自动生成**：从函数签名 + type hints 推导，供 LLM function calling。
2. **权限与配额**：注册时声明 `requires=["network"]`，运行时校验。
3. **版本与冲突**：同名工具注册时报错或覆盖策略可配置。
4. **热加载**：开发期 `importlib.reload`，生产用独立 worker 进程更新。
5. **测试**：每个工具提供 `dry_run` 或 Fake，避免单测打真外部服务。

### 一个更完整的最小闭环

```python
# __init__.py of tools package
autodiscover("app.tools")

TOOLS = list_tools()  # 交给 LLM 的 tool schema 列表

async def agent_step(user_msg: str, tool_calls: list[dict]):
    outputs = await asyncio.gather(*(invoke_tool(c) for c in tool_calls))
    return outputs
```

<font color=DeepSkyBlue>总结：装饰器注册 + 自动发现 + 统一 invoke 超时/错误处理，是当前一线 Agent 工具系统的标准骨架；Schema 从类型注解推导，可显著减少手写错误。</font>
