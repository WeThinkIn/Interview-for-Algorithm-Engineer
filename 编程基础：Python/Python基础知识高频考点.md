# 目录

- [1.Python是解释语言还是编译语言？](#1.python是解释语言还是编译语言？)
- [2.Python里有多线程吗？](#2.python里有多线程吗？)
- [3.Python中range和xrange的区别？](#3.python中range和xrange的区别？)
- [4.Python中列表和元组的区别？](#4.python中列表和元组的区别？)
- [5.Python中dict（字典）的底层结构？](#5.python中dict（字典）的底层结构？)
- [6.常用的深度学习框架有哪些，都是哪家公司开发的？](#6.常用的深度学习框架有哪些，都是哪家公司开发的？)
- [7.PyTorch动态图和TensorFlow静态图的区别？](#7.pytorch动态图和tensorflow静态图的区别？)
- [8.Python中assert的作用？](#8.python中assert的作用？)
- [9.Python中互换变量有不用创建临时变量的方法吗？](#9.python中互换变量有不用创建临时变量的方法吗？)
- [10.Python中的主要数据结构都有哪些？](#10.python中的主要数据结构都有哪些？)
- [11.Python中的可变对象和不可变对象？](#11.python中的可变对象和不可变对象？)
- [12.Python中None代表什么含义？](#12.python中none代表什么含义？)
- [13.Python中的实例方法、静态方法和类方法三者区别？](#13.python中的实例方法、静态方法和类方法三者区别？)
- [14.Python中常见的切片操作](#14.python中常见的切片操作)
- [15.Python中如何进行异常处理？](#15.python中如何进行异常处理？)
- [16.Python中remove，del以及pop之间的区别？](#16.python中remove，del以及pop之间的区别？)
- [17.Python中PIL和OpenCV处理图像的区别？](#17.python中pil和opencv处理图像的区别？)
- [18.Python中全局变量与局部变量之间的区别？](#18.python中全局变量与局部变量之间的区别？)
- [19.Python中`if "__name__" == __main__'`的作用?](#19.python中name==main?)
- [20.Python中assert的作用?](#20.python中assert的作用?)
- [21.python中如何无损打开图像，并无损保存图像?](#21.python中如何无损打开图像，并无损保存图像?)
- [22.PyTorch中张量操作Clone与Detach的区别?（腾讯实习二面）](#22.pytorch中张量操作clone与detach的区别?（腾讯实习二面）)
- [23.Python多进程中的fork和spawn模式有什么区别？](#23.python多进程中的fork和spawn模式有什么区别？)
- [24.什么是Python中的推导式？Python的推导式一共有多少种？](#24.什么是python中的推导式？python的推导式一共有多少种？)
- [25.python中一共都有哪些数据结构？](#25.python中一共都有哪些数据结构？)
- [26.python中index使用注意事项](#26.python中index使用注意事项)
- [27.Python中函数传参时会改变参数本身吗？](#27.python中函数传参时会改变参数本身吗？)
- [28.什么是python的全局解释器锁GIL？](#28.什么是python的全局解释器锁gil？)
- [29.什么是python的字符串格式化技术？](#29.什么是python的字符串格式化技术？)
- [30.Python中is和==的区别？](#30.python中is和==的区别？)
- [31.Python中type()和isinstance()的区别？](#31.python中type和isinstance的区别？)
- [32.Python中switch-case语句的实现？](#32.python中switch-case语句的实现？)
- [33.介绍一下Python中耦合和解耦的代码设计思想](#33.介绍一下python中耦合和解耦的代码设计思想)
- [34.Python中的函数参数有哪些类型与规则？](#34.python中的函数参数有哪些类型与规则？)
- [35.什么是Python中的魔术方法?](#35.什么是python中的魔术方法？)
- [36.介绍一下Python中常用的标准库以及功能](#36.介绍一下python中常用的标准库以及功能)
- [37.python中有哪些内建数据类型？](#37.python中有哪些内建数据类型？)
- [38.python中文件有哪些打开模式，它们的区别是什么？](#38.python中文件有哪些打开模式，它们的区别是什么？)
- [39.python中eval函数的作用？](#39.python中eval函数的作用？)
- [40.python中海象运算符的介绍](#40.python中海象运算符的介绍)
- [41.Python中tuple、list和dict有什么区别？](#41.python中tuple、list和dict有什么区别？)
- [42.为什么说Python是动态语言？](#42.为什么说python是动态语言？)
- [43.介绍一下Python中logging库的作用](#43.介绍一下python中logging库的作用)
- [44.Python中单下划线、双下划线、前后双下划线分别代表什么？](#44.python中单下划线双下划线前后双下划线分别代表什么)
- [45.Python中类变量和实例变量有什么区别？](#45.python中类变量和实例变量有什么区别)
- [46.Python中__new__和__init__有什么区别？](#46.python中__new__和__init__有什么区别)
- [47.Python中文件读取read、readline、readlines和迭代文件对象有什么区别？](#47.python中文件读取readreadlinereadlines和迭代文件对象有什么区别)
- [48.Python中的类型注解、dataclass和Pydantic在AI工程中有什么价值？](#48.python中的类型注解dataclass和pydantic在ai工程中有什么价值)
- [49.Python 3.13/3.14之后，GIL、JIT、t-string、延迟注解有哪些新变化？](#49.python-313314之后giljitt-string延迟注解有哪些新变化)
- [50.AIGC和AI Agent项目中，Python基础能力应该重点掌握哪些？](#50.aigc和ai-agent项目中python基础能力应该重点掌握哪些)


<h2 id="1.python是解释语言还是编译语言？">1.Python是解释语言还是编译语言？</h2>

准确说：Python 是<font color=DeepSkyBlue>先编译成字节码，再由虚拟机解释执行</font>，日常语境下归为解释型语言。

- 源码 `.py` 会被编译成 `.pyc`（字节码），由 CPython 解释器逐条执行。
- 优点：跨平台、迭代快、胶水能力强。
- 缺点：纯 Python 热路径比 C/C++/Rust 慢。

一线常见分工：Python 负责编排与业务，计算密集部分交给 NumPy/Torch/CUDA/C++ 扩展。


<h2 id="2.python里有多线程吗？">2.Python里有多线程吗？</h2>

<font color=DeepSkyBlue>有，但 CPython 长期受 GIL 影响，同一时刻只有一个线程执行 Python 字节码</font>。

- **I/O 密集**：多线程有效（等网络/磁盘时会释放 GIL）。
- **CPU 密集**：多线程几乎无收益，应使用多进程、原生库或 GPU。
- **高并发网络调用**：一线更推荐 `asyncio`，而不是开大量线程。

```python
import threading, time

def io_task():
    time.sleep(1)  # 模拟 I/O，会释放 GIL

threads = [threading.Thread(target=io_task) for _ in range(10)]
for t in threads: t.start()
for t in threads: t.join()
```

缓解路径：多进程、协程/异步、把计算下沉到 C/CUDA、以及 3.13+ 的 free-threaded 构建（生态仍在完善）。


<h2 id="3.python中range和xrange的区别？">3.Python中range和xrange的区别？</h2>

这是 Python 2 时代的考点。在 **Python 3 中只有 `range`，没有 `xrange`**。

| | Python 2 `range` | Python 2 `xrange` | Python 3 `range` |
|--|------------------|-------------------|------------------|
| 返回 | list | xrange 对象（惰性） | range 对象（惰性） |
| 内存 | 大 | 小 | 小 |

```python
# Python 3
r = range(10)
print(type(r))       # <class 'range'>
print(list(r))       # [0, 1, ..., 9]
print(r[3], 3 in r)  # 支持索引和成员判断，O(1)

for i in range(0, 10, 2):
    print(i)
```

面试可答：`xrange` 是 Py2 遗留；Py3 的 `range` 已融合其惰性优点。


<h2 id="4.python中列表和元组的区别？">4.Python中列表和元组的区别？</h2>

1. <font color=DeepSkyBlue>列表可变</font>，创建后可增删改。
2. <font color=DeepSkyBlue>元组不可变</font>，可作只读序列，也常作 dict 的 key。
3. 元组内存通常更省、构造更快（尤其作为返回值打包）。
4. 元组不是“不能赋值给变量”，而是“不能原地修改内部结构”。

```python
coords = (10.0, 20.0)          # 不可变，适合做 key / 常量
batch = [1, 2, 3]              # 可变，适合累积
batch.append(4)

# 函数返回多值时本质是元组
def min_max(xs):
    return min(xs), max(xs)
```

一线建议：配置常量、坐标、API 返回结构优先 tuple；需要动态修改用 list。


<h2 id="5.python中dict（字典）的底层结构？">5.Python中dict（字典）的底层结构？</h2>

dict 是<font color=DeepSkyBlue>哈希表</font>，平均查找 O(1)。CPython 使用开放寻址 + 紧凑字典（节省内存），并保持插入顺序（3.7+ 语言保证）。

要点：

1. key 必须可哈希（`__hash__` 且 `__eq__` 稳定）。
2. 冲突通过探测解决；装载因子过高会扩容。
3. 查找/插入/删除均摊 O(1)。

```python
# 常见用法：配置合并、索引、缓存
cfg = {"lr": 1e-4}
cfg.update({"epochs": 3})
print(cfg.get("batch_size", 32))  # 安全读取

# 成员判断用 dict/set，不要用 list
allowed = {"gpt-4o", "claude-3", "qwen"}
if model_name in allowed:
    ...
```


<h2 id="6.常用的深度学习框架有哪些，都是哪家公司开发的？">6.常用的深度学习框架有哪些，都是哪家公司开发的？</h2>

| 框架 | 主导方 | 现状备注 |
|------|--------|----------|
| PyTorch | Meta（原 Facebook） | 学术与大模型训练主流 |
| TensorFlow / Keras | Google | TF2 统一 Keras API；工业部署仍有存量 |
| JAX | Google | 函数式变换，高性能研究向 |
| PaddlePaddle | 百度 | 国内产业落地较多 |
| MindSpore | 华为 | 昇腾生态 |
| MXNet | Apache（曾 Amazon 主导） | 维护已明显减弱 |
| OneFlow | OneFlow 国产开源 | 分布式训练方向 |

推理/服务侧常见：vLLM、TensorRT / TensorRT-LLM、ONNX Runtime、OpenVINO、llama.cpp 等。

一线现状：<font color=DeepSkyBlue>训练侧 PyTorch 占绝对主导；推理侧多框架并存，按硬件与延迟要求选型</font>。


<h2 id="7.PyTorch动态图和TensorFlow静态图的区别？">7.PyTorch动态图和TensorFlow静态图的区别？</h2>

这是 TF1.x 时代的经典对比。现状是：**PyTorch 默认 eager（动态图）；TensorFlow 2.x 默认 eager，也可用 `tf.function` 转静态图**。

| | 动态图（Eager） | 静态图（Graph） |
|--|----------------|----------------|
| 定义方式 | 命令式，边执行边构图 | 先定义计算图再运行 |
| 调试 | 接近普通 Python，易断点 | 较难，需要图调试工具 |
| 灵活性 | 控制流随数据变 | 需特殊算子（tf.cond 等） |
| 优化/部署 | 依赖 trace/script | 图优化、算子融合友好 |

```python
import torch

def f(x):
    if x.sum() > 0:
        return x * 2
    return x / 2

x = torch.randn(4)
print(f(x))  # 动态图：Python if 直接可用

# 需要部署/编译时再静态化
compiled = torch.compile(f)  # PyTorch 2.x
```

面试结论：不要背“PyTorch 动态、TF 静态”的旧结论；应答<font color=DeepSkyBlue>两边现在都支持 eager + 静态化/编译路径</font>，核心差异在生态与调试体验。


<h2 id="8.Python中assert的作用？">8.Python中assert的作用？</h2>

`assert` 用于断言条件为真，失败时抛 `AssertionError`。用来表达“开发者认为这里必然成立”的不变量。

```python
def softmax(x):
    assert x.ndim >= 1, "expect at least 1-d input"
    ...
```

注意：

1. `python -O` 会去掉 assert，<font color=DeepSkyBlue>不要用它做对外输入校验</font>。
2. 对外参数、用户输入、API 边界应用显式 `if` + 业务异常，或 Pydantic。
3. assert 适合内部契约、单测、算法前置条件。

```python
# 坏：生产校验用 assert
assert user_age >= 0  # -O 下消失

# 好
if user_age < 0:
    raise ValueError("age must be >= 0")
```


<h2 id="9.Python中互换变量有不用创建临时变量的方法吗？">9.Python中互换变量有不用创建临时变量的方法吗？</h2>

有，Python 元组打包/解包是最常见写法：

```python
a, b = 1, 2
a, b = b, a
print(a, b)  # 2 1
```

也可用异或（仅整数演示，可读性差，工程不推荐）：

```python
a ^= b
b ^= a
a ^= b
```

一线常用场景：

```python
left, right = right, left
key, value = mapping.popitem()
```


<h2 id="10.Python中的主要数据结构都有哪些？">10.Python中的主要数据结构都有哪些？</h2>

内置：`list`、`tuple`、`dict`、`set`、`frozenset`、`str`、`bytes`、`bytearray`、`range`。

标准库补充：`collections.deque`、`Counter`、`defaultdict`、`OrderedDict`、`heapq`、`queue`、`dataclasses`。

| 结构 | 有序 | 可变 | 典型用途 |
|------|------|------|----------|
| list | 是 | 是 | 通用序列 |
| tuple | 是 | 否 | 不可变记录、dict key |
| dict | 插入序 | 是 | 键值索引、配置 |
| set | 否 | 是 | 去重、成员判断 |
| deque | 是 | 是 | 两端 O(1) 操作、滑动窗口 |
| Counter | 插入序 | 是 | 计数 |

```python
from collections import Counter, deque, defaultdict
import heapq

tokens = ["a", "b", "a"]
print(Counter(tokens))          # Counter({'a': 2, 'b': 1})

recent = deque(maxlen=100)      # 无界日志的替代
scores = [3, 1, 2]
heapq.heapify(scores)           # 最小堆
```


<h2 id="11.Python中的可变对象和不可变对象？">11.Python中的可变对象和不可变对象？</h2>

- <font color=DeepSkyBlue>不可变</font>：`int`、`float`、`str`、`bytes`、`tuple`、`frozenset`。修改即新建对象。
- <font color=DeepSkyBlue>可变</font>：`list`、`dict`、`set`、`bytearray`、自定义实例。可原地修改。

```python
a = (1, 2, [3])
a[2].append(4)     # 合法：元组外壳不可变，内部 list 可变
print(a)           # (1, 2, [3, 4])
```

工程影响：

1. 可变默认参数陷阱。
2. 多处引用同一 list/dict 会互相影响（拷贝语义）。
3. 可变对象不能做 dict key / set 元素。


<h2 id="12.Python中None代表什么含义？">12.Python中None代表什么含义？</h2>

`None` 是 `NoneType` 的唯一实例，表示“无值/空/未初始化”。

```python
x = None
print(x is None)      # 推荐
# print(x == None)    # 不推荐，自定义 __eq__ 可能干扰

def find_model(name: str):
    return registry.get(name)  # 不存在返回 None

model = find_model("llama")
if model is None:
    raise KeyError("model not found")
```

一线建议：能用哨兵就用哨兵区分“缺省”和“合法空值”；API 边界用 Optional/None + 显式校验（Pydantic）。


<h2 id="13.Python中的实例方法、静态方法和类方法三者区别？">13.Python中的实例方法、静态方法和类方法三者区别？</h2>

| | 实例方法 | classmethod | staticmethod |
|--|---------|-------------|--------------|
| 首参 | `self` | `cls` | 无 |
| 绑定 | 实例 | 类 | 不绑定 |
| 典型 | 业务逻辑 | 工厂、缓存、多态构造 | 纯工具函数 |

```python
import torch

class Engine:
    _cache = {}

    def __init__(self, name: str):
        self.name = name

    def infer(self, x):
        return x  # 实例方法

    @classmethod
    def from_pretrained(cls, name: str):
        if name not in cls._cache:
            cls._cache[name] = cls(name)
        return cls._cache[name]

    @staticmethod
    def validate_name(name: str) -> bool:
        return bool(name) and name.isalnum()
```

详见进阶篇第 8 题（模型重复加载案例）。


<h2 id="14.Python中常见的切片操作">14.Python中常见的切片操作</h2>

语法：`seq[start:stop:step]`（含 start，不含 stop）。

```python
s = "hello world"
print(s[0:5])     # hello
print(s[::-1])    # dlrow olleh
print(s[::2])     # hlowrd

xs = list(range(10))
print(xs[2:8:2])  # [2, 4, 6]
print(xs[-3:])    # [7, 8, 9]

# 切片产生浅拷贝
a = [1, 2, 3]
b = a[:]          # 新 list，元素引用相同
a[0] = 99
print(b)          # [1, 2, 3]
```

注意：切片不会 IndexError，越界返回空序列；步长为负表示反向。


<h2 id="15.Python中如何进行异常处理？">15.Python中如何进行异常处理？</h2>

使用 `try/except/else/finally`，可多层 except，自定义异常继承 `Exception`。

```python
import httpx

def call_api(url: str):
    try:
        resp = httpx.get(url, timeout=5)
        resp.raise_for_status()
    except httpx.TimeoutException:
        return None, "timeout"
    except httpx.HTTPStatusError as e:
        return None, f"http {e.response.status_code}"
    except Exception:
        raise
    else:
        return resp.json(), None
    finally:
        # 无论成败都执行：释放连接、打点
        pass
```

工程原则：

1. 捕获尽量窄的异常类型，避免裸 `except:`。
2. 不要用异常做普通控制流。
3. 日志要带上下文（request_id、model、trace）。
4. 对外服务统一错误模型（FastAPI `HTTPException`）。


<h2 id="16.Python中remove，del以及pop之间的区别？">16.Python中remove，del以及pop之间的区别？</h2>

| 操作 | 作用 | 返回 | 失败行为 |
|------|------|------|----------|
| `list.remove(x)` | 删除第一个值为 x 的元素 | None | ValueError |
| `del lst[i]` | 删除下标 i | None | IndexError |
| `list.pop(i=-1)` | 删除并返回下标 i | 元素值 | IndexError |
| `dict.pop(k)` | 删键并返回值 | 值 | KeyError（可给默认值） |

```python
xs = [1, 2, 3, 2]
xs.remove(2)     # [1, 3, 2]
last = xs.pop()  # 2
del xs[0]        # [3]

d = {"a": 1}
print(d.pop("b", None))  # None，安全
```


<h2 id="17.Python中PIL和OpenCV处理图像的区别？">17.Python中PIL和OpenCV处理图像的区别？</h2>

| | Pillow (PIL) | OpenCV |
|--|--------------|--------|
| 定位 | 通用图像读写/简单处理 | 计算机视觉全栈 |
| 颜色 | RGB | 默认 BGR |
| 依赖 | 轻 | 较重（含大量 CV 算子） |
| 与深度学习 | torchvision 常用 | 传统视觉、部署前处理常见 |

```python
from PIL import Image
import numpy as np

img = Image.open("cat.jpg").convert("RGB")
arr = np.asarray(img)  # HWC, RGB

# 训练前处理更常见 torch/PIL；传统视觉流水线常用 cv2
```

注意通道顺序：`cv2.imread` 得到 BGR，进 torch 前常要 `[:, :, ::-1]` 或 `cv2.cvtColor`。


<h2 id="18.Python中全局变量与局部变量之间的区别？">18.Python中全局变量与局部变量之间的区别？</h2>

- 局部变量：函数内定义，作用域仅函数。
- 全局变量：模块顶层定义，整个模块可见。
- 函数内默认读全局；写全局需 `global`；嵌套函数改外层需 `nonlocal`。

```python
cache = {}  # 模级全局，慎用

def put(key, value):
    global cache
    cache[key] = value

def make_counter():
    n = 0
    def inc():
        nonlocal n
        n += 1
        return n
    return inc
```

一线建议：少用可变全局状态；用类、依赖注入、模块级常量、上下文对象传递。多线程共享可变全局要加锁。


<h2 id="19.Python中name==main?">19.Python中`if "__name__" == __main__'`的作用?</h2>

当模块被直接运行时 `__name__` 为 `"__main__"`；被 import 时为模块名。

作用：

1. 脚本入口与可复用库代码分离。
2. **Windows/macOS spawn 多进程必须保护入口**，否则子进程会再次执行主逻辑。

```python
def main():
    print("run as script")

if __name__ == "__main__":
    main()
```

```python
# 多进程模板（Windows 必写）
from multiprocessing import Process

def worker():
    print("hi")

if __name__ == "__main__":
    Process(target=worker).start()
```


<h2 id="20.Python中assert的作用?">20.Python中assert的作用?</h2>

与第 8 题相同：断言内部不变量；`-O` 会移除；不用于对外校验。

```python
def top_k(scores, k):
    assert 0 < k <= len(scores)
    ...
```

测试中很常用：

```python
def test_chunk():
    assert len(chunk("abc", 2)) == 2
```


<h2 id="21.python中如何无损打开图像，并无损保存图像?">21.python中如何无损打开图像，并无损保存图像?</h2>

“无损”取决于格式与参数，不是所有保存都无损。

```python
from PIL import Image

# PNG/WebP(lossless)/BMP 相对无损；JPEG 本身有损
img = Image.open("input.png")
img.save("output.png", optimize=True)           # 无损
img.save("output.webp", lossless=True, quality=100)

# 避免不必要的模式转换
img = Image.open("input.png")  # 保持 mode，别无谓 convert RGB
img.save("output.png")
```

注意：

1. JPEG `quality=95` 仍是有损；要无损请换 PNG/WebP lossless。
2. 处理浮点图像（科学数据）用 NumPy/`tifffile`，别走 8-bit JPEG。
3. 元数据（EXIF）可能在重编码时丢失，需要 `exif`/`pnginfo` 显式保留。


<h2 id="22.pytorch中张量操作clone与detach的区别?（腾讯实习二面）">22.PyTorch中张量操作Clone与Detach的区别?（腾讯实习二面）</h2>

```python
import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)

y = x.clone()       # 复制数据，仍可参与梯度（若有）
z = x.detach()      # 不复制底层数据，共享 storage，脱离计算图
w = x.detach().clone()  # 常见组合：断梯度 + 独立副本
```

| | clone | detach | detach().clone() |
|--|-------|--------|------------------|
| 复制数据 | 是 | 否（共享存储） | 是 |
| 梯度 | 保留原 requires_grad 语义 | 永远不需要梯度 | 不需要梯度 |
| 改副本影响原张量？ | 否 | 是（共享内存时） | 否 |

一线高频坑：只 `detach()` 后改副本，原张量也被改；只 `clone()` 可能仍挂在计算图上。推断保存中间结果常用 `x.detach().cpu()` 或 `x.detach().clone()`。


<h2 id="23.python多进程中的fork和spawn模式有什么区别？">23.Python多进程中的fork和spawn模式有什么区别？</h2>

1. Windows/macOS 默认 **spawn**（Windows 仅支持 spawn）；Linux 默认 **fork**。
2. spawn：不继承父进程内存与状态，启动慢，更安全（尤其有线程/Qt/CUDA 时）。
3. fork：复制父进程，启动快，但可能继承锁、线程、文件描述符，导致诡异死锁。

```python
import multiprocessing as mp

# 显式指定
ctx = mp.get_context("spawn")  # 推荐跨平台写法
p = ctx.Process(target=worker, args=(...))
```

实践：跨平台库统一 `spawn`；入口必须 `if __name__ == "__main__":`；CUDA 上不要 fork。


<h2 id="24.什么是Python中的推导式？Python的推导式一共有多少种？">24.什么是Python中的推导式？Python的推导式一共有多少种？</h2>

共 4 种：列表、字典、集合、生成器推导式。

```python
# 列表
squares = [x ** 2 for x in range(10)]
# 字典
mapping = {x: x ** 2 for x in range(5)}
# 集合
unique = {x % 3 for x in range(10)}
# 生成器（惰性）
gen = (x ** 2 for x in range(10))
```

带条件与嵌套：

```python
pairs = [(i, j) for i in range(3) for j in range(3) if i < j]
```

一线注意：推导式过长会伤可读性，改写成普通循环或具名函数；大数据优先生成器，避免一次性物化。


<h2 id="25.python中一共都有哪些数据结构？">25.python中一共都有哪些数据结构？</h2>

与第 10 题互补，再补几个 AI 工程高频结构：

```python
from collections import defaultdict, Counter, deque
from dataclasses import dataclass, field
import heapq

# 检索候选
cand = defaultdict(list)
cand["query1"].append({"doc": "d1", "score": 0.8})

# 滑动窗口指标
window = deque(maxlen=100)
window.append(0.12)

# TopK
top = heapq.nlargest(3, scores)

@dataclass
class Message:
    role: str
    content: str
    tool_calls: list = field(default_factory=list)
```


<h2 id="26.python中index使用注意事项">26.python中index使用注意事项</h2>

`list.index(x)` 返回第一个匹配下标；找不到抛 `ValueError`。

```python
xs = [1, 2, 3, 2]
print(xs.index(2))       # 1（只返回首次出现）
print(xs.index(2, 2))    # 3（从下标 2 起找）

# 安全写法
i = xs.index(2) if 2 in xs else -1
# 或
try:
    i = xs.index(9)
except ValueError:
    i = None
```

注意：

1. 有重复元素时别假设“唯一索引”。
2. 查多个匹配用列表推导/`enumerate`。
3. dict 键查找 O(1)，比在 list 上反复 `index` 快得多。


<h2 id="27.python中函数传参时会改变参数本身吗？">27.Python中函数传参时会改变参数本身吗？</h2>

Python 是<font color=DeepSkyBlue>对象引用传递</font>（既不是纯值拷贝，也不是 C++ 意义上的引用引用）。

- 重新绑定参数名（`x = ...`）不影响外部。
- 原地修改可变对象（`x.append`）会影响外部。

```python
def rebind(x):
    x = [9, 9]          # 不影响外部

def mutate(x):
    x.append(9)         # 影响外部

a = [1]
rebind(a)
print(a)                # [1]

mutate(a)
print(a)                # [1, 9]
```

默认参数陷阱：

```python
# 错误
def add_item(item, bucket=[]):
    bucket.append(item)
    return bucket

# 正确
def add_item(item, bucket=None):
    if bucket is None:
        bucket = []
    bucket.append(item)
    return bucket
```


<h2 id="28.什么是python的全局解释器锁gil？">28.什么是python的全局解释器锁GIL？</h2>

GIL 是 CPython 的全局解释器锁，保证同一时刻只有一个线程执行 Python 字节码。

影响：

1. CPU 密集多线程无法吃满多核。
2. I/O 时会释放 GIL，所以 I/O 多线程仍有效。
3. 许多 C 扩展（NumPy、加密、压缩）在内部计算时会主动释放 GIL。

```python
# CPU 密集：用进程
from concurrent.futures import ProcessPoolExecutor

def heavy(n):
    return sum(i * i for i in range(n))

if __name__ == "__main__":
    with ProcessPoolExecutor(4) as ex:
        list(ex.map(heavy, [10**6] * 4))
```

新进展：3.13+ 提供 free-threaded 构建（实验/逐步生产可用），但 C 扩展兼容性仍需评估。


<h2 id="29.什么是python的字符串格式化技术？">29.什么是python的字符串格式化技术？</h2>

主流演进：

```python
name, score = "model", 0.91

# 1) % 格式化（老）
print("score=%.2f" % score)

# 2) str.format
print("name={} score={:.2f}".format(name, score))

# 3) f-string（推荐）
print(f"name={name} score={score:.2f}")

# 4) 模板/安全场景可考虑 template string（3.14 t-string）或 string.Template
```

```python
# 结构化日志不要靠 f-string 拼大 SQL/HTML
import logging
logging.info("request done", extra={"model": name, "score": score})
```

一线：日志与业务字符串优先 f-string；对外拼接关注注入风险（SQL/HTML/Shell）。


<h2 id="30.Python中is和==的区别？">30.Python中is和==的区别？</h2>

- `is`：比较身份（`id`），是否同一对象。
- `==`：比较值（可重载 `__eq__`）。

```python
a = [1, 2]
b = [1, 2]
print(a == b)   # True
print(a is b)   # False

x = None
print(x is None)  # True，推荐

# 小整数/驻留字符串可能 is 也为 True，但不可依赖
```

经验：判断 None/True/False/哨兵对象用 `is`；比较业务值用 `==`。


<h2 id="31.Python中type和isinstance的区别？">31.Python中type()和isinstance()的区别？</h2>

- `type(x)` 返回精确类型。
- `isinstance(x, T)` 接受继承关系，更常用。

```python
class Animal: ...
class Dog(Animal): ...

d = Dog()
print(type(d) is Dog)          # True
print(isinstance(d, Animal))   # True
print(type(d) is Animal)       # False
```

需要严格类型时用 `type(x) is T`；大多数业务判断用 `isinstance`。多重类型：`isinstance(x, (int, float))`。


<h2 id="32.Python中switch-case语句的实现？">32.Python中switch-case语句的实现？</h2>

Python 3.10+ 有 `match-case`（结构模式匹配）；更早版本用 if-elif、dict 分发或策略对象。

```python
# 3.10+ match
def handle(action: str):
    match action:
        case "search":
            return do_search()
        case "calc":
            return do_calc()
        case _:
            raise ValueError(action)
```

```python
# dict 分发（配置化更友好）
handlers = {
    "search": do_search,
    "calc": do_calc,
}

def handle(action: str):
    fn = handlers.get(action)
    if fn is None:
        raise ValueError(action)
    return fn()
```

Agent 工具路由场景更推荐 dict/注册表，便于插件化扩展。


<h2 id="33.介绍一下python中耦合和解耦的代码设计思想">33.介绍一下Python中耦合和解耦的代码设计思想</h2>

- **耦合**：模块间依赖过紧，改一处牵一发动全身。
- **解耦**：通过接口、依赖注入、事件、策略等降低直接依赖。

```python
# 紧耦合：内部直接 new 具体实现
class RAG:
    def __init__(self):
        self.llm = OpenAIClient()  # 写死

# 解耦：依赖注入
class RAG:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

rag = RAG(llm=LocalLLM(), retriever=FaissRetriever())
```

其他手段：Protocol/ABC 定义接口、回调/事件总线、配置驱动工厂、把副作用放到边界。


<h2 id="34.python中的函数参数有哪些类型与规则？">34.Python中的函数参数有哪些类型与规则？</h2>

顺序：位置参数 → 默认参数 → `*args` → 仅关键字参数 → `**kwargs`。

```python
def train(data, epochs=1, *, lr=1e-4, **kwargs):
    ...
```

| 类型 | 示例 | 说明 |
|------|------|------|
| 位置 | `f(a, b)` | 必需 |
| 默认 | `f(a=1)` | 可省略 |
| 可变位置 | `*args` | 打包为 tuple |
| 仅关键字 | `*` 之后 | 必须用关键字 |
| 可变关键字 | `**kwargs` | 打包为 dict |

注意：可变默认参数不要写 `[]`/`{}`。


<h2 id="35.什么是python中的魔术方法？">35.什么是Python中的魔术方法?</h2>

双下划线方法，用于接入 Python 数据模型。

```python
class Vector:
    def __init__(self, x, y):
        self.x, self.y = x, y
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"
    def __len__(self):
        return 2
    def __getitem__(self, i):
        return (self.x, self.y)[i]
```

常见：`__init__`/`__new__`/`__call__`/`__enter__`/`__exit__`/`__iter__`/`__next__`/`__eq__`/`__hash__`/`__str__`/`__repr__`。

自定义可迭代工具、上下文管理器、可调用策略对象时几乎必用。


<h2 id="36.介绍一下python中常用的标准库以及功能">36.介绍一下Python中常用的标准库以及功能</h2>

| 模块 | 用途 |
|------|------|
| `os`/`sys`/`pathlib` | 路径、环境、解释器 |
| `json`/`csv`/`sqlite3` | 数据序列化与存储 |
| `re` | 正则 |
| `datetime`/`time` | 时间 |
| `logging` | 日志 |
| `typing`/`dataclasses` | 类型与数据类 |
| `collections`/`heapq`/`itertools`/`functools` | 容器与函数工具 |
| `asyncio`/`threading`/`multiprocessing`/`concurrent.futures` | 并发 |
| `subprocess`/`shutil` | 进程与文件 |
| `unittest`/`doctest` | 测试 |
| `argparse`/`tomllib` | CLI 与配置 |

```python
from pathlib import Path
import json

for p in Path("data").glob("*.jsonl"):
    with p.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
```


<h2 id="37.python中有哪些内建数据类型？">37.python中有哪些内建数据类型？</h2>

- 数值：`int`、`float`、`complex`、`bool`
- 序列：`list`、`tuple`、`range`、`str`、`bytes`、`bytearray`
- 映射：`dict`
- 集合：`set`、`frozenset`
- 其他：`None`、函数、类、模块、异常等对象类型

```python
print(type(True), isinstance(True, int))  # bool 是 int 子类
```


<h2 id="38.python中文件有哪些打开模式，它们的区别是什么？">38.python中文件有哪些打开模式，它们的区别是什么？</h2>

| 模式 | 含义 |
|------|------|
| `r` | 只读（默认） |
| `w` | 写，清空原文件 |
| `a` | 追加 |
| `x` | 独占创建，已存在则失败 |
| `b` | 二进制 |
| `t` | 文本（默认） |
| `+` | 读写 |

```python
with open("log.txt", "a", encoding="utf-8") as f:
    f.write("line\n")

with open("model.bin", "rb") as f:
    data = f.read()
```

注意：Windows 换行、编码显式 `encoding="utf-8"`、大文件别一次 `read()`。


<h2 id="39.python中eval函数的作用？">39.python中eval函数的作用？</h2>

`eval(expr)` 执行字符串表达式并返回结果。极度危险，等同于执行任意代码。

```python
# 不要用在用户输入上
# result = eval(user_input)

# 安全替代
import ast
# 仅解析字面量
config = ast.literal_eval("{'lr': 0.1}")  # 仍只应来自可信来源
```

一线规则：<font color=DeepSkyBlue>生产代码禁止对不可信输入 eval/exec</font>。配置用 JSON/YAML+schema；表达式求值用专用解析器。


<h2 id="40.python中海象运算符的介绍">40.python中海象运算符的介绍</h2>

`:=` 在表达式中赋值（Python 3.8+）。

```python
while (n := len(data)) > 0:
    print(n)
    data.pop()

if (m := re.search(r"\d+", text)) is not None:
    print(m.group())
```

适合：避免重复计算/重复调用、条件内赋值。过度使用会伤可读性。


<h2 id="41.python中tuple、list和dict有什么区别？">41.Python中tuple、list和dict有什么区别？</h2>

| | tuple | list | dict |
|--|-------|------|------|
| 可变性 | 否 | 是 | 是 |
| 有序 | 是 | 是 | 插入序（3.7+） |
| 访问 | 下标 | 下标 | 键 |
| 可作 key | 是（元素需可哈希） | 否 | 键本身需可哈希 |
| 典型 | 记录、坐标、多返回值 | 动态序列 | 索引、配置、缓存 |


<h2 id="42.为什么说python是动态语言？">42.为什么说Python是动态语言？</h2>

1. 变量无静态类型声明，类型在运行时绑定。
2. 对象类型运行时确定，鸭子类型普遍。
3. 可动态增删属性/方法，自省与元编程能力强。

```python
x = 1
x = "now a string"  # 合法

def greet(obj):
    return obj.speak()  # 只要求有 speak
```

类型注解/Pydantic 可以补上“工程上的静态约束”，但不改变动态语言本质。


<h2 id="43.介绍一下python中logging库的作用">43.介绍一下Python中logging库的作用</h2>

标准库日志：分级（DEBUG/INFO/WARNING/ERROR/CRITICAL）、Handler、Formatter、过滤器、层级 logger。

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("agent")

logger.info("tool finished", extra={"tool": "search", "latency_ms": 120})
```

一线建议：

1. 库代码只 `getLogger(__name__)`，不要 `basicConfig`。
2. 服务统一 JSON 日志 + request_id/trace_id。
3. 不要只靠 print；敏感信息（key、完整 prompt）要脱敏。


<h2 id="44.python中单下划线双下划线前后双下划线分别代表什么">44.Python中单下划线、双下划线、前后双下划线分别代表什么？</h2>

| 写法 | 含义 |
|------|------|
| `_x` | 约定“内部使用” |
| `__x` | 类私有（name mangling：`_Class__x`） |
| `__x__` | 魔术方法/特殊属性，不要自造 |
| `_` | 一次性占位/无关变量 |

```python
class Client:
    def __init__(self):
        self.api_key = "public-ish"   # 公开
        self._session = None          # 内部
        self.__secret = "s"           # 真·类私有

    def __repr__(self):               # 魔术方法
        return "Client(...)"
```


<h2 id="45.python中类变量和实例变量有什么区别">45.Python中类变量和实例变量有什么区别？</h2>

- 类变量：属于类，所有实例共享。
- 实例变量：属于实例，定义在 `__init__`/`self.x`。

可变对象做成类变量是经典事故源：

```python
class BadSession:
    messages = []          # 类变量，共享！

    def add(self, msg):
        self.messages.append(msg)

a, b = BadSession(), BadSession()
a.add("userA")
b.add("userB")
print(a.messages)  # ['userA', 'userB'] 串了
```

正确：

```python
class Session:
    def __init__(self):
        self.messages = []
```

会话、Agent 记忆、请求上下文必须放实例变量或外部存储，不能误用类变量。


<h2 id="46.python中__new__和__init__有什么区别">46.Python中__new__和__init__有什么区别？</h2>

- `__new__`：创建并返回实例（静态方法语义）。
- `__init__`：初始化已创建的实例，不能返回非 None。

```python
class ModelClient:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, endpoint):
        self.endpoint = endpoint  # 每次构造仍会跑
```

注意：单例 + `__init__` 每次都会执行，需用 `_inited` 防护重复初始化。业务代码更推荐模块级单例/依赖注入。


<h2 id="47.python中文件读取readreadlinereadlines和迭代文件对象有什么区别">47.Python中文件读取read、readline、readlines和迭代文件对象有什么区别？</h2>

| 方法 | 行为 | 内存 | 场景 |
|------|------|------|------|
| `read()` | 全读 | 高 | 小配置文件 |
| `readline()` | 读一行 | 低 | 自控流程 |
| `readlines()` | 全行列表 | 高 | 小文本 |
| `for line in f` | 惰性逐行 | 低 | 日志/JSONL/大语料 |

```python
from pathlib import Path

with Path("traces.jsonl").open(encoding="utf-8") as f:
    for line in f:
        if "error" in line:
            print(line.strip())
```

RAG 切分、训练数据扫描、Agent trace 分析一律优先流式，避免 OOM。


<h2 id="48.python中的类型注解dataclass和pydantic在ai工程中有什么价值">48.Python中的类型注解、dataclass和Pydantic在AI工程中有什么价值？</h2>

类型注解不改变动态语言本质，但能提升可维护性、IDE、静态检查和接口可靠性。

```python
from dataclasses import dataclass

@dataclass
class GenerationConfig:
    prompt: str
    steps: int = 30
    guidance_scale: float = 7.5
```

```python
from pydantic import BaseModel, Field

class ToolArgs(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
```

AI 工程价值：

1. 工具参数校验（LLM 会传错字段）。
2. 结构化输出 schema 与重试。
3. 多模型服务的稳定请求/响应协议。
4. RAG/审核/日志链路的数据契约。

金句：Python 可以动态，AI 工程不能随意；类型边界把不确定的 LLM 输出接入确定性系统。


<h2 id="49.python-313314之后giljitt-string延迟注解有哪些新变化">49.Python 3.13/3.14之后，GIL、JIT、t-string、延迟注解有哪些新变化？</h2>

截至 2026 年，3.14 已是稳定大版本。对 AI 工程有实际影响的变化：

1. **Free-threaded Python**：3.13 起可选关闭 GIL，3.14 继续推进；生态与 C 扩展兼容性仍需评估。
2. **实验性 JIT**：持续改进解释器性能，但不替代 CUDA/C++。
3. **t-string（3.14）**：模板字符串，返回可处理模板对象，适合安全 SQL/HTML/日志模板/DSL。
4. **延迟注解与多解释器**：类型注解求值更友好；标准库出现 `InterpreterPoolExecutor` 等方向，为 CPU 并行提供新选项。

误区：

- free-threaded ≠ 所有项目应立刻关 GIL。
- JIT ≠ Python 取代 GPU 内核；瓶颈常在矩阵计算与 IO 架构。


<h2 id="50.aigc和ai-agent项目中python基础能力应该重点掌握哪些">50.AIGC和AI Agent项目中，Python基础能力应该重点掌握哪些？</h2>

AIGC/Agent 项目考的不是“会不会语法”，而是能不能把 Python 写成可靠工程。

高频清单：

- **数据结构**：list、dict、set、tuple、deque、heapq、Counter
- **对象模型**：传参、作用域、闭包、装饰器、类方法、魔术方法
- **IO 序列化**：JSON/JSONL、pathlib、流式读写
- **并发**：threading、multiprocessing、asyncio、池
- **类型校验**：typing、dataclass、Pydantic
- **服务化**：FastAPI、校验、异常、日志、健康检查
- **数据处理**：NumPy、Pillow、OpenCV、Torch Tensor
- **安全**：禁滥用 eval、不拼接 SQL、密钥进环境变量
- **可观测**：logging、request_id、token 用量、延迟、工具调用日志
- **工程化**：venv/uv、pyproject、依赖锁定、可复现部署

面试总结句：<font color=DeepSkyBlue>基础扎实 = 数据结构选对 + 并发模型选对 + 边界校验到位 + 可观测可复现。</font>
