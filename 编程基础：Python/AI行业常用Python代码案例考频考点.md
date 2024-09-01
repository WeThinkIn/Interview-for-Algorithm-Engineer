# 目录
- [1.多进程multiprocessing基本使用代码段](#1.多进程multiprocessing基本使用代码段)
- [2.指定脚本所使用的GPU设备](#2.指定脚本所使用的GPU设备)
- [3.介绍一下如何使用Python中的flask库搭建AI服务](#3.介绍一下如何使用Python中的flask库搭建AI服务)
- [4.介绍一下如何使用Python中的fastapi构建AI服务](#4.介绍一下如何使用Python中的fastapi构建AI服务)
- [5.python如何清理AI模型的显存占用?](#5.python如何清理AI模型的显存占用?)
- [6.python中对透明图的处理大全](#6.python中对透明图的处理大全)

<h2 id='1.多进程multiprocessing基本使用代码段'>1.多进程multiprocessing基本使用代码段</h2>


```python
# 基本代码段
from multiprocessing import Process

def runner(pool_id):
    print(f'WeThinkIn {pool_id}')

if __name__ == '__main__':
    process_list = []
    pool_size = 10
    for pool_id in range(pool_size):

        # 实例化进程对象
        p = Process(target=runner, args=(pool_id,))

        # 启动进程
        p.start()

        process_list.append(p)

    # 等待全部进程执行完毕
    for i in process_list:
        p.join()
```

注意：进程是python的最小资源分配单元，每个进程会独立进行内存分配和数据拷贝。


```python
# 进程间通信
# 另外Pipe也可以实现类似的通信功能。
import time
from multiprocessing import Process, Queue, set_start_method

def runner(pool_queue, pool_id):
    if not pool_queue.empty():
        print(f"WeThinkIn {pool_id}: read {pool_queue.get()}")
    pool_queue.put(f"the queue message from WeThinkIn {pool_id}")

if __name__ == "__main__":
    
    # mac默认启动进程的方式是fork
    set_start_method("fork")
    
    queue = Queue()
    
    process_list = []
    pool_size = 10
    for pool_id in range(pool_size):
        
        # 实例化进程对象
        p = Process(target=runner, args=(queue, pool_id,))
        
        # 启动进程
        p.start()
        time.sleep(1)
        process_list.append(p)
    
    # 等待全部进程执行完毕
    for i in process_list:
        p.join()
        
    print("############")
    while not queue.empty():
        print(f"Remain {queue.get()}")
```

```python
# 进程间维护全局数据
import time
from multiprocessing import Process, Queue, set_start_method

def runner(global_dict, pool_id):
    temp = "WeThinkIn!"
    global_dict[temp[pool_id]] = pool_id

if __name__ == "__main__":
    
    # mac默认启动进程的方式是fork
    set_start_method("fork")
    
    # queue = Queue()
    manager = Manager()
    global_dict = manager.dict()
    # global_list = manager.list()
    
    process_list = []
    pool_size = 10
    for pool_id in range(pool_size):
        
        # 实例化进程对象
        p = Process(target=runner, args=(global_dict, pool_id,))
        
        # 启动进程
        p.start()
        time.sleep(1)
        process_list.append(p)
    
    # 等待全部进程执行完毕
    for i in process_list:
        p.join()
        
    print("############")
    print(global_dict)
```

<h2 id='2.指定脚本所使用的GPU设备'>2.指定脚本所使用的GPU设备</h2>

1.命令行临时指定
```sh
export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0,1,2,3
```
若希望更新入环境变量：
```sh
. ~/.bashrc
```

2.执行脚本前指定
```sh
CUDA_VISIBLE_DEVICES=0 python WeThinkIn.py
```

3.python脚本中指定
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
```


<h2 id="3.介绍一下如何使用Python中的flask库搭建AI服务">3.介绍一下如何使用Python中的flask库搭建AI服务</h2>

搭建一个简单的AI服务，我们可以使用 `Flask` 作为 Web 框架，并结合一些常用的Python库来实现AI模型的加载、推理等功能。这个服务将能够接收来自客户端的请求，运行AI模型进行推理，并返回预测结果。

下面是一个完整的架构和详细步骤，可以帮助我们搭建一个简单明了的AI服务。

### 1. AI服务结构

首先，我们需要定义一下项AI服务的文件结构：

```
ai_service/
│
├── app.py                 # 主 Flask 应用
├── model.py               # AI 模型相关代码
├── requirements.txt       # 项目依赖
└── templates/
    └── index.html         # 前端模板（可选）
```

### 2. 编写模型代码 (model.py)

在 `model.py` 中，我们定义 AI 模型的加载和预测功能。假设我们有一个训练好的 `PyTorch` 模型来识别手写数字（例如使用 MNIST 数据集训练的模型）。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class AIModel:
    def __init__(self, model_path):
        self.model = SimpleCNN()
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()

    def predict(self, image_array):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        image_tensor = transform(image_array).unsqueeze(0)  # 添加批次维度
        with torch.no_grad():
            output = self.model(image_tensor)
            prediction = output.argmax(dim=1, keepdim=True)
        return prediction.item()
```

### 3. 编写 Flask 应用 (app.py)

在 `app.py` 中，我们使用 `Flask` 创建一个简单的 Web 应用，可以处理图像上传和模型推理请求。

```python
from flask import Flask, request, jsonify, render_template
from model import AIModel
from PIL import Image
import io

# 创建一个AI服务的APP对象
app = Flask(__name__)

# 实例化模型，假设模型保存为 'model.pth'
model = AIModel('model.pth')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 检查是否有文件上传
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # 将图像转换为PIL格式
        img = Image.open(file).convert('L')  # 假设灰度图像
        img = img.resize((28, 28))  # 调整到模型输入尺寸

        # 调用模型进行预测
        prediction = model.predict(img)

        # 返回预测结果
        return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
```

### 4. 运行服务

在命令行中运行以下命令启动 Flask 应用：

```bash
python app.py
```

默认情况下，Flask 应用将运行在 `http://127.0.0.1:5000/`。我们可以打开浏览器访问这个地址并上传图像进行测试。

### 5. 完整流程讲解

- **前端 (index.html)**：用户通过浏览器上传图像文件。
- **Flask 路由 (`/predict`)**：接收上传的图像，并将其传递给 AI 模型进行预测。
- **AI 模型 (`model.py`)**：加载预训练的模型，处理图像并返回预测结果。
- **响应返回**：Flask 将预测结果以 JSON 格式返回给客户端，用户可以看到预测的类别或其他结果。

### 6. 细节关键点讲解

上面代码中的`@app.route('/predict', methods=['POST'])` 是 Flask 中的路由装饰器，用于定义 URL 路由和视图函数。它们决定了用户访问特定 URL 时，Flask 应用程序如何响应。

#### **`@app.route('/predict', methods=['POST'])`** 的作用

- **`@app.route('/predict', methods=['POST'])`** 的含义：
  - 这是一个路由装饰器，Flask 使用它来将 `/predict` 路由映射到一个视图函数。
  - `'/predict'` 表示路径 `/predict`，即当用户访问 `http://127.0.0.1:5000/predict` 时，这个路由会被触发。
  - `methods=['POST']` 指定了这个路由只接受 `POST` 请求。`POST` 请求通常用于向服务器发送数据，例如表单提交、文件上传等。与之对应的 `GET` 请求则用于从服务器获取数据。

#### 作用：

- 当客户端（通常是浏览器或其他应用程序）发送一个 `POST` 请求到 `http://127.0.0.1:5000/predict`，并附带一个文件时，Flask 会调用 `predict()` 函数来处理这个请求。
- `predict()` 函数接收上传的图像文件，对其进行预处理，然后将图像传递给预训练的 AI 模型进行预测。
- 预测结果以 JSON 格式返回给客户端，客户端可以使用这些数据来进行后续操作，如显示预测结果等。



<h2 id="4.介绍一下如何使用Python中的fastapi构建AI服务">4.介绍一下如何使用Python中的fastapi构建AI服务</h2>

使用 FastAPI 构建一个 AI 服务是一个非常强大和灵活的解决方案。FastAPI 是一个快速的、基于 Python 的 Web 框架，特别适合构建 API 和处理异步请求。它具有类型提示、自动生成文档等特性，非常适合用于构建 AI 服务。下面是一个详细的步骤指南，我们可以从零开始构建一个简单的 AI 服务。

### 1. **构建基本的 FastAPI 应用**

首先，我们创建一个 Python 文件（如 `main.py`），在其中定义基本的 FastAPI 应用。

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI service!"}
```

这段代码创建了一个基本的 FastAPI 应用，并定义了一个简单的根路径 `/`，返回一个欢迎消息。

### 2. **引入 AI 模型**

接下来，我们将引入一个简单的 AI 模型，比如一个预训练的文本分类模型。假设我们使用 Hugging Face 的 Transformers 库来加载模型。

在我们的 `main.py` 中加载这个模型：

```python
from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

# 加载预训练的模型（例如用于情感分析）
classifier = pipeline("sentiment-analysis")

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI service!"}

@app.post("/predict/")
def predict(text: str):
    result = classifier(text)
    return {"prediction": result}
```

在这个例子中，我们加载了一个用于情感分析的预训练模型，并定义了一个 POST 请求的端点 `/predict/`。用户可以向该端点发送文本数据，服务会返回模型的预测结果。

### 3. **测试我们的 API**

使用 Uvicorn 运行我们的 FastAPI 应用：

```bash
uvicorn main:app --reload
```

- `main:app` 指定了应用所在的模块（即 `main.py` 中的 `app` 对象）。
- `--reload` 使服务器在代码更改时自动重新加载，适合开发环境使用。

启动服务器后，我们可以在浏览器中访问 `http://127.0.0.1:8000/` 查看欢迎消息，还可以向 `http://127.0.0.1:8000/docs` 访问自动生成的 API 文档。

### 4. **通过 curl 或 Postman 测试我们的 AI 服务**

我们可以使用 `curl` 或 Postman 发送请求来测试 AI 服务。

使用 `curl` 示例：

```bash
curl -X POST "http://127.0.0.1:8000/predict/" -H "Content-Type: application/json" -d "{\"text\":\"I love WeThinkIn!\"}"
```

我们会收到类似于以下的响应：

```json
{
  "prediction": [
    {
      "label": "POSITIVE",
      "score": 0.9998788237571716
    }
  ]
}
```

### 5. **添加请求数据验证**

为了确保输入的数据是有效的，我们可以使用 FastAPI 的 Pydantic 模型来进行数据验证。Pydantic 允许我们定义请求体的结构，并自动进行验证。

```python
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

classifier = pipeline("sentiment-analysis")

class TextInput(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI service!"}

@app.post("/predict/")
def predict(input: TextInput):
    result = classifier(input.text)
    return {"prediction": result}
```

现在，POST 请求 `/predict/` 需要接收一个 JSON 对象，格式为：

```json
{
  "text": "I love WeThinkIn"
}
```

如果输入数据不符合要求，FastAPI 会自动返回错误信息。

### 6. **异步处理（可选）**

FastAPI 支持异步处理，这在处理 I/O 密集型任务时非常有用。假如我们的 AI 模型需要异步调用，我们可以使用 `async` 和 `await` 关键字：

```python
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

classifier = pipeline("sentiment-analysis")

class TextInput(BaseModel):
    text: str

@app.get("/")
async def read_root():
    return {"message": "Welcome to the AI service!"}

@app.post("/predict/")
async def predict(input: TextInput):
    result = await classifier(input.text)
    return {"prediction": result}
```

在这个例子中，我们假设 `classifier` 可以使用 `await` 异步调用。

### 7. **部署我们的 FastAPI 应用**

开发完成后，我们可以将应用部署到生产环境。常见的部署方法包括：

- 使用 Uvicorn + Gunicorn 进行生产级部署：
  
  ```bash
  gunicorn -k uvicorn.workers.UvicornWorker main:app
  ```

- 部署到云平台，如 AWS、GCP、Azure 等。

- 使用 Docker 构建容器化应用，便于跨平台部署。



<h2 id="6.python如何清理AI模型的显存占用?">6.python如何清理AI模型的显存占用?</h2>

在AIGC、传统深度学习、自动驾驶领域，在AI项目服务的运行过程中，当我们不再需要使用AI模型时，可以通过以下两个方式来释放该模型占用的显存：

1. 删除AI模型对象、清除缓存，以及调用垃圾回收（Garbage Collection）来确保显存被释放。
2. 将AI模型对象从GPU迁移到CPU中进行缓存。

### 1. 第一种方式（删除清理）

```python
import torch
import gc

# 定义一个简单的模型
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = torch.nn.Linear(10, 10)
        self.fc2 = torch.nn.Linear(10, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 创建模型并将其移动到 GPU
model = SimpleModel().cuda()

# 模拟训练或推理
dummy_input = torch.randn(1, 10).cuda()
output = model(dummy_input)

# 删除模型
del model

# 清除缓存
# 使用 `torch.cuda.empty_cache()` 来清除未使用的显存缓存。这不会释放显存，但会将未使用的缓存显存返回给 GPU，以便其他 CUDA 应用程序可以使用。
torch.cuda.empty_cache()

# 调用垃圾回收
# 使用 Python 的 `gc` 模块显式调用垃圾回收器，以确保删除模型对象后未引用的显存能够被释放：
gc.collect()

# 额外说明
# `torch.cuda.empty_cache()`: 这个函数会释放 GPU 中缓存的内存，但不会影响已经分配的内存。它将缓存的内存返回给 GPU 以供其他 CUDA 应用程序使用。
# `gc.collect()`: Python 的垃圾回收器会释放所有未引用的对象，包括 GPU 内存。如果删除对象后显存没有立即被释放，调用 `gc.collect()` 可以帮助确保显存被释放。

# 检查显存使用情况
print(torch.cuda.memory_allocated())
print(torch.cuda.memory_reserved())
```

### 1. 第二种方式（迁移清理）

```python
import torch
import gc

# 定义一个简单的模型
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = torch.nn.Linear(10, 10)
        self.fc2 = torch.nn.Linear(10, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 创建模型并将其移动到 GPU
model = SimpleModel().cuda()

# 模拟训练或推理
dummy_input = torch.randn(1, 10).cuda()
output = model(dummy_input)

# 迁移模型
model.cpu()

# 清除缓存
# 使用 `torch.cuda.empty_cache()` 来清除未使用的显存缓存。这不会释放显存，但会将未使用的缓存显存返回给 GPU，以便其他 CUDA 应用程序可以使用。
torch.cuda.empty_cache()

# 调用垃圾回收
# 使用 Python 的 `gc` 模块显式调用垃圾回收器，以确保删除模型对象后未引用的显存能够被释放：
gc.collect()

# 额外说明
# `torch.cuda.empty_cache()`: 这个函数会释放 GPU 中缓存的内存，但不会影响已经分配的内存。它将缓存的内存返回给 GPU 以供其他 CUDA 应用程序使用。
# `gc.collect()`: Python 的垃圾回收器会释放所有未引用的对象，包括 GPU 内存。如果删除对象后显存没有立即被释放，调用 `gc.collect()` 可以帮助确保显存被释放。

# 检查显存使用情况
print(torch.cuda.memory_allocated())
print(torch.cuda.memory_reserved())
```


<h2 id="49.python中对透明图的处理大全">49.python中对透明图的处理大全</h2>

### 判断输入图像是不是透明图

要判断一个图像是否具有透明度（即是否是透明图像），我们可以检查图像是否包含 **Alpha 通道**。Alpha通道是用来表示图像中每个像素的透明度的通道。如果图像有 Alpha 通道，则它可能是透明图像。我们可以用下面的Python代码来判断图像是否是透明图：

```python
from PIL import Image

def is_transparent_image(image_path):
    # 打开图像
    img = Image.open(image_path)

    # 检查图像模式是否包含Alpha通道。`RGBA` 和 `LA` 模式包含 Alpha 通道，`P` 模式可能包含透明度信息（通过 `img.info` 中的 `transparency` 属性）。
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        # 如果图像有alpha通道，逐个像素检查是否存在透明部分
        alpha = img.split()[-1]  # 获取alpha通道
        # 如果图像中任何一个像素的alpha值小于255，则图像是透明的
        if alpha.getextrema()[0] < 255:
            return True

    # 如果图像没有Alpha通道或者所有像素都是不透明的
    return False

# 示例路径，替换为我们的图像路径
image_path = "/本地路径/example.png"
if is_transparent_image(image_path):
    print("这是一个透明图像。")
else:
    print("这是一个不透明图像。")
```

### 判断输入图像是否是透明图，将透明图的透明通道提取，剩余部分作为常规图像进行处理

要判断输入图像是否是透明图，并且将透明部分分离，保留剩余部分用于后续处理，我们可以使用下面的Python代码完成这项任务：

```python
from PIL import Image

def process_image(image_path, output_path):
    # 打开图像
    img = Image.open(image_path)

    # 检查图像模式是否包含Alpha通道
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        # 如果图像有alpha通道，逐个像素检查是否存在透明部分
        alpha = img.split()[-1]  # 获取alpha通道
        # 如果图像中任何一个像素的alpha值小于255，则图像是透明的
        if alpha.getextrema()[0] < 255:
            print("这是一个透明图像。")
        else:
            print("这是一个不透明图像。")
            img.save(output_path)
            return img
        
        # 将图像的透明部分分离出来
        # 分离alpha通道。如果图像是透明图像，将其拆分为红、绿、蓝和 Alpha 通道（透明度）。
        r, g, b, alpha = img.split() if img.mode == 'RGBA' else (img.convert('RGBA').split())

        # 创建一个完全不透明的背景图像
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))

        # 将原图像的透明部分分离
        img_no_alpha = Image.composite(img, bg, alpha)

        # 将结果保存或用于后续处理
        img_no_alpha.save(output_path)
        print(f"透明部分已分离，图像已保存为: {output_path}")
        
        return img_no_alpha  # 返回没有透明度的图像以便后续处理
    else:
        print("这不是一个透明图像，直接进行后续处理。")
        # 直接进行后续处理
        img.save(output_path)
        return img

# 示例路径
image_path = "/本地路径/example.png"
output_path = "/本地路径/processed_image.png"

# 处理图像
processed_image = process_image(image_path, output_path)
```

### 将常规图像转换成透明图

要将一张普通图片转换成带有透明背景的图片，下面是实现这个功能的代码示例：

```python
from PIL import Image

def convert_to_transparent(image_path, output_path, color_to_transparent):
    # 打开图像
    img = Image.open(image_path)
    
    # 确保图像有alpha通道
    img = img.convert("RGBA")
    
    # 获取图像的像素数据
    datas = img.getdata()

    # 创建新的像素数据列表
    new_data = []
    for item in datas:
        # 检查像素是否与指定的颜色匹配
        if item[:3] == color_to_transparent:
            # 将颜色变为透明
            new_data.append((255, 255, 255, 0))
        else:
            # 保留原来的颜色
            new_data.append(item)

    # 更新图像数据
    img.putdata(new_data)
    
    # 保存带透明背景的图像
    img.save(output_path, "PNG")
    print(f"图像已成功转换为透明背景，并保存为: {output_path}")

# 示例路径，替换为你的图像路径和颜色
image_path = "/本地路径/example.jpg"
output_path = "/本地路径/transparent_image.png"
color_to_transparent = (255, 255, 255)  # 白色背景

# 将图片转换成透明背景
convert_to_transparent(image_path, output_path, color_to_transparent)
```

### 读取透明图，不丢失透明通道信息

在 Python 中使用 `OpenCV` 或 `Pillow` 读取图像时，可以确保不丢失图像的透明通道。以下是如何使用这两个库读取带有透明通道的图像的代码示例。

#### 使用 OpenCV 读取带透明通道的图像

默认情况下，`OpenCV` 读取图像时可能会丢失透明通道（即 Alpha 通道）。为了确保保留透明通道，我们需要使用 `cv2.IMREAD_UNCHANGED` 标志来读取图像。

```python
import cv2

# 读取带透明通道的图像
image_path = "/本地路径/example.png"
img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

# 检查图像通道数，确保Alpha通道存在
if img.shape[2] == 4:
    print("图像成功读取，并且包含透明通道（Alpha）。")
else:
    print("图像成功读取，但不包含透明通道（Alpha）。")
```

#### 使用 Pillow 读取带透明通道的图像

在Python中使用`Pillow` 库在读取图像时默认保留透明通道，因此我们可以直接使用 `Image.open()` 读取图像并保留 Alpha 通道：

```python
from PIL import Image

# 读取带透明通道的图像
image_path = "/本地路径/example.png"
img = Image.open(image_path)

# 确保图像是 RGBA 模式（包含透明通道）
if img.mode == "RGBA":
    print("图像成功读取，并且包含透明通道（Alpha）。")
else:
    print("图像成功读取，但不包含透明通道（Alpha）。")
```

### PIL格式图像与OpenCV格式图像互相转换时，保留透明通道

要将 `Pillow`（PIL）格式的图像与 `OpenCV` 格式的图像互相转换，并且保留透明通道（即 Alpha 通道），我们可以按照以下步骤操作：

#### 1. 从 `Pillow` 转换为 `OpenCV`

```python
from PIL import Image
import numpy as np
import cv2

# 打开一个Pillow图像对象，并确保图像是RGBA模式
pil_image = Image.open('input.png').convert('RGBA')

# 将Pillow图像转换为NumPy数组
opencv_image = np.array(pil_image)

# 将图像从RGBA格式转换为OpenCV的BGRA格式
opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGBA2BGRA)

# 现在，opencv_image是一个保留透明通道的OpenCV图像，可以使用cv2.imshow显示或cv2.imwrite保存
cv2.imwrite('output_opencv.png', opencv_image)
```

#### 2. 从 `OpenCV` 转换为 `Pillow`

```python
import cv2
from PIL import Image

# 读取一个OpenCV图像，确保读取时保留Alpha通道
opencv_image = cv2.imread('input.png', cv2.IMREAD_UNCHANGED)

# 将图像从BGRA格式转换为RGBA格式。使用 `cv2.cvtColor` 将图像从 `BGRA` 格式转换为 `RGBA` 格式，因为 `Pillow` 使用的是 `RGBA` 格式。
opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGRA2RGBA)

# 将OpenCV图像转换为Pillow图像
pil_image = Image.fromarray(opencv_image)

# 现在，pil_image是一个保留透明通道的Pillow图像，可以使用pil_image.show()显示或pil_image.save保存
pil_image.save('output_pillow.png')
```
