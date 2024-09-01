# 目录
- [1.多进程multiprocessing基本使用代码段](#1.多进程multiprocessing基本使用代码段)
- [2.指定脚本所使用的GPU设备](#2.指定脚本所使用的GPU设备)

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
