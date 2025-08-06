在 PyCaret 中，**create\_api** 函数用于为训练好的模型生成一个 REST API 端点，基于 FastAPI 框架。然而，PyCaret 的 **create\_api** 函数一次只能为单个模型生成一个 API 文件。

如果你想要创建多个接口（例如，为多个模型生成多个 API 端点），需要为每个模型分别调用 **create\_api**，并手动管理生成的 API 文件和端点，以避免冲突。

以下是具体步骤和注意事项的详细说明：

**实现步骤**

1. **训练多个模型**：

使用 PyCaret 的 **create\_model** 函数训练多个模型。例如，假设你想为逻辑回归（Logistic Regression）和随机森林（Random Forest）模型分别创建 API。
**python**

```python
from pycaret.classification import *
from pycaret.datasets import get_data

# 加载数据集
data = get_data('iris')

# 初始化 PyCaret 环境
exp = setup(data, target='species', session_id=123)

# 训练多个模型
lr_model = create_model('lr')  # 逻辑回归
rf_model = create_model('rf')  # 随机森林
```

**为每个模型创建 API**： 使用 **create\_api** 为每个模型生成单独的 API 文件。需要为每个 API 指定唯一的文件名，以避免覆盖。
**python**

```python
# 为逻辑回归模型创建 API
create_api(lr_model, 'lr_api')

# 为随机森林模型创建 API
create_api(rf_model, 'rf_api')
```

上述代码会在当前工作目录下生成两个 Python 文件：**lr\_api.py** 和 **rf\_api.py**，每个文件包含一个独立的 FastAPI 应用程序。

**运行多个 API**： 默认情况下，**create\_api** 生成的 API 使用 FastAPI，并且需要手动运行生成的 Python 文件。你可以通过以下方式运行每个 API，但需要为每个 API 指定不同的端口以避免冲突。
**bash**

```bash
# 运行逻辑回归模型的 API（默认端口 8000）
python lr_api.py
```

**bash**

```bash
# 运行随机森林模型的 API（指定不同端口，例如 8001）
uvicorn rf_api:app --port 8001
```

**注意**：**create\_api** 默认生成的 API 文件使用端口 8000。如果要同时运行多个 API，必须修改端口号。可以通过命令行参数（如上所示）或编辑生成的 API 文件来更改端口。**

**修改 API 文件以支持多个端点（可选）**： 如果你希望在单个 FastAPI 应用程序中为多个模型提供不同的端点，可以手动合并生成的 API 文件。例如，修改一个 API 文件，将多个模型的预测逻辑整合到一个 FastAPI 应用中。**以下是一个合并后的示例代码（假设你手动编辑 **combined\_api.py**）：**
**python**

```python
from fastapi import FastAPI
from pycaret.classification import load_model, predict_model
import pandas as pd

app = FastAPI()

# 加载多个模型
lr_model = load_model('lr_model')  # 假设模型已保存
rf_model = load_model('rf_model')  # 假设模型已保存

@app.post("/predict_lr")
async def predict_lr(data: dict):
    data = pd.DataFrame([data])
    predictions = predict_model(lr_model, data=data)
    return predictions.to_dict()

@app.post("/predict_rf")
async def predict_rf(data: dict):
    data = pd.DataFrame([data])
    predictions = predict_model(rf_model, data=data)
    return predictions.to_dict()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

在这个例子中：

为逻辑回归模型创建了 **/predict\_lr** 端点。为随机森林模型创建了 **/predict\_rf** 端点。两个端点共享同一个 FastAPI 应用，运行在同一端口（例如 8000）。

运行合并后的 API：

```bash
python combined_api.py
```

然后，你可以通过以下 URL 访问不同的端点：

* 逻辑回归模型：http://localhost:8000/predict\_lr
* 随机森林模型：http://localhost:8000/predict\_rf

**测试 API**： 使用工具如 **curl**、Postman 或 Python 的 **requests** 库测试每个端点。例如：
**python**

```python
import requests

# 测试逻辑回归模型的 API
url_lr = "http://localhost:8000/predict_lr"
data = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
response_lr = requests.post(url_lr, json=data)
print(response_lr.json())

# 测试随机森林模型的 API
url_rf = "http://localhost:8000/predict_rf"
response_rf = requests.post(url_rf, json=data)
print(response_rf.json())
```

**部署和扩展（可选）**：

* **Docker 支持**：PyCaret 提供了 **create\_docker** 函数，可以为每个 API 生成 Dockerfile 和 **requirements.txt** 文件，方便部署。
  **python**

  ```python
  create_docker('lr_api')
  create_docker('rf_api')
  ```

  这会为每个 API 生成相应的 Docker 文件，分别位于 **lr\_api** 和 **rf\_api** 文件夹中。
* **云部署**：你可以将 API 部署到云平台（如 AWS、GCP 或 Azure）。PyCaret 的 **deploy\_model** 函数支持直接部署到云端，但需要配置相应的认证信息。例如：

  ```python
  deploy_model(lr_model, model_name='lr_aws', platform='aws', authentication={'bucket': 'pycaret-test'})
  ```

  如果需要多个模型的 API，可以为每个模型分别调用 **deploy\_model**。

注意事项

 **端口冲突**：如果在同一台机器上运行多个 API，必须为每个 API 指定不同的端口号。可以通过命令行参数（**uvicorn --port**）或修改 API 文件中的端口设置来实现。

* **文件管理**：**create\_api** 为每个模型生成一个独立的 Python 文件。确保文件名唯一，否则后生成的 API 文件会覆盖之前的文件。
* **依赖管理**：确保安装了 FastAPI 和 Uvicorn（PyCaret 的 **mlops** 额外依赖包含这些库）。安装命令：
  **bash**

  ```bash
  pip install pycaret[mlops]
  ```
* **API 数据格式**：生成的 API 期望输入数据为 JSON 格式，字段名应与训练数据中的特征名一致。返回的预测结果通常包括预测标签和概率（对于分类任务）。
* **局限性**：PyCaret 的 **create\_api** 函数本身不支持直接在一个文件中生成多个端点。如果需要更复杂的 API 结构（如多个模型共享一个端点），需要手动编写 FastAPI 代码（如步骤 4 所示）。
* **已知问题**：根据 GitHub 上的问题（），**create\_api** 在某些版本中可能存在数据模型注解问题（例如 Pydantic 版本冲突）。确保使用最新版本的 PyCaret（例如 3.3.2）并检查依赖兼容性：

  ```bash
  pip install pycaret==3.3.2
  ```

**总结**

PyCaret 的 **create\_api** 函数一次只能为一个模型生成一个 API 文件。要创建多个接口，你可以：

1. 为每个模型调用 **create\_api**，生成独立的 API 文件。
2. 运行每个 API 时指定不同端口，或手动合并 API 文件以在一个 FastAPI 应用中提供多个端点。
3. 使用 **create\_docker** 或 **deploy\_model** 进一步简化部署。
