## 1, 搭建虚拟环境

* 1.1新建虚拟环境的目录（也可手动新建目录，只要是程序主目录的子目录就可以）   python -m venv demo1_env
* 1.2，启动虚拟环境命令       .\demo1_env\Scripts\activate
  启动后poeershell命令行前会带有 目录名！
* 1.3、退出虚拟环境   deactivate
* 1.4、安装虚拟环境的命令  pip install virtualenv

## 2，flask Local Testing in colab

* 2.1,安装专门的pyngrok   （ 注意：flask-ngrok不管用！）
  !pip install pyngrok
* 2.2，文件引入和token配置

```python
from flask import Flask,render_template
from pyngrok import ngrok
```

* 2.3 指定token

```python
ngrok_token = '30i5zXWIPnjgCzssvKL6iadM5PQ_x2agd67MbEGTsRBCmC2t'
ngrok.set_auth_token(ngrok_token)
```

正常编写代码：

```python
app = Flask(__name__)
port = 5000        #指定运行端口

@app.route("/")
def root():
return "Hello from Flask in Colab!"
```

* 2.4，启动

```python
if __name__ == "__main__":
try:
public_url = ngrok.connect(port).public_url
print(f"Flask app running at: {public_url}")
app.run(port=port)
finally:
ngrok.disconnect(public_url=public_url)
```
