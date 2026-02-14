# 使用说明

## 运行方法

1. 使用镜像创建容器，然后使用 SSH 终端登录容器；

2. 尝试运行命令：

    ```sh
    /opt/fusang/fusang.py
    ```

3. 运行测试用例：

    ```sh
    bash /opt/fusang/ghfund/run-all.sh
    ```

4. 运行单个测试用例（可选）：

    ```sh
    bash /opt/fusang/ghfund/demo-1.sh
    ```

    ```sh
    bash /opt/fusang/ghfund/demo-2.sh
    ```

    ```sh
    bash /opt/fusang/ghfund/demo-3.sh
    ```

## 镜像构建方法

1. 使用基础镜像 `TensorFlow / 2.18.0 / py3.10-ubuntu22.04 / dtk25.04.2` 创建 SSH 终端类型容器（选择 `kshdexclu09` 队列，使用“异构加速卡1/16GB”）；

2. 使用文件管理工具，将测试数据上传到 `/opt/fusang-data/`；

3. 打开命令行终端（E-Shell），运行如下命令：

    ```sh
    apt update
    apt install mafft

    cd /path/to/work/
    git clone https://github.com/yanlinlin82/fusang

    pip install --upgrade pip
    pip install -r fusang/ghfund/requirements.txt
    ```

4. 测试：

    ```sh
    /opt/fusang/fusang.py
    ```

    若运行成功，则可继续保存容器到镜像。
