# MinIO

## 命令行
1. 安装
   ```bash
    curl https://dl.min.io/client/mc/release/linux-amd64/mc --create-dirs  -o $HOME/minio-binaries/mc
    chmod +x $HOME/minio-binaries/mc
    # set env
    export PATH=$PATH:$HOME/minio-binaries/
   ```
2. 添加`modelzoo`存储桶
   ```bash
    bash +o history
    mc alias set modelzoo http://10.23.4.220:9011 modelzoo modelzoo666
    bash -o history

    # 移除
    mc config host remove modelzoo

    # 测试连接
    mc admin info modelzoo
   ```
3. 获取权重三件套

   ```bash
    mc ls modelzoo/vacc-weights

    mc cp -r modelzoo/vacc-weights/**  ./
   ```

## 浏览器

1. 访问`10.23.4.220:9010`
2. 输入用户名&密码
   - user: modelzoo
   - passwd：modelzoo666
