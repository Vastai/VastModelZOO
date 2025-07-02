# DS3 Docker Compose 

在启动 容器 前，请先确定已经使用 docker-compose down 命令停止了已经启动的容器   

## 测试 DeepSeek-V3
1. 更改 PATH 值    
- 更改 当前目录下的 .env 文件里的 DS_V3_MODEL_LOCAL_PATH 值，指定为host上 DeepSeek-V3 模型全路径

2. 启动 docker container
```bash 
sudo docker-compose -f docker-compose/ds-v3-docker-compose.yaml up -d
```

3. 查看 container logs 
```bash
sudo docker logs -f vllm_deepseek-v3
```

4. 停止 docker container
```bash
sudo docker-compose -f docker-compose/ds-v3-docker-compose.yaml down
```


## 测试 DeepSeek-V3-0324 
1. 更改 PATH 值    
- 更改 当前目录下的 .env 文件里的 DS_V3_0324_MODEL_LOCAL_PATH 值，指定为host上 DeepSeek-V3-0324 模型全路径

2. 启动 docker container
```bash 
sudo docker-compose -f docker-compose/ds-v3-0324-docker-compose.yaml up -d
```

3. 查看 container logs 
```bash
sudo docker logs -f vllm_deepseek-v3-0324
```

4. 停止 docker container
```bash
sudo docker-compose -f docker-compose/ds-v3-0324-docker-compose.yaml down
```


## 测试 DeepSeek-R1
1. 更改 PATH 值    
- 更改 当前目录下的 .env 文件里的 DS_R1_MODEL_LOCAL_PATH 值，指定为host上 DeepSeek-R1 模型全路径

2. 启动 docker container
```bash 
sudo docker-compose -f docker-compose/ds-r1-docker-compose.yaml up -d
```

3. 查看 container logs 
```bash
sudo docker logs -f vllm_deepseek-r1
```

4. 停止 docker container
```bash
sudo docker-compose -f docker-compose/ds-r1-docker-compose.yaml down
```


## 快速测试容器

### 测试  vllm_deepseek-v3 
```bash
# 注意修改 ip 地址
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer token-abc123" \
-d '{
  "model": "DeepSeek-V3",
  "messages": [
      {"role": "system", "content": "你是一个专业助手"},
      {"role": "user", "content": "请介绍四大文明古国"}
 ],
  "max_tokens": 100
}'

```


### 测试  vllm_deepseek-v3-0324
```bash
# 注意修改 ip 地址
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer token-abc123" \
-d '{
  "model": "DeepSeek-V3-0324",
  "messages": [
    {"role": "system", "content": "你是一个专业助手"},
    {"role": "user", "content": "请介绍四大文明古国"}
  ],
  "max_tokens": 100
}'

```


### 测试  vllm_deepseek-r1
```bash
# 注意修改 ip 地址
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer token-abc123" \
-d '{
  "model": "DeepSeek-R1",
  "messages": [
    {"role": "system", "content": "你是一个专业助手"},
    {"role": "user", "content": "请介绍四大文明古国"}
  ],
  "max_tokens": 100
}'

```


### 测试  vllm_deepseek-r1-00528
```bash
# 注意修改 ip 地址
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer token-abc123" \
-d '{
  "model": "DeepSeek-R1-0528",
  "messages": [
    {"role": "system", "content": "你是一个专业助手"},
    {"role": "user", "content": "请介绍四大文明古国"}
  ],
  "max_tokens": 100
}'

```
