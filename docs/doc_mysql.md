# mysql

## 访问数据库

- 用户信息
  - ip: 10.23.4.220
  - port: 3307
  - user: service
  - pw: modelzoo666

### 命令行访问

- docker内访问

  ```bash
  # 1. 进入docker
  docker exec -it vamp_mysql /bin/bash
  # 2. 用户登录
  mysql -uservice -p --default-character-set=utf8
  # 输入密码：modelzoo666
  ```
- mysql操作命令

  [w3cschool](https://www.w3school.com.cn/sql/sql_select.asp)

### Navicat界面访问

1. win系统下载安装[Navicat](https://learnku.com/articles/67706)
2. 使用软件，连接数据库：

   ![connect](../images/mysql/mysql_connect.jpg)
