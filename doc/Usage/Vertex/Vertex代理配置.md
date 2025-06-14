# Vertex 代理配置

通过配置 `VERTEX` 环境变量，启用 Vertex 代理功能，添加使用 Vertex AI 平台的 Gemini 模型。

---

## 启用 Generative Language API	

 * 如果您的项目还未启用 Generative Language API，需要先设置启用，已经启用可以跳过。
 * 在左侧边栏中选择`API 和服务`中的`已启用的API 和服务`。
  ![](image/5.0.jpg)
 * 在打开的页面中点击`+ 启用API 和服务`。
  ![](image/5.1.jpg)
 * 在搜索栏中输入搜索`Generative Language API`。
  ![](image/5.2.jpg)
 * 选择启用 Generative Language API
  ![](image/5.3.jpg)

---

## 创建服务账号

 * 在 [Google Cloud Platform](https://cloud.google.com/) 中登陆并激活账户。
 * 在左侧边栏中选择`IAM和管理`并点击`服务账号`。
   ![](image/1.0.jpg)
 * 点击`创建服务账号`。
   ![](image/2.0.jpg)
 * 在`服务账号详情`中随意填写一个`服务账号ID`。选择`创建并继续`。
   ![](image/3.0.jpg)
 * 在角色中选择`Vertex Al Service Agent`。
   ![](image/3.1.jpg)
   ![](image/3.2.jpg)
 * 点击`完成`创建服务账号。
   ![](image/3.3.jpg)

---

## 创建API凭证

 * 点击服务账号右侧的三点图标并选择`管理密钥`。
  ![](image/4.0.jpg)
  ![](image/4.1.jpg)
 * 在页面中点击`添加键`并选择`创建新密钥`。
  ![](image/4.2.jpg)
 * 选择JSON格式并点击创建。
  ![](image/4.3.jpg)

---

## 添加 API 凭证到环境变量

在变量文件中添加一个`VERTEX`变量，并将下载的JSON格式变量完整地粘贴到变量中。\
参考格式
```
VERTEX={
  "type": "service_account",
  "project_id": "XXXXXXXXXXX",
  "private_key_id": "XXXXXXXXXXX",
  "private_key": "-----BEGIN PRIVATE KEY-----\ABCD\n-----END PRIVATE KEY-----\n",
  "client_email": "XXXXXXXXXXX.iam.gserviceaccount.com",
  "client_id": "XXXXXXXXXXX",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/
  "universe_domain": "googleapis.com"
}
```