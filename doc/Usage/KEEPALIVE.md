# 功能介绍: KEEPALIVE模式

## Docker/Hugging Face Space/Node.js

通过设置变量`KEEPALIVE`为`1`时，可以启用KEEPALIVE响应模式。
```
KEEPALIVE=1
```
启用KEEPALIVE模式后，使用流式请求到脚本，脚本会持续向客户端发送心跳响应以保持客户端的持续连接，防止客户端发生异常断开的情况。

注意，当使用`KEEPALIVE`功能时，请确保使用的请求密钥(Worker Api Key)的安全设定为关闭(Disabled)，且在请求时使用流式响应。
`KEEPALIVE`功能不会影响到非流式响应或启用了安全设定的流式响应。

## Cloudflare Worker

Worker版本使用与docker版本不同，需要在worker环境变量中将设置`KEEPALIVE_ENABLED`为`true`，设置正确后使用一个关闭了安全设定的worker api key并启用流式传输即可使用该功能。
