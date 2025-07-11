#!/bin/bash

echo "正在取消环境变量代理..."

# 取消环境变量代理
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

echo "正在取消 Git 代理设置..."

# 取消 Git 代理设置
git config --global --unset http.proxy
git config --global --unset https.proxy

echo "代理设置已清除完成！"
echo "当前环境变量:"
echo "http_proxy: $http_proxy"
echo "https_proxy: $https_proxy"
echo "HTTP_PROXY: $HTTP_PROXY"
echo "HTTPS_PROXY: $HTTPS_PROXY" 
# https://sakuracat1203.xn--3iq226gfdb94q.com/api/v1/client/subscribe?token=86f33c6db1ea5ef6371a16fc6abe3459