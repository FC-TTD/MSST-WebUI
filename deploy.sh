#!/bin/bash

# MSST-WebUI 部署脚本
# 用法: ./deploy.sh [部署类型]
# 部署类型: stage (Gradio应用) 或 api (API服务器)

DEPLOY_TYPE=${1}
MODE=${2:-deploy}

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # 无颜色

# 显示横幅
echo -e "${GREEN}"
echo "====================================="
echo "     MSST-WebUI 部署工具"
echo "====================================="
echo -e "${NC}"

# 根据部署类型设置目标主机和标签
case "$DEPLOY_TYPE" in
    "stage")
        TARGET_HOST="ttd-worker"
        TAGS="stage"
        echo -e "${GREEN}开始部署 MSST-WebUI Gradio 应用到 $TARGET_HOST${NC}"
        ;;
    *)
        echo -e "${RED}错误: 无效的部署类型 '$DEPLOY_TYPE'。请使用 'stage'${NC}"
        exit 1
        ;;
esac

EXTRA_ARGS=""
if [ "$MODE" = "sync" ]; then
    echo -e "${YELLOW}当前运行于同步模式：仅同步文件，不执行 Docker 命令${NC}"
    EXTRA_ARGS="--skip-tags docker"
fi

# 执行ansible playbook
ANSIBLE_STDOUT_CALLBACK=debug echo -e "${YELLOW}执行命令: ansible-playbook ./docker/playbook.yml --tags $TAGS -i "$TARGET_HOST," -v $EXTRA_ARGS${NC}"
ANSIBLE_STDOUT_CALLBACK=debug ansible-playbook ./docker/playbook.yml --tags $TAGS -i "$TARGET_HOST," -v $EXTRA_ARGS

# 检查部署结果
DEPLOY_RESULT=$?
if [ $DEPLOY_RESULT -eq 0 ]; then
    echo -e "${GREEN}部署成功!${NC}"
else
    echo -e "${RED}部署失败，返回代码: $DEPLOY_RESULT${NC}"
    exit $DEPLOY_RESULT
fi
