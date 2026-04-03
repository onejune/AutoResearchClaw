#!/bin/bash
# 实验管理器守护进程启动脚本
# 所有路径从 expkit.conf 读取，相对于项目根目录

# 自动推断项目根目录（本脚本在 scripts/ 下，上一级即项目根）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 读取 expkit.conf
_conf() {
    local key="$1" default="$2"
    local val
    val=$(grep -E "^${key}=" "$PROJECT_DIR/expkit.conf" 2>/dev/null | tail -1 | cut -d= -f2-)
    echo "${val:-$default}"
}

DAEMON_LOG="$PROJECT_DIR/$(_conf DAEMON_LOG "logs/daemon.log")"
DAEMON_PID="$PROJECT_DIR/$(_conf DAEMON_PID "logs/daemon.pid")"
DAEMON_INTERVAL="$(_conf DAEMON_INTERVAL "30")"

# 确保日志目录存在
mkdir -p "$(dirname "$DAEMON_LOG")"
mkdir -p "$(dirname "$DAEMON_PID")"

cd "$PROJECT_DIR"

case "$1" in
    start)
        if [ -f "$DAEMON_PID" ]; then
            PID=$(cat "$DAEMON_PID")
            if ps -p $PID > /dev/null 2>&1; then
                echo "实验管理器已在运行 (PID: $PID)"
                exit 1
            fi
        fi

        echo "🚀 启动实验管理器..."
        nohup python -u experiment_manager.py --daemon --interval "$DAEMON_INTERVAL" > "$DAEMON_LOG" 2>&1 &
        echo $! > "$DAEMON_PID"
        echo "✅ 实验管理器已启动 (PID: $(cat $DAEMON_PID))"
        echo "   日志文件：$DAEMON_LOG"
        ;;

    stop)
        if [ -f "$DAEMON_PID" ]; then
            PID=$(cat "$DAEMON_PID")
            if ps -p $PID > /dev/null 2>&1; then
                echo "🛑 停止实验管理器 (PID: $PID)..."
                kill $PID
                rm -f "$DAEMON_PID"
                echo "✅ 已停止"
            else
                echo "⚠️  进程不存在，清理 PID 文件"
                rm -f "$DAEMON_PID"
            fi
        else
            echo "⚠️  实验管理器未运行"
        fi
        ;;

    restart)
        $0 stop
        sleep 2
        $0 start
        ;;

    status)
        if [ -f "$DAEMON_PID" ]; then
            PID=$(cat "$DAEMON_PID")
            if ps -p $PID > /dev/null 2>&1; then
                echo "✅ 实验管理器正在运行 (PID: $PID)"
                echo ""
                echo "最近日志:"
                tail -20 "$DAEMON_LOG"
            else
                echo "❌ 进程已停止 (PID 文件存在但进程不存在)"
            fi
        else
            echo "⚠️  实验管理器未运行"
        fi
        ;;

    log)
        if [ -f "$DAEMON_LOG" ]; then
            tail -50 "$DAEMON_LOG"
        else
            echo "⚠️  日志文件不存在：$DAEMON_LOG"
        fi
        ;;

    *)
        echo "用法：$0 {start|stop|restart|status|log}"
        exit 1
        ;;
esac
