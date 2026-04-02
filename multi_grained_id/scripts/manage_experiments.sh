#!/bin/bash
# 实验管理器守护进程启动脚本

PROJECT_DIR="/mnt/workspace/open_research/autoresearch/multi_grained_id"
LOG_FILE="/tmp/experiment_manager.log"
PID_FILE="/tmp/experiment_manager.pid"

cd "$PROJECT_DIR"

case "$1" in
    start)
        # 检查是否已在运行
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p $PID > /dev/null 2>&1; then
                echo "实验管理器已在运行 (PID: $PID)"
                exit 1
            fi
        fi
        
        echo "🚀 启动实验管理器..."
        nohup python experiment_manager.py --daemon --interval 30 > "$LOG_FILE" 2>&1 &
        echo $! > "$PID_FILE"
        echo "✅ 实验管理器已启动 (PID: $(cat $PID_FILE))"
        echo "   日志文件：$LOG_FILE"
        ;;
    
    stop)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p $PID > /dev/null 2>&1; then
                echo "🛑 停止实验管理器 (PID: $PID)..."
                kill $PID
                rm -f "$PID_FILE"
                echo "✅ 已停止"
            else
                echo "⚠️  进程不存在，清理 PID 文件"
                rm -f "$PID_FILE"
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
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p $PID > /dev/null 2>&1; then
                echo "✅ 实验管理器正在运行 (PID: $PID)"
                echo ""
                echo "最近日志:"
                tail -20 "$LOG_FILE"
            else
                echo "❌ 进程已停止 (PID 文件存在但进程不存在)"
            fi
        else
            echo "⚠️  实验管理器未运行"
        fi
        ;;
    
    log)
        if [ -f "$LOG_FILE" ]; then
            tail -50 "$LOG_FILE"
        else
            echo "⚠️  日志文件不存在"
        fi
        ;;
    
    *)
        echo "用法：$0 {start|stop|restart|status|log}"
        echo ""
        echo "命令:"
        echo "  start   - 启动守护进程"
        echo "  stop    - 停止守护进程"
        echo "  restart - 重启守护进程"
        echo "  status  - 查看状态"
        echo "  log     - 查看日志"
        exit 1
        ;;
esac
