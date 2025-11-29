#!/bin/bash

# vLLM Server Management Helper Script
# Provides easy commands to manage Qwen and Llama servers

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

show_usage() {
    echo "vLLM Server Management Helper"
    echo ""
    echo "Usage: $0 {status|start|stop|restart|logs} [qwen|llama|both]"
    echo ""
    echo "Commands:"
    echo "  status   - Check server status"
    echo "  start    - Start server(s)"
    echo "  stop     - Stop server(s)"
    echo "  restart  - Restart server(s)"
    echo "  logs     - Show server logs"
    echo ""
    echo "Examples:"
    echo "  $0 status both        # Check both servers"
    echo "  $0 start qwen         # Start Qwen server"
    echo "  $0 stop llama         # Stop Llama server"
    echo "  $0 restart both       # Restart both servers"
    echo "  $0 logs qwen          # Show Qwen logs"
}

check_status() {
    local model=$1
    case $model in
        qwen)
            if curl -s http://localhost:8002/v1/models > /dev/null 2>&1; then
                print_color $GREEN "✅ Qwen server: RUNNING (port 8002)"
            else
                print_color $RED "❌ Qwen server: NOT RUNNING (port 8002)"
            fi
            if pgrep -f "Qwen2.5-7B-Instruct" > /dev/null; then
                print_color $CYAN "   Process: ACTIVE (PID: $(pgrep -f 'Qwen2.5-7B-Instruct'))"
            fi
            ;;
        llama)
            if curl -s http://localhost:8003/v1/models > /dev/null 2>&1; then
                print_color $GREEN "✅ Llama server: RUNNING (port 8003)"
            else
                print_color $RED "❌ Llama server: NOT RUNNING (port 8003)"
            fi
            if pgrep -f "Llama-3.1-8B-Instruct" > /dev/null; then
                print_color $CYAN "   Process: ACTIVE (PID: $(pgrep -f 'Llama-3.1-8B-Instruct'))"
            fi
            ;;
        both)
            check_status qwen
            echo ""
            check_status llama
            ;;
    esac
}

start_server() {
    local model=$1
    case $model in
        qwen)
            print_color $BLUE "🚀 Starting Qwen server..."
            bash ./qwen.sh
            ;;
        llama)
            print_color $BLUE "🚀 Starting Llama server..."
            bash ./llama.sh
            ;;
        both)
            start_server qwen
            echo ""
            start_server llama
            ;;
    esac
}

stop_server() {
    local model=$1
    case $model in
        qwen)
            print_color $YELLOW "🛑 Stopping Qwen server..."
            pkill -9 -f "Qwen2.5-7B-Instruct" 2>/dev/null
            pkill -9 -f "port 8002" 2>/dev/null
            sleep 2
            print_color $GREEN "✅ Qwen server stopped"
            ;;
        llama)
            print_color $YELLOW "🛑 Stopping Llama server..."
            pkill -9 -f "Llama-3.1-8B-Instruct" 2>/dev/null
            pkill -9 -f "port 8003" 2>/dev/null
            sleep 2
            print_color $GREEN "✅ Llama server stopped"
            ;;
        both)
            stop_server qwen
            echo ""
            stop_server llama
            ;;
    esac
}

restart_server() {
    local model=$1
    stop_server $model
    echo ""
    sleep 3
    start_server $model
}

show_logs() {
    local model=$1
    case $model in
        qwen)
            if [ -f "logs/qwen_server.log" ]; then
                print_color $CYAN "📋 Qwen Server Logs (last 50 lines):"
                tail -n 50 logs/qwen_server.log
            else
                print_color $RED "❌ No log file found: logs/qwen_server.log"
            fi
            ;;
        llama)
            if [ -f "logs/llama_server.log" ]; then
                print_color $CYAN "📋 Llama Server Logs (last 50 lines):"
                tail -n 50 logs/llama_server.log
            else
                print_color $RED "❌ No log file found: logs/llama_server.log"
            fi
            ;;
        both)
            show_logs qwen
            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo ""
            show_logs llama
            ;;
    esac
}

# Main
if [ $# -lt 1 ]; then
    show_usage
    exit 1
fi

COMMAND=$1
MODEL=${2:-both}

case $COMMAND in
    status)
        check_status $MODEL
        ;;
    start)
        start_server $MODEL
        ;;
    stop)
        stop_server $MODEL
        ;;
    restart)
        restart_server $MODEL
        ;;
    logs)
        show_logs $MODEL
        ;;
    *)
        print_color $RED "❌ Unknown command: $COMMAND"
        echo ""
        show_usage
        exit 1
        ;;
esac
