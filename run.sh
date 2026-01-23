#!/bin/bash

# Colors
GREEN="\033[1;32m"
YELLOW="\033[1;33m"
RED="\033[1;31m"
RESET="\033[0m"

FRONTEND_DIR="vision-truth-finder"
BACKEND_DIR="project"
PID_FILE=".serve_pid"

# Function to start the app
start_app() {
  DEFAULT_PORT=8080
  PORT=${2:-$DEFAULT_PORT}
  echo -e "${YELLOW}🧱 Building Vite frontend...${RESET}"
  cd "$FRONTEND_DIR"
  npm install
  npm run build
  cd ..

  echo -e "${YELLOW}🔧 Starting Docker containers...${RESET}"
  cd "$BACKEND_DIR"
  docker compose up --detach --build
  cd ..

  echo -e "${YELLOW}🌐 Serving frontend on http://localhost:${PORT}...${RESET}"
  cd "$FRONTEND_DIR"
  nohup npm run preview -- --port "${PORT}" > ../frontend.log 2>&1 &
  echo "$PORT:$!" > ../$PID_FILE
  cd ..

  echo -e "${GREEN}✅ App started successfully!${RESET}"
}

# Function to stop the app
stop_app() {
  echo -e "${YELLOW}🛑 Stopping Docker containers...${RESET}"
  cd "$BACKEND_DIR"
  docker compose down
  cd ..

  # Stop frontend with improved process handling
  if [ -f "$PID_FILE" ]; then
    IFS=":" read PORT SERVE_PID < "$PID_FILE"
    if ps -p "$SERVE_PID" > /dev/null 2>&1; then
      echo -e "${YELLOW}🧼 Stopping Vite preview (PID $SERVE_PID)...${RESET}"
      
      # Try graceful shutdown first (SIGTERM)
      kill -TERM "$SERVE_PID" 2>/dev/null
      
      # Wait up to 10 seconds for graceful shutdown
      for i in {1..10}; do
        if ! ps -p "$SERVE_PID" > /dev/null 2>&1; then
          echo -e "${GREEN}✅ Frontend stopped gracefully${RESET}"
          break
        fi
        sleep 1
      done     
      # If still running, force kill
      if ps -p "$SERVE_PID" > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠️ Graceful shutdown failed, force killing...${RESET}"
        kill -KILL "$SERVE_PID" 2>/dev/null
        sleep 1
        
        if ps -p "$SERVE_PID" > /dev/null 2>&1; then
          echo -e "${RED}❌ Failed to stop process $SERVE_PID${RESET}"
        else
          echo -e "${GREEN}✅ Frontend force stopped${RESET}"
        fi
      fi
    else
      echo -e "${YELLOW}⚠️ PID $SERVE_PID not running${RESET}"
    fi
    rm "$PID_FILE"
  else
    echo -e "${YELLOW}⚠️ No PID file found${RESET}"
  fi

  # Additional cleanup: kill any remaining processes on port
  echo -e "${YELLOW}🔍 Checking for any remaining processes on port $PORT...${RESET}"
  PORT_PID=$(lsof -ti:$PORT 2>/dev/null)
  if [ ! -z "$PORT_PID" ]; then
    echo -e "${YELLOW}Found process $PORT_PID on port $PORT, killing it...${RESET}"
    kill -KILL $PORT_PID 2>/dev/null
    echo -e "${GREEN}✅ Port $PORT cleaned up${RESET}"
  fi

  echo -e "${GREEN}🧹 Clean shutdown complete${RESET}"
}

# Function to check status
status_app() {
  echo -e "${YELLOW}📊 Checking application status...${RESET}"
  
  # Check Docker containers
  echo -e "${YELLOW}Docker containers:${RESET}"
  cd "$BACKEND_DIR"
  if docker compose ps | grep -q "Up"; then
    echo -e "${GREEN}✅ Backend containers running${RESET}"
  else
    echo -e "${RED}❌ Backend containers not running${RESET}"
  fi
  cd ..
  
  # Check frontend
  if [ -f "$PID_FILE" ]; then
    IFS=":" read PORT SERVE_PID < "$PID_FILE"

    if ps -p "$SERVE_PID" > /dev/null 2>&1; then
      echo -e "${GREEN}✅ Frontend running (PID $SERVE_PID)${RESET}"
    else
      echo -e "${RED}❌ Frontend not running (stale PID file)${RESET}"
      rm "$PID_FILE"
    fi
  else
    echo -e "${RED}❌ Frontend not running (no PID file)${RESET}"
  fi
  
  # Check port
  if lsof -ti:$PORT > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Port $PORT is in use${RESET}"
  else
    echo -e "${RED}❌ Port $PORT is free${RESET}"
  fi
}

# Function to restart the app
restart_app() {
  echo -e "${YELLOW}🔄 Restarting application...${RESET}"

  if [ -f "$PID_FILE" ]; then
    IFS=":" read SAVED_PORT _ < "$PID_FILE"
  else
    SAVED_PORT=""
  fi

  stop_app
  sleep 2

  if [ -n "$SAVED_PORT" ]; then
    start_app "start" "$SAVED_PORT"
  else
    start_app
  fi
}

# Entry point
case "$1" in
  start)
    start_app "$@"
    ;;
  stop)
    stop_app
    ;;
  restart)
    restart_app
    ;;
  status)
    status_app
    ;;
  *)
    echo -e "${YELLOW}Usage:${RESET} ./run.sh {start|stop|restart|status}"
    echo -e "${YELLOW}Commands:${RESET}"
    echo -e "  start   - Start the application"
    echo -e "  stop    - Stop the application"
    echo -e "  restart - Restart the application"
    echo -e "  status  - Check application status"
    ;;
esac
