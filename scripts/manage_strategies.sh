#!/usr/bin/env bash
# FinRL Multi-Strategy Manager
#
# Manages Docker-based trading strategies on a remote desktop host.
# Run from your dev laptop — uses Docker context for remote execution.
#
# Setup (one-time):
#   ./scripts/manage_strategies.sh setup <desktop-ip> [ssh-user]
#
# Usage:
#   ./scripts/manage_strategies.sh build          # Build engine image
#   ./scripts/manage_strategies.sh up             # Start all strategies
#   ./scripts/manage_strategies.sh up ib          # Start IB strategies only
#   ./scripts/manage_strategies.sh up gmgp1-gold  # Start one strategy
#   ./scripts/manage_strategies.sh down           # Stop all
#   ./scripts/manage_strategies.sh stop gmgp1-gold
#   ./scripts/manage_strategies.sh restart gmgp1-gold
#   ./scripts/manage_strategies.sh ps             # Status
#   ./scripts/manage_strategies.sh logs gmgp1-gold
#   ./scripts/manage_strategies.sh logs gmgp1-gold -f   # Follow
#   ./scripts/manage_strategies.sh shell gmgp1-gold      # Shell into container
#   ./scripts/manage_strategies.sh context [local|desktop] # Switch Docker context
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_DIR="$PROJECT_ROOT/docker/live"
COMPOSE_BASE="$COMPOSE_DIR/docker-compose.yaml"
COMPOSE_DESKTOP="$COMPOSE_DIR/docker-compose.desktop.yaml"
CONTEXT_NAME="finrl-desktop"

# Detect docker compose command (v2 plugin vs v1 standalone)
if docker compose version &>/dev/null; then
    DC_BIN="docker compose"
elif docker-compose version &>/dev/null; then
    DC_BIN="docker-compose"
else
    echo "ERROR: docker compose not found. Install Docker Desktop."
    exit 1
fi
DC="$DC_BIN -f $COMPOSE_BASE -f $COMPOSE_DESKTOP"

usage() {
    cat <<'EOF'
FinRL Multi-Strategy Manager

Commands:
  setup <ip> [user]     Set up Docker context for remote desktop
  context [local|desktop] Switch Docker context
  build                 Build the shared engine image
  up [profile|service]  Start strategies (profile: ib, crypto, all; or service name)
  down                  Stop all strategies
  stop <service>        Stop a single strategy
  restart <service>     Restart a single strategy
  ps                    Show status of all containers
  logs <service> [-f]   View logs (add -f to follow)
  shell <service>       Open shell in a container
  sync <project-path>   Sync project files to desktop via rsync
  vnc                   Open VNC to IB Gateway (port 5900)
  portainer             Show Portainer URL
EOF
}

# ---- Commands ----

cmd_setup() {
    local ip="${1:?Usage: setup <desktop-ip> [ssh-user]}"
    local user="${2:-$(whoami)}"

    echo "Setting up Docker context for ${user}@${ip}..."

    # Remove existing context if present
    docker context rm "$CONTEXT_NAME" 2>/dev/null || true

    # Create SSH-based Docker context
    docker context create "$CONTEXT_NAME" \
        --docker "host=ssh://${user}@${ip}"

    echo ""
    echo "Context '$CONTEXT_NAME' created."
    echo ""
    echo "Next steps on the desktop machine (${ip}):"
    echo "  1. Install Docker Desktop (if not already)"
    echo "  2. Enable OpenSSH Server:"
    echo "     Settings > System > Optional Features > Add: OpenSSH Server"
    echo "     Then: Start-Service sshd; Set-Service -Name sshd -StartupType Automatic"
    echo "  3. Set up SSH key auth (from this laptop):"
    echo "     ssh-copy-id ${user}@${ip}"
    echo "  4. Clone/sync the project on desktop:"
    echo "     git clone <repo> C:\\FinRL\\FinRL-Pro_DS"
    echo "  5. Create .env on desktop:"
    echo "     cd C:\\FinRL\\FinRL-Pro_DS\\docker\\live && cp .env.example .env"
    echo "     # Fill in credentials"
    echo ""
    echo "To activate: ./scripts/manage_strategies.sh context desktop"
}

cmd_context() {
    local target="${1:-}"
    if [ -z "$target" ]; then
        echo "Current context:"
        docker context show
        echo ""
        echo "Available contexts:"
        docker context ls
        return
    fi

    case "$target" in
        desktop)
            docker context use "$CONTEXT_NAME"
            echo "Now targeting desktop Docker host"
            ;;
        local)
            docker context use default
            echo "Now targeting local Docker host"
            ;;
        *)
            echo "Unknown context: $target (use 'local' or 'desktop')"
            exit 1
            ;;
    esac
}

cmd_build() {
    echo "Building finrl-live-engine image..."
    $DC build engine-base
    echo "Build complete."
}

cmd_up() {
    local target="${1:-}"

    if [ -z "$target" ]; then
        echo "Starting all strategies..."
        $DC --profile all up -d
    elif [[ "$target" == "ib" || "$target" == "crypto" || "$target" == "all" ]]; then
        echo "Starting profile: $target"
        $DC --profile "$target" up -d
    else
        # Specific service name — infra starts automatically via depends_on
        echo "Starting: $target"
        $DC up -d "$target"
    fi

    echo ""
    $DC ps
}

cmd_down() {
    echo "Stopping all strategies..."
    $DC --profile all down
}

cmd_stop() {
    local service="${1:?Usage: stop <service-name>}"
    echo "Stopping: $service"
    $DC stop "$service"
}

cmd_restart() {
    local service="${1:?Usage: restart <service-name>}"
    echo "Restarting: $service"
    $DC restart "$service"
}

cmd_ps() {
    $DC ps -a
}

cmd_logs() {
    local service="${1:?Usage: logs <service-name> [-f]}"
    shift
    $DC logs "$service" "$@"
}

cmd_shell() {
    local service="${1:?Usage: shell <service-name>}"
    docker exec -it "$service" bash
}

cmd_sync() {
    local remote_path="${1:?Usage: sync <user@ip:/path/to/project>}"
    echo "Syncing project to desktop..."
    rsync -avz \
        --exclude '.git' --exclude 'data/' --exclude 'wandb/' \
        --exclude '__pycache__' --exclude '*.pyc' --exclude '.env' \
        --exclude 'checkpoints/' --exclude '.agent/memory/' \
        --exclude 'randd_log.md' --exclude 'randd_archive/' \
        --exclude '*.pth' --exclude 'experiments/' \
        "$PROJECT_ROOT/" "$remote_path/"
    echo "Sync complete."
}

cmd_vnc() {
    local ip
    ip=$(docker context inspect "$CONTEXT_NAME" 2>/dev/null \
        | sed -n 's|.*ssh://[^@]*@\([^"]*\).*|\1|p' || echo "localhost")
    [ -z "$ip" ] && ip="localhost"
    echo "VNC: ${ip}:5900"
    echo "Connect with any VNC client to debug IB Gateway."
}

cmd_portainer() {
    local ip
    ip=$(docker context inspect "$CONTEXT_NAME" 2>/dev/null \
        | sed -n 's|.*ssh://[^@]*@\([^"]*\).*|\1|p' || echo "localhost")
    [ -z "$ip" ] && ip="localhost"
    echo "Portainer: https://${ip}:9443"
    echo "First-time setup: create admin user at the URL above."
}

# ---- Dispatch ----

cmd="${1:-help}"
shift || true

case "$cmd" in
    setup)      cmd_setup "$@" ;;
    context)    cmd_context "$@" ;;
    build)      cmd_build "$@" ;;
    up|start)   cmd_up "$@" ;;
    down)       cmd_down "$@" ;;
    stop)       cmd_stop "$@" ;;
    restart)    cmd_restart "$@" ;;
    ps|status)  cmd_ps "$@" ;;
    logs)       cmd_logs "$@" ;;
    shell)      cmd_shell "$@" ;;
    sync)       cmd_sync "$@" ;;
    vnc)        cmd_vnc "$@" ;;
    portainer)  cmd_portainer "$@" ;;
    help|--help|-h) usage ;;
    *)
        echo "Unknown command: $cmd"
        usage
        exit 1
        ;;
esac
