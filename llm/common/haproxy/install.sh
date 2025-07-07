#!/bin/bash

# Usage: ./deploy.sh <driver_package_path> [tool_path]

set -euo pipefail  # Enable strict mode: exit on error, unset variables, and pipe failures

# Function to display error messages and exit
error_exit() {
    echo "[ERROR] $1" >&2
    exit 1
}

# Function to display informational messages
info_msg() {
    echo "[INFO] $1"
}

# Check if script is run with root privileges
check_root() {
    if [ "$(id -u)" -ne 0 ]; then
        error_exit "This script must be run as root. Please use sudo."
    fi
}

# Function to detect system architecture
detect_architecture() {
    ARCH=$(uname -m)
    case "$ARCH" in
        x86_64)  echo "x86" ;;
        aarch64) echo "arm" ;;
        arm64)   echo "arm" ;;
        *)       error_exit "Unsupported architecture: $ARCH" ;;
    esac
}

DEFAULT_VASMI_PATH="/usr/bin/vasmi"

# Function to validate and set vasmi path
set_vasmi_path() {
    local vasmi_path="$1"
    
    if [ -n "$vasmi_path" ]; then
        # Use provided path if specified
        if [ ! -x "$vasmi_path" ]; then
            error_exit "Specified vasmi path is not executable: $vasmi_path"
        fi
        echo "$vasmi_path"
    else
        if [ -z "${DEFAULT_VASMI_PATH:-}" ]; then
            error_exit "Default vasmi path is not exists"
        fi
        # Use default path if not specified
        if [ ! -x "$DEFAULT_VASMI_PATH" ]; then
            error_exit "Default vasmi not found at $DEFAULT_VASMI_PATH and no alternate path provided"
        fi
        echo "$DEFAULT_VASMI_PATH"
    fi
}

# Validate input parameters
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    error_exit "Usage: $0 <driver_package_path> [vasmi_path]"
fi

DRIVER_PACKAGE="$1"
VASMI_PATH=$(set_vasmi_path "${2:-}")

# Verify driver package exists
if [ ! -f "$DRIVER_PACKAGE" ]; then
    error_exit "Driver package not found at specified path: $DRIVER_PACKAGE"
fi

# Main install function
install_env() {
    info_msg "Starting install process..."

    # Detect system architecture
    info_msg "Detecting system architecture..."
    ARCH=$(detect_architecture)
    info_msg "Detected architecture: $ARCH"
    
    # Check if driver is already installed
    info_msg "Checking for existing driver installation..."
    if lsmod | grep -i vastai_pci; then
        info_msg "Existing driver found. Proceeding with uninstallation..."
        
        # Uninstall existing driver
        if ! sudo "$DRIVER_PACKAGE" uninstall; then
            error_exit "Failed to uninstall existing driver."
        fi
        info_msg "Driver uninstalled successfully."
    else
        info_msg "No existing driver installation found."
    fi
    
    # Install new driver
    info_msg "Installing new driver..."
    if ! sudo "$DRIVER_PACKAGE" install; then
        error_exit "Failed to install new driver."
    fi
    info_msg "Driver installed successfully."
    
    # Get number of cards
    info_msg "Detecting number of cards..."
    DIE_NUM=$(lspci -d:100 2>/dev/null | wc -l)
    if [ "$DIE_NUM" -gt 0 ]; then
        CARD_NUM=$((DIE_NUM / 4))
    else
        error_exit "Failed to detect number of cards or invalid card count."
    fi
    # CARD_NUM=$($VASMI_PATH getcardnum)
    if [ -z "$CARD_NUM" ] || ! [[ "$CARD_NUM" =~ ^[0-9]+$ ]]; then
        error_exit "Failed to detect number of cards or invalid card count."
    fi
    info_msg "Detected $CARD_NUM cards in the system."
    
    # Configure DPM settings
    info_msg "Configuring DPM settings..."
    if [ "$CARD_NUM" -gt 0 ]; then
        # Set kernel parameter
        if ! sudo "$DRIVER_PACKAGE" install --setkoparam "dpm=1"; then
            error_exit "Failed to set kernel parameter for DPM."
        fi
        
        # Set DPM for all cards (0-based index)
        CARD_RANGE="0-$((CARD_NUM - 1))"
        if ! $VASMI_PATH setconfig dpm=enable -d "$CARD_RANGE"; then
            error_exit "Failed to enable DPM for cards $CARD_RANGE."
        fi
        info_msg "DPM configured successfully for cards $CARD_RANGE."
    else
        info_msg "No cards found to configure DPM."
    fi

    info_msg "Configuring CPU performance governor..."
    # Install required tools
    if ! command -v cpupower >/dev/null 2>&1; then
        info_msg "Installing cpupower tools..."
        if ! apt-get install -y linux-tools-common linux-tools-$(uname -r); then
            error_exit "Failed to install cpupower tools"
        fi
    fi
    # Set performance governor
    if cpupower frequency-set --governor performance >/dev/null 2>&1; then
        info_msg "CPU governor successfully set to performance mode"
    else
        error_exit "Failed to set CPU performance governor"
    fi
    # Verify the setting
    CURRENT_GOVERNOR=$(cpupower frequency-info -p | grep -i "governor" | awk '{print $3}')
    if [ "$CURRENT_GOVERNOR" != '"performance"' ]; then
        error_exit "CPU governor verification failed. Current governor: $CURRENT_GOVERNOR"
    fi

}

# Main execution
check_root
install_env

exit 0