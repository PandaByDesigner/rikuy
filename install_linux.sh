#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
data_home="${XDG_DATA_HOME:-${HOME}/.local/share}"
config_home="${XDG_CONFIG_HOME:-${HOME}/.config}"
bin_home="${HOME}/.local/bin"
app_dir="${data_home}/rikuy"
config_dir="${config_home}/rikuy"
applications_dir="${data_home}/applications"
icon_dir="${data_home}/icons/hicolor/256x256/apps"

install -d "$app_dir" "$config_dir" "$bin_home" "$applications_dir" "$icon_dir"
install -m 0644 "$repo_dir/rikuy.py" "$app_dir/rikuy.py"
install -m 0644 "$repo_dir/style.qss" "$app_dir/style.qss"
install -m 0644 "$repo_dir/Rikuy_Condor_Icon.ico" "$app_dir/Rikuy_Condor_Icon.ico"
install -m 0644 "$repo_dir/config.example.ini" "$app_dir/config.example.ini"
install -m 0755 "$repo_dir/linux/rikuy" "$bin_home/rikuy"
install -m 0644 "$repo_dir/linux/rikuy.desktop" "$applications_dir/rikuy.desktop"

if [[ ! -e "$config_dir/config.ini" ]]; then
    install -m 0644 "$repo_dir/config.example.ini" "$config_dir/config.ini"
fi

if command -v magick >/dev/null 2>&1; then
    magick "$repo_dir/Rikuy_Condor_Icon.ico[5]" -resize 256x256 "$icon_dir/rikuy.png"
else
    printf 'ImageMagick is unavailable; the app launcher will use a generic icon.\n' >&2
fi

if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$applications_dir"
fi

printf 'Rikuy installed. Launch it from the app menu or run: %s\n' "$bin_home/rikuy"
