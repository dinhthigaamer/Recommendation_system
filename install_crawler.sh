#!/usr/bin/env bash
set -e

# Chuyển vào thư mục chứa script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ưu tiên sử dụng môi trường crawl_env vừa tạo trong thư mục nếu có
if [ -f "$SCRIPT_DIR/crawl_env/bin/pip" ]; then
    PIP_CMD="$SCRIPT_DIR/crawl_env/bin/pip"
    PYTHON_CMD="$SCRIPT_DIR/crawl_env/bin/python"
    echo "==> Đang sử dụng Python từ: $SCRIPT_DIR/crawl_env"
else
    PIP_CMD="pip"
    PYTHON_CMD="python3"
    echo "==> Đang sử dụng Python/pip hiện tại trong PATH"
fi

echo "==> 1. Cập nhật pip..."
"$PIP_CMD" install --upgrade pip

echo "==> 2. Cài đặt các package cần thiết từ requirements_crawler.txt..."
"$PIP_CMD" install -r requirements_crawler.txt

echo "==> 3. Cài đặt trình duyệt Playwright (Chromium)..."
"$PYTHON_CMD" -m playwright install chromium

echo ""
echo "=========================================================="
echo " [THÀNH CÔNG] Đã cài đặt xong toàn bộ môi trường crawler!"
echo "=========================================================="
echo "Để kích hoạt môi trường:"
echo "    conda activate \"$SCRIPT_DIR/crawl_env\""
echo ""
echo "Để chạy crawler:"
echo "    python crawl_books.py"
echo "=========================================================="
