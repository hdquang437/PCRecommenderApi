@echo off
REM Điều hướng đến thư mục chứa file chính
cd /d "%~dp0"

REM Kích hoạt môi trường ảo Python (nếu có)
call env\Scripts\activate

REM Chạy API bằng Uvicorn
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

REM Dừng lại sau khi API kết thúc
pause