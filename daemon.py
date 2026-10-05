# TuMinh Background Daemon (Chạy ngầm tàng hình)
import time
import subprocess
import os
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent

def notify_windows(title: str, message: str):
    """Gửi thông báo Toast góc màn hình Windows không làm phiền người dùng"""
    ps_cmd = f"""
    [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null
    $template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
    $textNodes = $template.GetElementsByTagName("text")
    $textNodes.Item(0).AppendChild($template.CreateTextNode("{title}")) > $null
    $textNodes.Item(1).AppendChild($template.CreateTextNode("{message}")) > $null
    $toast = [Windows.UI.Notifications.ToastNotification]::new($template)
    [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("TuMinh AI Agent").Show($toast)
    """
    subprocess.run(["powershell", "-WindowStyle", "Hidden", "-Command", ps_cmd], capture_output=True)

def run_daily_study():
    """Đánh thức Copilot giải 1 bài toán khó ở chế độ nền"""
    log_dir = WORKSPACE_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "daemon_study.log"
    
    # Giao bài học cho Copilot ngầm
    task_prompt = "Hãy viết một hàm Python validate dữ liệu email/số điện thoại chuẩn quốc tế, có kèm bài kiểm tra pytest đầy đủ."
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n--- Bắt đầu phiên tự học: {time.ctime()} ---\n")
        
        try:
            # Gọi Copilot CLI ở chế độ ngầm (capture_output)
            res = subprocess.run(
                ["copilot", task_prompt],
                capture_output=True,
                text=True,
                shell=True,
                cwd=str(WORKSPACE_ROOT),
                timeout=300
            )
            f.write(res.stdout)
            notify_windows("TuMinh AI", "Đã hoàn thành 1 phiên tự học từ Copilot thành công!")
        except Exception as e:
            f.write(f"Lỗi: {str(e)}\n")

if __name__ == "__main__":
    # Đợi 10 giây sau khi bật máy tính để mạng ổn định
    time.sleep(10)
    run_daily_study()
