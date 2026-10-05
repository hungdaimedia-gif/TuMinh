Set WshShell = CreateObject("WScript.Shell")
' 0 nghĩa là ẩn hoàn toàn cửa sổ, không bao giờ chớp màn hình đen
WshShell.Run "pythonw C:\Users\Admim\TuMinh\daemon.py", 0, False
