import time
import threading
import subprocess
import os
import sys

# --- Cài đặt chương trình ---

# Trạng thái hiện tại: False = Dừng click, True = Đang click
clicking = False
# Biến để kiểm soát việc thoát toàn bộ chương trình
exit_program = False

# Tốc độ click (tính bằng giây). 0.02 giây = 50 lần click/giây
DELAY = 0.02 
# Lệnh click chuột trái bằng xdotool (click 1 là chuột trái)
CLICK_COMMAND = "xdotool click 1"
# Phím tắt mềm để BẬT/TẮT click (nhập vào console)
TOGGLE_CHAR = 's'
# Phím tắt mềm để THOÁT chương trình (nhập vào console)
EXIT_CHAR = 'q'

def check_xdotool():
    """Kiểm tra xem xdotool đã được cài đặt chưa."""
    try:
        # Chạy lệnh xdotool version để kiểm tra sự tồn tại
        subprocess.run(['xdotool', 'version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("LỖI: xdotool chưa được cài đặt hoặc không tìm thấy.")
        print("Bạn cần cài đặt xdotool trước bằng lệnh: sudo apt install xdotool")
        print("Chương trình sẽ thoát.")
        return False

def clicker():
    """Hàm thực hiện việc click chuột tự động bằng xdotool trong một luồng riêng."""
    global clicking
    global exit_program
    
    while not exit_program:
        if clicking:
            # Thực hiện lệnh click chuột trái bằng xdotool
            try:
                # Gọi lệnh xdotool thông qua shell
                # Chú ý: Dùng run không cần capture output để tối ưu tốc độ
                subprocess.run(CLICK_COMMAND, shell=True, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except Exception as e:
                # In lỗi và thoát nếu xdotool gặp sự cố
                print(f"Lỗi khi thực hiện xdotool: {e}")
                
            # Chờ một khoảng thời gian giữa các lần click
            time.sleep(DELAY)
        else:
            # Nếu không click, chờ một chút để giảm tải CPU
            time.sleep(0.05)
            
    print("Luồng clicker đã dừng.")

def input_listener():
    """Hàm lắng nghe lệnh điều khiển từ bàn phím console."""
    global clicking
    global exit_program
    
    print(f"-> Nhập '{TOGGLE_CHAR}' (sau đó Enter) để BẬT/TẮT click tự động.")
    print(f"-> Nhập '{EXIT_CHAR}' (sau đó Enter) để THOÁT chương trình.")
    
    while not exit_program:
        try:
            # Đợi người dùng nhập lệnh
            command = input().strip().lower()
            
            if command == TOGGLE_CHAR:
                clicking = not clicking
                if clicking:
                    print(f"-> Trạng thái: Đã BẬT click tự động. Tốc độ: {1/DELAY:.0f} clicks/s. Đang click...")
                else:
                    print("-> Trạng thái: Đã TẮT click tự động. Đang chờ...")
            
            elif command == EXIT_CHAR:
                exit_program = True
                print("\nĐang thoát chương trình Auto Clicker...")
                
            else:
                print(f"Lệnh không hợp lệ. Nhấn '{TOGGLE_CHAR}' để BẬT/TẮT, '{EXIT_CHAR}' để THOÁT.")
                
        except EOFError:
            # Xử lý trường hợp input bị đóng (ví dụ: Ctrl+D)
            exit_program = True
            print("\nĐang thoát chương trình Auto Clicker...")
        except Exception:
             # Xử lý các lỗi khác liên quan đến input
             pass


# --- Khởi chạy chương trình ---

if check_xdotool():
    print("--- Auto Clicker (Dùng xdotool) đã sẵn sàng ---")
    
    # Khởi tạo và chạy luồng clicker (luồng phụ 1)
    click_thread = threading.Thread(target=clicker)
    click_thread.daemon = True 
    click_thread.start()

    # Khởi tạo và chạy luồng lắng nghe input (luồng phụ 2)
    input_thread = threading.Thread(target=input_listener)
    input_thread.daemon = True 
    input_thread.start()
    
    # Giữ luồng chính chạy cho đến khi exit_program là True
    while not exit_program:
        time.sleep(1)
    
    # Đảm bảo các luồng kết thúc sau khi luồng chính thoát
    print("Auto Clicker đã thoát thành công.")
else:
    sys.exit(1)
