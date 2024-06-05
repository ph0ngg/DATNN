from moviepy.video.io.VideoFileClip import VideoFileClip
# Đường dẫn tới video gốc
input_video_path = r'D:\\PhongNghiem\\FastAPI1\\upload_folder\\4p-c1.avi'

# Đường dẫn lưu video đã cắt
output_video_path = r'D:\\PhongNghiem\\FastAPI1\\upload_folder\\4p-c1-cut.avi'

# Thời gian bắt đầu và kết thúc cắt (tính bằng giây)
start_time = 10  # Bắt đầu từ giây thứ 10
end_time = 40    # Kết thúc tại giây thứ 20

# Đọc video gốc
video = VideoFileClip(input_video_path)

# Cắt video
cut_video = video.subclip(start_time, end_time)

# Lưu video đã cắt
cut_video.write_videofile(output_video_path, codec="libx264")

# Giải phóng bộ nhớ
video.close()
cut_video.close()