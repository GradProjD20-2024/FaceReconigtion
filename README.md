# Face Recognition API

## Start service

- Tải file weight từ [link](https://drive.google.com/file/d/1Ujr2VAdMuHkFdPviP8dSA862SKlnFSfm/view?usp=sharing) rồi lưu vào đường dẫn `app/weights/best_model.pth`
- Sửa đường dẫn model trong file `.env` thành `MODEL_PATH="weights/best_model.pth"`
- Thay đổi threshold tùy thích trong file `.env` (default: 0.75)
- Chạy docker compose

```bash
cd app
docker compose up -d
```


## Enroll a new face

```bash
curl -X POST "http://localhost:8001/api/v1/enroll" \
-H "Content-Type: multipart/form-data" \
-F "file=@path/to/image.jpg"
```

# Check in a face

```bash
curl -X POST "http://localhost:8001/api/v1/check-in" \
-H "Content-Type: multipart/form-data" \
-F "file=@path/to/image.jpg"
```
