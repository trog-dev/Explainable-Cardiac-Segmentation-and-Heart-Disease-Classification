# Phân đoạn tim 3D và phân loại bệnh tim bằng Explainable AI

Đồ án môn **Deep Learning** tại Trường Đại học Sư phạm Kỹ thuật Vĩnh Long. Dự án phân đoạn thất trái, cơ tim và thất phải trên MRI tim 3D bằng **SegResNet3D** và **VNet3D**. Nhánh phân loại kết hợp các chỉ số tính từ mặt nạ VNet3D với đặc trưng sâu để dự đoán ba nhóm bệnh. Các notebook phân đoạn có Grad-CAM 3D và bản đồ độ bất định để khảo sát dự đoán.

## Notebook và tệp đi kèm

| Thành phần | Nội dung |
| --- | --- |
| [SegResNet3D.ipynb](./22022010-22022012-22022013/Phandoan/SegResNet3D.ipynb) | Huấn luyện mô hình MONAI, đánh giá phân đoạn và trực quan hóa Grad-CAM, độ bất định. |
| [VNet3D.ipynb](./22022010-22022012-22022013/Phandoan/VNet3D.ipynb) | Huấn luyện VNet3D bằng PyTorch, đánh giá phân đoạn và trực quan hóa theo từng tầng. |
| [VNet3D-3LOP.ipynb](./22022010-22022012-22022013/Phanloai/VNet3D-3LOP.ipynb) | Trích xuất chỉ số từ mặt nạ dự đoán và huấn luyện bộ phân loại ba nhóm. |
| [Trọng số SegResNet3D](./22022010-22022012-22022013/Phandoan/best_model_SegResNet3D/best_model.pth) và [nhật ký huấn luyện](./22022010-22022012-22022013/Phandoan/best_model_SegResNet3D/training_log.csv) | Tệp `state_dict` và CSV để đánh giá lại mô hình hoặc vẽ biểu đồ huấn luyện. |
| [Heart_Disease_Features_Final.csv](./22022010-22022012-22022013/Phanloai/Heart_Disease_Features_Final.csv) | Bảng đặc trưng có sẵn. Notebook phân loại ghép các cột `Deep_*` theo `SUBJECT_CODE`. |

## Chuẩn bị môi trường và dữ liệu

Notebook được viết cho Kaggle. Bật GPU khi huấn luyện mô hình 3D và dùng môi trường có PyTorch. Cài các gói còn thiếu trong notebook.

```python
%pip install monai medpy nibabel numpy scipy pandas scikit-learn matplotlib seaborn tqdm
```

Cài thêm `xgboost` và `imbalanced-learn` để chạy notebook phân loại. Một số cell trực quan hóa tùy chọn sử dụng `plotly`, `scikit-image`, `torchinfo` hoặc Graphviz.

Chuẩn bị bộ [M&Ms-2](https://www.ub.edu/mnms-2/) và trỏ `ROOT_DIR` đến thư mục chứa các thư mục bệnh nhân. Khối phân đoạn tìm các cặp ảnh và mặt nạ **short-axis** ở cuối tâm trương (ED) và cuối tâm thu (ES) theo tên sau.

```text
dataset/
└── <SUBJECT_CODE>/
    ├── <SUBJECT_CODE>_SA_ED.nii
    ├── <SUBJECT_CODE>_SA_ED_gt.nii
    ├── <SUBJECT_CODE>_SA_ES.nii
    └── <SUBJECT_CODE>_SA_ES_gt.nii
```

Đổi đường dẫn `/kaggle/input/...` theo bộ dữ liệu đã gắn. Nếu dữ liệu dùng đuôi `.nii.gz`, cập nhật phần dò tên tệp trong notebook. Hai mô hình resize khối ảnh về **128 × 128 × 64** và chuẩn hóa cường độ theo Z-score.

## Chạy phân đoạn

- **SegResNet3D.** Chạy phần `Config` để đọc dữ liệu, chia tập và khởi tạo mô hình. Để dùng trọng số có sẵn, bỏ qua vòng lặp huấn luyện và chỉnh `model_path`, `chk_path` hoặc các đường dẫn checkpoint trong cell cần chạy. Trỏ `LOG_PATH` đến CSV đi kèm nếu muốn vẽ biểu đồ từ lần huấn luyện đã lưu. Giữ `SAVE_DIR` là một thư mục ghi riêng vì cell cấu hình tạo mới `training_log.csv` tại đó.
- **VNet3D.** Trong cell huấn luyện đầu tiên, sửa dòng `if self.augment: you` thành `if self.augment:` trước khi chạy. Huấn luyện để tạo `vnet3d_final_results/best_model.pth` hoặc dùng checkpoint VNet3D bạn đã có. Cập nhật `MODEL_PATH` trong các cell Grad-CAM có đường dẫn Kaggle ghi sẵn.

Hai notebook chia theo **cặp ảnh–mặt nạ** với tỷ lệ 70/15/15. Vì ED và ES được đưa vào danh sách như hai mẫu riêng, hai pha của cùng một bệnh nhân có thể nằm ở các tập khác nhau.

Các cell Grad-CAM tạo bản đồ nhiệt theo lớp hoặc theo tầng. Bản đồ độ bất định của SegResNet3D dùng **Monte Carlo Dropout** và entropy dự đoán. VNet3D đo độ biến động của dự đoán khi thêm nhiễu nhẹ vào ảnh đầu vào.

## Chạy phân loại ba nhóm

Mở [VNet3D-3LOP.ipynb](./22022010-22022012-22022013/Phanloai/VNet3D-3LOP.ipynb) và cập nhật các đường dẫn trong `CONFIG`.

| Biến | Dữ liệu cần cung cấp |
| --- | --- |
| `DATA_ROOT` | Thư mục MRI theo mã bệnh nhân để vẽ ảnh gốc và mặt nạ chồng lớp. |
| `CSV_INFO` | Tệp `dataset_information.csv` của bộ dữ liệu, có các cột `SUBJECT_CODE` và `DISEASE`. |
| `PRED_MASK_DIR` | Thư mục chứa mặt nạ VNet3D theo cặp `<SUBJECT_CODE>_SA_ED_mask.nii` và `<SUBJECT_CODE>_SA_ES_mask.nii`. |
| `SAVED_DEEP_FEATURES` | Đường dẫn đến tệp `Heart_Disease_Features_Final.csv` đi kèm. |

Nếu cần tạo mặt nạ từ đầu, bỏ comment ở cell xuất mặt nạ của notebook phân loại, đặt `MODEL_PATH` thành checkpoint VNet3D và đặt `PRED_MASK_DIR` trỏ đến `OUTPUT_DIR` của cell đó.

Bộ phân loại ghép sáu chỉ số từ mặt nạ, gồm thể tích thất, phân suất tống máu thất trái và khối lượng cơ tim, với các cột `Deep_*`. Notebook chia train/test 80/20 theo tỷ lệ lớp, chuẩn hóa từ tập train và thử áp dụng SMOTE trên tập train. Kết quả là dự đoán **bệnh lý tim trái**, **bệnh lý tim phải** hoặc **bình thường** bằng Soft Voting giữa XGBoost, Random Forest và SVM.

## Kết quả thực nghiệm

Kết quả phân đoạn dưới đây lấy từ output kiểm thử đã lưu, với 108 khối ảnh trong mỗi notebook.

| Mô hình | Dice trung bình | IoU trung bình | HD95 trên lưới voxel |
| --- | ---: | ---: | ---: |
| SegResNet3D | **0,9106** | **0,8397** | **1,70** |
| VNet3D | 0,8885 | 0,8049 | 1,93 |

Notebook tính HD95 với khoảng cách voxel mặc định trên ảnh đã resize. Khi đọc điểm theo từng cấu trúc, đối chiếu mã nhãn trong mặt nạ vì `get_metrics` và một số cell trực quan hóa đang dùng thứ tự RV/MYO khác nhau.

Bộ phân loại ba nhóm đạt **81,13% accuracy** trên **53 ca kiểm thử** trong lần chạy được lưu. Notebook cũng có ma trận nhầm lẫn, đường cong ROC theo lớp và biểu đồ độ quan trọng đặc trưng của Random Forest.

## Tài liệu và nhóm thực hiện

- [Báo cáo đồ án (PDF)](./22022010-22022012-22022013/Deep_Learning_final.pdf)
- [Slide thuyết trình (PowerPoint)](./22022010-22022012-22022013/DeepLearning_Final.pptx)

**Thành viên** Trần Lâm Minh, Nguyễn Đức Trọng và Nguyễn Thị Diễm My.

**Giảng viên hướng dẫn** ThS. Nguyễn Thị Mỹ Nga.
