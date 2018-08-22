#!/usr/bin/env python
# coding: utf8
from __future__ import unicode_literals
from pandas import read_csv
import re
from pyvi import ViPosTagger,ViTokenizer 
import regex
# source: https://github.com/stopwords/vietnamese-stopwords

# Stop words
# this file is used to save the result of CleanContent
"""
This class is used to clean stop words, special characters, number, ...
in corpus and in sentence

    data : the document need to clean
    STOP_WORDS: list stop word
"""
class DataCleaner:
    def __init__(self, data = None):
        self.data = data
        # stop words set
        self.STOP_WORDS = set("""
a_lô
a_ha
ai
ai_ai
ai_nấy
ai_đó
alô
amen
anh
anh_ấy
ba_bau
ba_bản
ba_cùng
ba_họ
bạn
ba_ngày
ba_ngôi
ba_tăng
bao_giờ
bao_lâu
bao_nả
bay_biến
biết_bao
biết_bao_nhiêu
biết_chắc
biết_chừng_nào
biết_mình
biết_mấy
biết_thế
biết_trước
biết_việc
biết_đâu
biết_đâu_chừng
biết_đâu_đấy
biết_được
buổi
buổi_làm
buổi_mới
buổi_ngày
buổi_sớm
bà
bà_ấy
bài
bài_bác
bài_bỏ
bài_cái
bác
bán_cấp
bán_dạ
bán_thế
bây_bẩy
bây_chừ
bây_giờ
bây_nhiêu
bèn
béng
bên
bên_bị
bên_có
bông
bước
bước_khỏi
bước_tới
bước_đi
bản
bản_bộ
bản_riêng
bản_ý
bất_chợt
bất_giác
bất_kể
bất_nhược
bất_quá
bất_quá_chỉ
bất_thình_lình
bất_tử
bất_đồ
bấy
bấy_chầy
bấy_chừ
bấy_giờ
bấy_lâu
bấy_lâu_nay
bấy_nay
bấy_nhiêu
bập_bà_bập_bõm
bập_bõm
bắt_đầu_từ
bằng
bằng_cứ
bằng_không
bằng_người
bằng_như
bằng_nào
bằng_nấy
bằng_vào
bằng_được
bằng_ấy
bển
bệt
bị
bị_chú
bị_vì
bỏ
bỏ_bà
bỏ_cha
bỏ_cuộc
bỏ_không
bỏ_lại
bỏ_mình
bỏ_mất
bỏ_mẹ
bỏ_nhỏ
bỏ_quá
bỏ_ra
bỏ_riêng
bỏ_việc
bỏ_xa
bỗng
bỗng_chốc
bỗng_dưng
bỗng_không
bỗng_nhưng
bỗng_thấy
bỗng_đâu
bộ
bộ_thuộc
bộ_điều
bội_phần
bớ
bởi
bởi_ai
bởi_chưng
bởi_nhưng
bởi_sao
bởi_thế
bởi_thế_cho_nên
bởi_tại
bởi_đâu
bớt
bức
cao_lâu
cao_ráo
cao_răng
cao_sang
cao_số
cao_thấp
cao_xa
cha
cha_chả
chao_ôi
cho_biết
cho_nhau
cho_rằng
cho_rồi
cho_thấy
cho_tin
cho_tới
cho_tới_khi
cho_về
cho_ăn
cho_đang
cho_được
cho_đến
cho_đến_nỗi
choa
chu_cha
chui_cha
chung
chung_cho
chung_chung
chung_cục
chung_nhau
chung_qui
chung_quy
chung_quy_lại
chung_ái
chuyển_tự
chuyển_đạt
chành_chạnh
chí_chết
chính
chính_bản
chính_giữa
chính_là
chính_thị
chính_điểm
chùn_chùn
chùn_chũn
chú
chú_dẫn
chú_khách
chú_mày
chú_mình
chúng
chúng_mình
chúng_ông
chăn_chắn
chăng
chăng_chắc
chăng_nữa
chơi_họ
chưa_dùng
chưa_dễ
chưa_kể
chưa_từng
chậc
chắc
chắc_dạ
chắc_hẳn
chắc_lòng
chắc_người
chắc_vào
chắc_ăn
chẳng_nữa
chết_nỗi
chết_thật
chết_tiệt
chỉ
chỉ_chính
chỉ_có
chỉ_là
chỉ_tên
chỉn
chị
chị_bộ
chị_ấy
chịu_chưa
chịu_lời
chịu_tốt
chịu_ăn
chọn
chọn_bên
chọn_ra
chốc_chốc
chớ
chớ_chi
chớ_gì
chớ_không
chớ_kể
chớ_như
chợt
chợt_nghe
chợt_nhìn
chủn
chứ
chứ_ai
chứ_còn
chứ_gì
chứ_không
chứ_không_phải
chứ_lại
chứ_lị
chứ_như
chứ_sao
coi_bộ
coi_mòi
con_con
con_dạ
con_nhà
con_tính
cu_cậu
cuối
cuối_điểm
cuốn
cuộc
càng_càng
càng_hay
cá_nhân
các
các_cậu
cách
cách_bức
cách_không
cách_nhau
cách_đều
cái_gì
cái_họ
cái_đã
cái_đó
cái_ấy
câu_hỏi
cây
cây_nước
còn
còn_như
còn_nữa
còn_thời_gian
còn_về
có_ai
có_chuyện
có_chăng
có_chăng_là
có_chứ
có_cơ
có_dễ
có_họ
có_khi
có_ngày
có_người
có_nhiều
có_nhà
có_phải
có_số
có_tháng
có_thế
có_ý
có_ăn
có_điều
có_điều_kiện
có_đáng
có_đâu
có_được
cóc_khô
cô_mình
cô_quả
cô_tăng
cô_ấy
công_nhiên
cùng
cùng_chung
cùng_cực
cùng_nhau
cùng_tuổi
cùng_tột
cùng_với
cùng_ăn
căn
căn_cái
căn_cắt
căn_tính
cũng
cũng_như
cũng_nên
cơ
cơ_chỉ
cơ_chừng
cơ_cùng
cơ_dẫn
cơ_hồ
cơ_mà
cơn
cả_nghe
cả_nghĩ
cả_ngày
cả_người
cả_nhà
cả_năm
cả_thảy
cả_thể
cả_tin
cả_ăn
cả_đến
cảm_thấy
cấp
cấp_số
cấp_trực_tiếp
cần
cần_cấp
cần_gì
cần_số
cật_lực
cật_sức
cổ_lai
cụ_thể
cụ_thể_là
cụ_thể_như
của
của_ngọt
của_tin
cứ
cứ_như
cứ_việc
cứ_điểm
cực_lực
do
do_vì
duy
duy_chỉ
duy_có
dài
dài_lời
dài_ra
dành_dành
dào
dì
dù
dù_cho
dù_dì
dù_gì
dù_rằng
dùng
dùng_cho
dùng_hết
dùng_làm
dùng_đến
dưới
dưới_nước
dạ
dạ_bán
dạ_con
dạ_dài
dạ_dạ
dạ_khách
dần_dà
dần_dần
dầu_sao
dẫn
dẫu
dẫu_mà
dẫu_rằng
dẫu_sao
dễ
dễ_dùng
dễ_gì
dễ_khiến
dễ_nghe
dễ_ngươi
dễ_như_chơi
dễ_sợ
dễ_sử_dụng
dễ_thường
dễ_thấy
dễ_ăn
dễ_đâu
dở_chừng
dữ
dữ_cách
em
em_em
giảm_chính
giảm_thế
giống
giống_người
giờ
giờ_lâu
giờ_đi
giờ_đây
giờ_đến
giữ
giữ_lấy
giữ_ý
giữa_lúc
gây
gây_cho
gây_giống
gây_ra
gây_thêm
gì
gì_gì
gì_đó
gần
gần_bên
gần_ngày
gần_như
gần_xa
gần_đây
gần_đến
gặp_phải
hay_biết
hay_hay
hay_không
hay_làm
hay_nhỉ
hay_nói
hay_sao
hay_tin
hay_đâu
hiểu
hoặc_là
hãy
hãy_còn
hơn
hơn_cả
hơn_hết
hơn_là
hơn_nữa
hơn_trước
hết_chuyện
hết_cả
hết_của
hết_nói
hết_ráo
hết_rồi
hết_ý
họ
họ_gần
họ_xa
hỏi_xin
khi
khi_khác
khi_không
khi_nên
khi_trước
khiến
khoảng_không
khác
khác_gì
khác_khác
khác_nào
khác_xa
khách
khó
khó_biết
khó_chơi
khó_làm
khó_mở
khó_nghe
khó_nghĩ
khó_nói
khó_thấy
không_ai
không_chỉ
không_có
không_có_gì
không_cùng
không_cứ
không_gì
không_hay
không_khỏi
không_kể
không_ngoài
không_nhận
không_những
không_phải_không
không_tính
không_điều_kiện
không_đầy
không_để
khỏi
khỏi_nói
kể
kể_cả
kể_như
kể_tới
kể_từ
loại_từ
luôn
luôn_cả
luôn_tay
là
là_cùng
là_là
là_nhiều
là_phải
là_thế_nào
là_vì
là_ít
làm
làm_bằng
làm_cho
làm_dần_dần
làm_gì
làm_lòng
làm_lại
làm_lấy
làm_mất
làm_ngay
làm_như
làm_nên
làm_ra
làm_riêng
làm_theo
làm_tin
làm_tôi
làm_tăng
làm_tại
làm_tắp_lự
làm_vì
làm_đúng
làm_được
lâu
lâu_các
lâu_lâu
lâu_nay
lâu_ngày
lên_cơn
lên_nước
lên_số
lên_đến
lòng
lòng_không
lúc
lúc_khác
lúc_lâu
lúc_nào
lúc_này
lúc_sáng
lúc_trước
lúc_đi
lúc_đó
lúc_đến
lúc_ấy
lượng
lượng_cả
lượng_số
lượng_từ
lại
lại_bộ
lại_cái
lại_còn
lại_giống
lại_làm
lại_người
lại_nói
lại_nữa
lại_quả
lại_thôi
lại_ăn
lại_đây
lấy
lấy_có
lấy_cả
lấy_giống
lấy_làm
lấy_lý_do
lấy_lại
lấy_ra
lấy_ráo
lấy_sau
lấy_số
lấy_thế
lấy_xuống
lấy_được
lấy_để
lần_khác
lần_lần
lần_sang
lần_theo
lần_tìm
lớn
lớn_lên
lớn_nhỏ
lời
lời_chú
lời_nói
mang
mang_lại
mang_mang
mang_nặng
mang_về
mà
mà_cả
mà_không
mà_lại
mà_thôi
mà_vẫn
mình
mạnh
mất
mất_còn
mọi
mọi_giờ
mọi_khi
mọi_lúc
mọi_nơi
mọi_sự
mọi_thứ
mọi_việc
mối
mỗi
mỗi_lúc
mỗi_lần
mỗi_một
mỗi_ngày
mỗi_người
một
một_cách
một_cơn
một_khi
một_lúc
một_số
một_vài
mới
mới_hay
mới_rồi
mới_đây
mở_mang
mở_nước
mở_ra
mợ
nay
ngay
ngay_bây_giờ
ngay_cả
ngay_khi
ngay_khi_đến
ngay_lúc
ngay_lúc_này
ngay_lập_tức
ngay_thật
ngay_tức_khắc
ngay_tức_thì
ngay_từ
nghe
nghe_chừng
nghe_hiểu
nghe_không
nghe_lại
nghe_nhìn
nghe_như
nghe_nói
nghe_ra
nghe_rõ
nghe_thấy
nghe_tin
nghe_trực_tiếp
nghe_đâu
nghe_đâu_như
nghe_được
nghen
nghiễm_nhiên
nghĩ
nghĩ_lại
nghĩ_ra
nghĩ_tới
nghĩ_xa
nghĩ_đến
nghỉm
ngoài
ngoài_này
ngoài_ra
ngoài_xa
ngoải
nguồn
ngày
ngày_càng
ngày_cấp
ngày_giờ
ngày_ngày
ngày_nào
ngày_này
ngày_nọ
ngày_qua
ngày_rày
ngày_tháng
ngày_xưa
ngày_xửa
ngày_đến
ngày_ấy
ngôi
ngôi_nhà
ngôi_thứ
ngõ_hầu
ngăn_ngắt
ngươi
người
người_hỏi
người_khác
người_khách
người_mình
người_nghe
người_người
người_nhận
ngọn
ngọn_nguồn
ngọt
ngồi
ngồi_bệt
ngồi_không
ngồi_sau
ngồi_trệt
ngộ_nhỡ
nhanh
nhanh_lên
nhanh_tay
nhau
nhiên_hậu
nhiều_ít
nhiệt_liệt
nhung_nhăng
nhà
nhà_chung
nhà_khó
nhà_làm
nhà_ngoài
nhà_ngươi
nhà_tôi
nhà_việc
nhân_dịp
nhân_tiện
nhé
nhìn
nhìn_chung
nhìn_lại
nhìn_nhận
nhìn_theo
nhìn_thấy
nhìn_xuống
nhón_nhén
như
như_ai
như_chơi
như_không
như_là
như_nhau
như_quả
như_sau
như_thường
như_thế
như_thể
như_trên
như_trước
như_tuồng
như_vậy
như_ý
nhưng
nhưng_mà
nhược_bằng
nhất
nhất_loạt
nhất_luật
nhất_là
nhất_mực
nhất_nhất
nhất_quyết
nhất_sinh
nhất_thiết
nhất_thì
nhất_tâm
nhất_tề
nhất_đán
nhận
nhận_biết
nhận_họ
nhận_làm
nhận_nhau
nhận_ra
nhận_thấy
nhận_việc
nhận_được
nhằm
nhằm_khi
nhằm_lúc
nhằm_vào
nhằm_để
nhỉ
nhỏ
nhỏ_người
nhớ
nhớ_bập_bõm
nhớ_lại
nhớ_lấy
nhớ_ra
nhờ
nhờ_chuyển
nhờ_có
nhờ_nhờ
nhờ_đó
nhỡ_ra
những
những_ai
những_khi
những_là
những_lúc
những_muốn
những_như
nào
nào_cũng
nào_hay
nào_là
nào_phải
nào_đâu
nào_đó
này
này_nọ
nên
nên_chi
nên_chăng
nên_làm
nên_người
nên_tránh
nó
nóc
nói
nói_bông
nói_chung
nói_khó
nói_là
nói_lên
nói_lại
nói_nhỏ
nói_phải
nói_qua
nói_ra
nói_riêng
nói_rõ
nói_thêm
nói_thật
nói_toẹt
nói_trước
nói_tốt
nói_với
nói_xa
nói_ý
nói_đến
nói_đủ
năm
năm_tháng
nơi
nơi_nơi
nước
nước_bài
nước_cùng
nước_lên
nước_nặng
nước_quả
nước_xuống
nước_ăn
nước_đến
nấy
nặng
nặng_căn
nặng_mình
nặng_về
nếu
nếu_có
nếu_cần
nếu_không
nếu_mà
nếu_như
nếu_thế
nếu_vậy
nếu_được
nền
nọ
nớ
nức_nở
nữa
nữa_khi
nữa_là
nữa_rồi
oai_oái
oái
pho
phè
phè_phè
phía
phía_bên
phía_bạn
phía_dưới
phía_sau
phía_trong
phía_trên
phía_trước
phóc
phót
phăn_phắt
phương_chi
phải
phải_biết
phải_chi
phải_chăng
phải_cách
phải_cái
phải_giờ
phải_khi
phải_không
phải_lại
phải_lời
phải_người
phải_như
phải_rồi
phải_tay
phần
phần_lớn
phần_nhiều
phần_nào
phần_sau
phần_việc
phắt
phỉ_phui
phỏng
phỏng_như
phỏng_nước
phỏng_theo
phỏng_tính
phốc
phụt
phứt
qua
qua_chuyện
qua_khỏi
qua_lại
qua_lần
qua_ngày
qua_tay
qua_thì
qua_đi
quan_trọng_vấn_đề
quay
quay_bước
quay_lại
quay_số
quay_đi
quá
quá_bộ
quá_giờ
quá_lời
quá_mức
quá_nhiều
quá_tay
quá_thì
quá_tin
quá_tuổi
quá_đáng
quá_ư
quả
quả_là
quả_thật
quả_thế
quả_vậy
ra
ra_bài
ra_bộ
ra_chơi
ra_gì
ra_lại
ra_lời
ra_ngôi
ra_người
ra_sao
ra_tay
ra_vào
ra_ý
ra_điều
ra_đây
ren_rén
riu_ríu
riêng
riêng_từng
riệt
rày
ráo
ráo_cả
ráo_nước
ráo_trọi
rén
rén_bước
rích
rón_rén
rõ
rõ_là
rõ_thật
rút_cục
răng
răng_răng
rất
rất_lâu
rằng
rằng_là
rốt_cuộc
rốt_cục
rồi
rồi_nữa
rồi_ra
rồi_sao
rồi_sau
rồi_tay
rồi_thì
rồi_xem
rồi_đây
rứa
sa_sả
sang
sang_năm
sang_sáng
sang_tay
sao
sao_bản
sao_bằng
sao_cho
sao_vậy
sao_đang
sau
sau_chót
sau_cuối
sau_cùng
sau_hết
sau_này
sau_nữa
sau_sau
sau_đây
sau_đó
so
so_với
song_le
suýt
suýt_nữa
sáng
sáng_ngày
sáng_rõ
sáng_thế
sáng_ý
sì
sì_sì
sất
sắp
sắp_đặt
sẽ
sẽ_biết
sẽ_hay
số
số_cho_biết
số_cụ_thể
số_loại
số_là
số_người
số_phần
số_thiếu
sốt_sột
sớm
sớm_ngày
sở_dĩ
sự
sự_thế
tanh
tanh_tanh
tay
tay_quay
tha_hồ
tha_hồ_chơi
tha_hồ_ăn
than_ôi
thanh
thanh_ba
thanh_chuyển
thanh_không
thanh_thanh
thanh_tính
thanh_điều_kiện
thanh_điểm
thay_đổi
thay_đổi_tình_trạng
theo
theo_bước
theo_như
theo_tin
thi_thoảng
thiếu
thiếu_gì
thiếu_điểm
thoạt
thoạt_nghe
thoạt_nhiên
thoắt
thuần
thuần_ái
thuộc
thuộc_bài
thuộc_cách
thuộc_lại
thuộc_từ
thà
thà_là
thà_rằng
thành_ra
thành_thử
thái_quá
tháng
tháng_ngày
tháng_năm
tháng_tháng
thêm
thêm_chuyện
thêm_giờ
thêm_vào
thì
thì_giờ
thì_là
thì_phải
thì_ra
thì_thôi
thình_lình
thích
thích_cứ
thích_thuộc
thích_tự
thích_ý
thím
thôi
thôi_việc
thúng_thắng
thương_ôi
thường
thường_bị
thường_hay
thường_khi
thường_số
thường_sự
thường_thôi
thường_thường
thường_tính
thường_tại
thường_xuất_hiện
thường_đến
thảo_hèn
thảo_nào
thấp
thấp_cơ
thấp_thỏm
thấp_xuống
thấy
thấy_tháng
thẩy
thậm
thậm_chí
thậm_cấp
thậm_từ
thật
thật_chắc
thật_là
thật_lực
thật_quả
thật_ra
thật_sự
thật_thà
thật_tốt
thật_vậy
thế
thêm
bớt
thế_chuẩn_bị
thế_là
thế_lại
thế_mà
thế_nào
thế_nên
thế_ra
thế_sự
thế_thì
thế_thôi
thế_thường
thế_thế
thế_à
thế_đó
thếch
thỉnh_thoảng
thỏm
thốc
thốc_tháo
thốt
thốt_nhiên
thốt_nói
thốt_thôi
thộc
thời_gian
thời_gian_sử_dụng
thời_gian_tính
thời_điểm
thục_mạng
thứ
thứ_bản
thứ_đến
thửa
thực_hiện
thực_hiện_đúng
thực_ra
thực_sự
thực_tế
thực_vậy
tin
tin_thêm
tin_vào
tiếp_theo
tiếp_tục
tiếp_đó
tiện_thể
toà
toé_khói
toẹt
trong_khi
trong_lúc
trong_mình
trong_ngoài
trong_này
trong_số
trong_vùng
trong_đó
trong_ấy
tránh
tránh_khỏi
tránh_ra
tránh_tình_trạng
tránh_xa
trên
trên_bộ
trên_dưới
trước
trước_hết
trước_khi
trước_kia
trước_nay
trước_ngày
trước_nhất
trước_sau
trước_tiên
trước_tuổi
trước_đây
trước_đó
trả
trả_của
trả_lại
trả_ngay
trả_trước
trếu_tráo
trển
trệt
trệu_trạo
trỏng
trời_đất_ơi
trở_thành
trừ_phi
trực_tiếp
trực_tiếp_làm
tuy
tuy_có
tuy_là
tuy_nhiên
tuy_rằng
tuy_thế
tuy_vậy
tuy_đã
tuyệt_nhiên
tuần_tự
tuốt_luốt
tuốt_tuồn_tuột
tuốt_tuột
tuổi
tuổi_cả
tuổi_tôi
tà_tà
tên
tên_chính
tên_cái
tên_họ
tên_tự
tênh
tênh_tênh
tìm_bạn
tìm_cách
tìm_ra
tìm_việc
tính
tính_cách
tính_căn
tính_người
tính_phỏng
tính_từ
tít_mù
tò_te
tôi
tôi_con
tông_tốc
tù_tì
tăm_tắp
tăng_chúng
tăng_cấp
tăng_giảm
tăng_thêm
tại
tại_lòng
tại_nơi
tại_tôi
tại_đâu
tại_đây
tại_đó
tạo
tạo_cơ_hội
tạo_nên
tạo_ra
tạo_ý
tạo_điều_kiện
tấm
tấm_bản
tấm_các
tấn
tấn_tới
tất_cả_bao_nhiêu
tất_thảy
tất_tần_tật
tất_tật
tắp
tắp_lự
tắp_tắp
tọt
tỏ_ra
tỏ_vẻ
tốc_tả
tối_ư
tốt
tốt_bạn
tốt_bộ
tốt_hơn
tốt_mối
tốt_ngày
tột
tột_cùng
tới_gần
tới_nơi
tới_thì
tức_thì
tức_tốc
từ
từ_căn
từ_giờ
từ_khi
từ_loại
từ_nay
từ_thế
từ_tính
từ_tại
từ_từ
từ_ái
từ_điều
từ_đó
từ_ấy
từng
từng_cái
từng_giờ
từng_nhà
từng_phần
từng_thời_gian
từng_đơn_vị
từng_ấy
tự
tự_cao
tự_khi
tự_lượng
tự_tính
tự_tạo
tự_vì
tự_ý
tự_ăn
tựu_trung
veo
veo_veo
việc
việc_gì
vung_thiên_địa
vung_tàn_tán
vung_tán_tàn
và
vài
vài_ba
vài_người
vài_nhà
vài_nơi
vài_tên
vài_điều
vào
vào_gặp
vào_khoảng
vào_lúc
vào_vùng
vào_đến
vâng
vâng_chịu
vâng_dạ
vâng_vâng
vâng_ý
vèo
vèo_vèo
vì
vì_chưng
vì_rằng
vì_sao
vì_thế
vì_vậy
ví_bằng
ví_dù
ví_phỏng
ví_thử
vô_hình_trung
vô_kể
vô_luận
vô_vàn
vùng
vùng_lên
vùng_nước
văng_tê
vượt_khỏi
vượt_quá
vạn_nhất
vả_chăng
vả_lại
vẫn
vẫn_thế
vậy
vậy_là
vậy_mà
vậy_nên
vậy_ra
vậy_thì
vậy_ư
về
về_không
về_nước
về_phần
về_sau
về_tay
vị_tất
vốn_dĩ
với
với_lại
với_nhau
vở
vụt
vừa
vừa_khi
vừa_lúc
vừa_mới
vừa_qua
vừa_rồi
vừa_vừa
xa
xa_cách
xa_gần
xa_nhà
xa_tanh
xa_tắp
xa_xa
xa_xả
xem
xem_lại
xem_ra
xem_số
xin
xin_gặp
xin_vâng
xiết_bao
xon_xón
xoành_xoạch
xoét
xoẳn
xoẹt
xuất_kì_bất_ý
xuất_kỳ_bất_ý
xuể
xăm_xúi
xăm_xăm
xăm_xắm
xềnh_xệch
xệp
à
à_này
à_ơi
ào
ào_vào
ào_ào
á
á_à
ái
ái_chà
ái_dà
áng
áng_như
âu_là
ít
ít_biết
ít_có
ít_hơn
ít_khi
ít_lâu
ít_nhiều
ít_nhất
ít_nữa
ít_quá
ít_ra
ít_thôi
ít_thấy
ô_hay
ô_hô
ô_kê
ô_kìa
ôi_chao
ôi_thôi
ông
ông_nhỏ
ông_tạo
ông_từ
ông_ấy
ông_ổng
úi
úi_chà
úi_dào
ý
ý_chừng
ý_da
ý_hoặc
ăn
ăn_chung
ăn_chắc
ăn_chịu
ăn_cuộc
ăn_hết
ăn_hỏi
ăn_làm
ăn_người
ăn_ngồi
ăn_quá
ăn_riêng
ăn_sáng
ăn_tay
ăn_trên
ăn_về
đang
đang_tay
đang_thì
điều
điều_gì
điều_kiện
điểm
điểm_chính
điểm_gặp
điểm_đầu_tiên
đành_đạch
đáng
đáng_kể
đáng_lí
đáng_lý
đáng_lẽ
đáng_số
đánh_giá
đánh_đùng
đáo_để
đâu
đâu_có
đâu_cũng
đâu_như
đâu_nào
đâu_phải
đâu_đâu
đâu_đây
đâu_đó
đây
đây_này
đây_rồi
đây_đó
đã
đã_hay
đã_không
đã_là
đã_lâu
đã_thế
đã_vậy
đã_đủ
đó
đó_đây
đúng
đúng_ngày
đúng_ra
đúng_tuổi
đúng_với
đưa
đưa_cho
đưa_chuyện
đưa_em
đưa_ra
đưa_tay
đưa_tin
đưa_tới
đưa_vào
đưa_về
đưa_xuống
đưa_đến
được
được_cái
được_lời
được_nước
được_tin
đại_loại
đại_nhân
đại_phàm
đại_để
đạt
đầy
đầy_năm
đầy_phè
đầy_tuổi
đặc_biệt
đặt
đặt_làm
đặt_mình
đặt_mức
đặt_ra
đặt_trước
đặt_để
đến
đến_bao_giờ
đến_cùng
đến_cùng_cực
đến_cả
đến_giờ
đến_gần
đến_hay
đến_khi
đến_lúc
đến_lời
đến_nay
đến_ngày
đến_nơi
đến_nỗi
đến_thì
đến_thế
đến_tuổi
đến_xem
đến_điều
đến_đâu
đều
đều_bước
đều_nhau
đều_đều
để
để_cho
để_giống
để_không
để_lòng
để_lại
để_mà
để_phần
để_được
để_đến_nỗi
đối_với
đồng_thời
đủ
đủ_dùng
đủ_nơi
đủ_số
đủ_điều
đủ_điểm
ơ
ơ_hay
ơ_kìa
ơi
ơi_là
ư
ạ
ạ_ơi
ấy
ấy_là
ầu_ơ
ắt
ắt_hẳn
ắt_là
ắt_phải
ắt_thật
ối_dào
ối_giời
ối_giời_ơi
ồ
ồ_ồ
ổng
ớ
ớ_này
ờ
ờ_ờ
ở
ở_lại
ở_như
ở_nhờ
ở_năm
ở_trên
ở_vào
ở_đây
ở_đó
ở_được
ủa
ứ_hự
ứ_ừ
ừ
ừ_nhé
ừ_thì
ừ_ào
ừ_ừ
ử
""".split('\n'))
    """
    This function check a word is stop word or not!!!
    input: A word
    ouput:  True if word is a stop word
            False if word isn't a stop word
    """
    def is_stop_word(self, word):
        return word in self.STOP_WORDS
    
    """
    This function replace special characters by other character
    and replace acronym by the fully word.
    input: document
    output: document was replaced special characters
    """
    def execute_special_character(self, text):
        text = regex.sub(' vn-index ', ' vnindex ', text)
        text = regex.sub('vn-index ', 'vnindex ', text)
        text = regex.sub(' vn-index', ' vnindex', text)
        text = regex.sub(' cp ', ' cổ phiếu ', text)
        text = regex.sub('cp ', 'cổ phiếu ', text)
        text = regex.sub(' cp', ' cổ phiếu', text)
        # this file include financial symbols
        symbol_arr = []
        with open ('data/stockslist.txt',encoding = 'utf-8') as acro_file:
            lines = acro_file.readlines()
            for line in lines:
                symboli = line.rstrip().split(',')
                symbol_arr.append(' '+ symboli[0].lower() + ' ')
        # use regular expression to replace special characer and acronym
        text = regex.sub("(?s)<ref>.+?</ref>", "", text) # remove reference links
        text = regex.sub("(?s)<[^>]+>", "", text) # remove html tags
        text = regex.sub("&[a-z]+;", "", text) # remove html entities
        text = regex.sub("(?s){{.+?}}", "", text) # remove markup tags
        text = regex.sub("(?s){.+?}", "", text) # remove markup tags
        text = regex.sub("(?s)\[\[([^]]+\|)", "", text) # remove link target strings
        text = regex.sub("(?s)\[\[([^]]+\:.+?]])", "", text) # remove media links
        text = regex.sub("\S*\d\S*", "", text)  # remove word contain string and number
        text = regex.sub("[0-9]{2}/[0-9]{2}/[0-9]{4}", "", text) # remove media links
        text = regex.sub("[-+]?[0-9]*\.?[0-9]*", "", text)  # remove float number
        text = regex.sub("[']{5}", "", text) # remove italic+bold symbols
        text = regex.sub("[']{3}", "", text) # remove bold symbols
        text = regex.sub("[']{2}", "", text) # remove italic symbols
        text = regex.sub("\'"," ",text)  # remove ' character
        text = regex.sub(","," ",text)  # remove , character
        text = regex.sub("\["," ",text)  # remove [ character
        text = regex.sub("\]"," ",text)  # remove [ character
        text = regex.sub('[@!#$%^&;*()<–>?/\"“”,|0-9}{~:]',' ',text)
        for i in range(len(symbol_arr)):
            text = regex.sub(r'%s' % symbol_arr[i], '  ', text)
        acronym_arr = []
        with open ('data/acronym.txt',encoding = 'utf-8') as acro_file:
            lines = acro_file.readlines()
            for line in lines:
                acroi = line.rstrip().split(',')
                acronym_arr.append(acroi)
        # print (acronym_arr)
        for i in range(len(acronym_arr)):
            text = regex.sub(r'%s' % acronym_arr[i][0], r'%s' % acronym_arr[i][1], text)
        return text
    
    """
    This function clean special character and stop words
    in sentence and corpus
    input: self.data -- corpus or sentence need to clean
    output: corpus and sentence was cleaned
    """
    def clean_content(self):
        # acronym word
        
        self.data = self.execute_special_character(self.data)
        sentences = self.data.split('\n')
        new_sentences = []
        for sentence in sentences:
            part_of_sentence = sentence.split('  ')
            
            for part in part_of_sentence:
                # file.write(part + '\n')
                new_sentences.append(part)
        # print ("check")
        all_words = []
        all_sentences_split = []
        for sentence in new_sentences:
            words = ViPosTagger.postagging(ViTokenizer.tokenize(sentence))[0]
            # file.write('\n')
            sentencei = []
            for i , word in enumerate(words):
                if(self.is_stop_word(word) == False):
                    all_words.append(word)
                    sentencei.append(word)
                    
                    # # print (i)
                    # if (i == len(words)-1):
                    #     file.write('\n')
                    # else:
                    #     file.write(word + ' ')
            all_sentences_split.append(sentencei)
        all_words = set(all_words)
        return all_words, all_sentences_split
    def separate_sentence (self):
        # this file include financial symbols
        symbol_arr = []
        with open ('data/stockslist.txt',encoding = 'utf-8') as acro_file:
            lines = acro_file.readlines()
            for line in lines:
                symboli = line.rstrip().split(',')
                symbol_arr.append(symboli[0].lower())
        self.data = self.execute_special_character(self.data)
        sentences = self.data.split('\n')
        new_sentences = []
        for sentence in sentences:
            part_of_sentence = sentence.split('  ')
            
            for part in part_of_sentence:
                # file.write(part + '\n')
                new_sentences.append(part)
        all_words = []
        for sentence in new_sentences:
            words = ViPosTagger.postagging(ViTokenizer.tokenize(sentence))[0]
            for i , word in enumerate(words):
                if(self.is_stop_word(word) == False):
                    if word not in symbol_arr:
                        all_words.append(word)
        return all_words
    def remove_stopword_sent(self,sent):
        s = ViPosTagger.postagging(ViTokenizer.tokenize(sent))[0]
        for i in range(len(s)):
            if self.is_stop_word(s[i]):
                print("S[i]",s[i])
                s[i] = ""
        new_sent = ""
        for i in range(len(s)):
            new_sent += s[i]+" "
        return new_sent