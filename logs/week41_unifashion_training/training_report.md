# Báo cáo fine-tuning UniFashion (bản tối thiểu)

- Trạng thái: training
- Tiến độ: 9/10 epoch, 81000 cập nhật tối ưu
- Loss huấn luyện gần nhất: 0.4144 (TB 100 bước cuối)
- Monitor loss tốt nhất: 0.3129 (epoch 8)

| epoch | train_loss | monitor_loss |
|---|---|---|
| 1 | 0.3047 | 0.6317 |
| 2 | 0.3572 | 0.6592 |
| 3 | 0.3253 | 0.3232 |
| 4 | 0.2887 | 0.6998 |
| 5 | 0.2653 | 0.7263 |
| 6 | 0.2034 | 0.4618 |
| 7 | 0.1698 | 0.3266 |
| 8 | 0.1337 | 0.3129 |
| 9 | 0.1030 | 0.3480 |

Lưu ý: monitor loss là loss trên tập cố định, KHÔNG phải Recall@K.