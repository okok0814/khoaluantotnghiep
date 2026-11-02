"""Combine verified training completion, checkpoint selection and model comparison."""
import argparse
import json
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.fashioniq_completion import read_result, candidate, choose_original, require


def generate(inputs, output, final_evaluation=None, attention_evaluation=None):
    inputs, output = Path(inputs), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    evidence = json.loads((inputs / 'training_verification.json').read_text(encoding='utf-8'))
    require(evidence['training_completed'] and evidence['optimizer_steps'] == 90000, 'Training not verified')
    data = inputs / 'evaluation_data'
    baseline = read_result(inputs / 'baselines', 'fashionclip_additive', data)
    methods = ('unifashion_image_caption_score', 'unifashion_image_score')
    base = [read_result(inputs / 'base', m, data, 'original', 60000) for m in methods]
    final = [read_result(final_evaluation, m, data, 'original', 90000) for m in methods] if final_evaluation else None
    attention = [read_result(attention_evaluation, m, data, 'attention') for m in methods] if attention_evaluation else None
    for r in base + (final or []):
        require(r['provenance']['training_fingerprint'] == evidence['fingerprint'], 'Wrong original run')
    a = candidate('step60000', base[0])
    b = candidate('step90000', final[0]) if final else None
    selected = choose_original(a, b)
    selection = {'criterion': 'mean full-validation macro Recall@10 and Recall@50',
                 'scoring_method': methods[0], 'tie_rule': 'retain step60000',
                 'selected': selected, 'candidates': [a] + ([b] if b else []),
                 'final_checkpoint_evaluated': final is not None,
                 'scope': 'step60000 versus step90000 of the original run; not every epoch',
                 'attention_fusion_is_separate_model': True}
    (output / 'checkpoint_selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    pd.DataFrame([{k: c[k] for k in ('name', 'optimizer_steps', 'Recall@10', 'Recall@50', 'mean_R', 'checkpoint_sha256')}
                  for c in selection['candidates']]).to_csv(output / 'checkpoint_comparison.csv', index=False)
    rows = []
    measured = [('FashionCLIP additive', baseline), ('UniFashion step60000 + caption', base[0]),
                ('UniFashion step60000 no caption score', base[1])]
    if final:
        measured += [('UniFashion step90000 + caption', final[0]), ('UniFashion step90000 no caption score', final[1])]
    if attention:
        measured += [('UniFashion + attention-fusion + caption', attention[0]),
                     ('UniFashion + attention-fusion no caption score', attention[1])]
    for label, result in measured:
        rows += [{'model': label, 'split': 'val', 'method': result['method'], **r} for r in result['summary']]
    pd.DataFrame(rows).to_csv(output / 'model_comparison.csv', index=False)
    copy = output / 'checkpoint_copy.json'
    receipt = json.loads(copy.read_text(encoding='utf-8')) if copy.exists() else {}
    copy_ok = (receipt.get('checkpoint_sha256') == selected['checkpoint_sha256']
               and receipt.get('copy_sha256_verified') is True
               and receipt.get('cpu_deserialization_verified') is True)
    state = {'original_training_completed': True, 'original_final_evaluated': final is not None,
             'selected_checkpoint_copy_receipt_valid': copy_ok,
             'attention_fusion_evaluated': attention is not None,
             'comparison_split': 'val', 'official_test_recall_available': False,
             'test_blocker': '6118 public test queries have no target labels in the supplied data',
             'last_week_task_completed': final is not None and copy_ok,
             'three_model_validation_comparison_completed': final is not None and attention is not None,
             'original_test_task_completed': False}
    (output / 'status.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    epochs = pd.read_csv(inputs / 'training_epochs.csv')
    macro = lambda r: next(s for s in r['summary'] if s['category'] == 'MACRO')
    baseline_macro = macro(baseline)
    fmt = lambda value: f'{value:.2f}'.replace('.', ',')
    imported = output / 'evidence_import.json'
    imported = json.loads(imported.read_text(encoding='utf-8')) if imported.exists() else None
    lines = ['# Báo cáo gộp tuần 43–44: hoàn tất UniFashion, lưu checkpoint và so sánh Recall', '',
        'Nhiệm vụ: (1) hoàn tất huấn luyện và lưu checkpoint tốt nhất; '
        '(2) so sánh baseline additive, UniFashion fine-tune và attention-fusion bằng Recall@10/@50.', '',
        '**Phạm vi hiện tại là validation, không phải test chính thức.** '
        'Dữ liệu local có 6.118 test query nhưng toàn bộ target bị ẩn; chưa thể tính test Recall. '
        'Cần thống nhất phạm vi với giảng viên hoặc có cơ chế đánh giá test chính thức. '
        'Không đổi tên validation thành test.', '',
        '| Hạng mục | Trạng thái |', '|---|---|',
        '| Hoàn tất huấn luyện mô hình gốc | Hoàn tất 10 epoch / 90.000 update |',
        '| Lưu checkpoint tốt nhất trong các ứng viên đã đánh giá | '
        + ('Hoàn tất, đã nhận receipt copy/hash/CPU reload |' if state['last_week_task_completed'] else 'Chưa hoàn tất |'),
        '| Baseline additive và UniFashion trên validation | '
        + ('Đã kiểm tra đủ 6.016 query, gồm checkpoint cuối |' if final else 'Chờ checkpoint cuối |'),
        '| Attention-fusion trên cùng giao thức | ' + ('Đã có kết quả |' if attention else 'Chưa có kết quả |'),
        '| So sánh đủ ba mô hình trên test chính thức | Chưa hoàn tất; thiếu nhãn test |',
        '', '## 1. Minh chứng hoàn tất huấn luyện', '',
        'ZIP do người chạy cung cấp xác nhận run gốc `full_train_fp32_b2_seed42` hoàn tất '
        '**10/10 epoch, 90.000 optimizer update, 180.000 lượt ví dụ**. '
        'Mỗi epoch có 18.000 query, batch 2, 9.000 update và không bỏ mẫu. '
        'Đã kiểm tra thứ tự step/batch, các giá trị hữu hạn, tính lại loss từng epoch và đối chiếu '
        'fingerprint với run gốc đã đánh giá tại step 60.000.', '',
        'Cấu hình: FP32, seed 42, AdamW max LR 2e-5, OneCycle 10 epoch, EVA-G đóng băng; '
        'Tesla T4, PyTorch 2.6.0+cu124. Monitor là 96 query cố định, khác full validation 6.016 query.', '',
        '| Epoch | Step | Train loss | Monitor loss |', '|---:|---:|---:|---:|']
    lines += [f"| {int(r.epoch)} | {int(r.step)} | {r.train_loss:.6f} | {r.monitor_loss:.6f} |"
              for r in epochs.itertuples()]
    plot = output / 'training_evidence/training_curves.png'
    if plot.exists():
        lines += ['', '![Đường cong training từ ZIP đã nhận](training_evidence/training_curves.png)']
    elif (output.parent / 'training_training_curves.png').exists():
        lines += ['', '![Đường cong training từ run gốc](../training_training_curves.png)']
    lines += ['', f"Monitor loss thấp nhất: **{evidence['best_monitor_loss']:.6f} ở epoch {evidence['best_monitor_epoch']}**. "
              '`best_monitor.pt` được chọn theo loss; không suy ra đây là checkpoint tốt nhất theo Recall. '
              'Train loss giảm so với giai đoạn đầu nhưng tăng nhẹ từ epoch 9 sang epoch 10; '
              'chưa có cơ sở kết luận hội tụ hay chất lượng truy xuất của checkpoint cuối từ loss.', '',
              f"Tổng thời gian các update được ghi là {evidence['recorded_update_hours']:.2f} giờ; "
              'không phải tổng GPU/wall-clock cost vì chưa gồm setup, validation, lưu checkpoint và các bước chạy lại. '
              'XML trong ZIP training ghi nhận 16 tests đã đạt tại môi trường Colab; đây là kết quả kiểm thử được xuất, '
              'không phải một lượt GPU test mới tại máy local.', '', '## 2. Lựa chọn checkpoint theo Recall', '',
              '| Checkpoint gốc | Macro R@10 (%) | Macro R@50 (%) | Mean R (%) |', '|---|---:|---:|---:|']
    for c in selection['candidates']:
        lines.append(f"| {c['name']} | {c['Recall@10']:.2f} | {c['Recall@50']:.2f} | {c['mean_R']:.2f} |")
    if not final:
        lines.append('| Step 90000 | Chưa đánh giá | Chưa đánh giá | Chưa đánh giá |')
    lines += ['', f"Checkpoint được chọn trong các ứng viên đã đo: **{selected['name']}**. "
              'Tiêu chí cố định: mean macro R@10/R@50 với score caption gallery; hòa thì giữ step60000. '
              'Không đưa attention-fusion vào cùng quyết định vì đó là mô hình khác.', '',
              ('Đã có receipt xác minh hash bản sao và nạp lại checkpoint trên CPU. '
               'Checkpoint/source và hash được lưu trong `checkpoint_copy.json`.' if copy_ok else
               '**Chưa có receipt hợp lệ chứng minh lưu/nạp lại bản best riêng.** '
               'ZIP training không chứa trọng số; việc ghi `Saved checkpoint` trong log không thay thế kiểm tra file.')]
    if final:
        lines += ['', f"Step90000 so với step60000: R@10 {b['Recall@10'] - a['Recall@10']:+.2f}, "
                  f"R@50 {b['Recall@50'] - a['Recall@50']:+.2f}, mean R {b['mean_R'] - a['mean_R']:+.2f} "
                  'điểm phần trăm. Đây là so sánh hai snapshot của cùng run, không phải kết quả của trial tuning.', '',
                  '**Phạm vi lựa chọn:** step60000 và step90000; chưa đánh giá Recall ở mọi epoch, '
                  'nên không khẳng định checkpoint này tốt nhất trong toàn bộ quá trình huấn luyện.']
    if copy_ok and receipt.get('checkpoint'):
        lines += ['', 'Bản checkpoint được receipt ghi nhận trên Drive:', '',
                  f"`{receipt['checkpoint']}`", '', f"SHA-256: `{receipt['checkpoint_sha256']}`.", '',
                  'Receipt xác nhận bản sao có hash trùng nguồn và deserialize thành công trên CPU ở Colab. '
                  'ZIP chỉ chứa minh chứng, không chứa weights; không thực hiện nạp weights hay inference GPU lại tại local.']
    lines += ['', '## 3. So sánh các mô hình trên full validation', '',
              'Cùng 6.016 query: dress 2.017/3.817 gallery, shirt 2.038/6.346, toptee 1.961/5.373. '
              'Loại reference khỏi ranking, cùng raw text và gallery caption cho các biến thể UniFashion; '
              'macro là trung bình đều ba category. Baseline được tái sử dụng từ kết quả đã đo và kiểm tra '
              'lại rank; không tuyên bố chạy inference baseline mới.', '',
              '| Mô hình / checkpoint | Category | R@10 (%) | R@50 (%) |', '|---|---|---:|---:|']
    for r in rows:
        if r['category'] != 'MICRO':
            lines.append(f"| {r['model']} | {r['category']} | {r['Recall@10']:.2f} | {r['Recall@50']:.2f} |")
    if final:
        final_macro, no_caption = macro(final[0]), macro(final[1])
        lines += ['', f"Checkpoint cuối có caption đạt macro {fmt(final_macro['Recall@10'])}%/"
                  f"{fmt(final_macro['Recall@50'])}%, cao hơn baseline additive "
                  f"{fmt(final_macro['Recall@10'] - baseline_macro['Recall@10'])}/"
                  f"{fmt(final_macro['Recall@50'] - baseline_macro['Recall@50'])} điểm phần trăm. "
                  f"Bỏ score caption gallery còn {fmt(no_caption['Recall@10'])}%/{fmt(no_caption['Recall@50'])}%; "
                  f"chênh lệch do thêm score caption là {fmt(final_macro['Recall@10'] - no_caption['Recall@10'])}/"
                  f"{fmt(final_macro['Recall@50'] - no_caption['Recall@50'])} điểm phần trăm.", '',
                  '**Giải thích ablation:** cả hai biến thể UniFashion vẫn dùng ảnh reference và text chỉnh sửa '
                  'làm query; “no caption score” chỉ bỏ thành phần score caption của gallery, không phải image-only query.']
        seconds = final[0]['provenance'].get('evaluation_seconds')
        if seconds is not None:
            lines += ['', f'Lượt evaluation ghi nhận {seconds / 60:.2f} phút cho cả hai scoring; '
                      'không cộng thời gian setup, kiểm tra và sao chép checkpoint ngoài lượt này.']
    if (output / 'recall_comparison.png').exists():
        lines += ['', '![So sánh macro Recall trên validation](recall_comparison.png)']
    if not attention:
        lines += ['', '**Attention-fusion: chưa có kết quả cùng giao thức.** '
                  'Theo cập nhật mới nhất của người chạy, teammate chưa hoàn tất run dự định dùng. '
                  'Không lấy điểm của gate FashionCLIP độc lập hoặc notebook cũ để điền vào hàng '
                  'UniFashion + attention-fusion. Bổ sung checkpoint và full-validation JSON/CSV khi sẵn sàng.']
    lines += ['', '## 4. Giới hạn và phần còn lại', '',
              '- Huấn luyện gốc đã hoàn tất; không cần train lại hoặc chạy lại tuning.',
              '- ' + ('Đã đo checkpoint cuối.' if final else 'Cần đánh giá snapshot step90000 trước khi chốt lựa chọn cuối.'),
              '- ' + ('Đã nhận receipt bảo toàn checkpoint.' if copy_ok else 'Cần lưu bản best riêng, xác minh checksum và nạp lại.'),
              '- ' + ('Đã nhận kết quả attention-fusion.' if attention else 'Chờ teammate hoàn tất và đánh giá attention-fusion.'),
              '- Validation đã dùng lựa chọn checkpoint, không phải test độc lập. '
              'Baseline cache chưa được đối chiếu hoàn toàn pixel ảnh nguồn với run hiện tại.',
              '- Điểm cao hơn paper về số học không chứng minh vượt paper vì protocol/caption/preprocessing khác nhau.',
              '- Minh chứng kiểm tra log/rank không thay thế chạy lại mô hình độc lập.', '', '## 5. Đoạn tóm tắt báo cáo', '',
              'Nhóm đã hoàn tất huấn luyện UniFashion trên FashionIQ với 10 epoch và 90.000 bước cập nhật '
              'ở cấu hình FP32, batch size 2. Minh chứng xuất từ Colab được kiểm tra tính liên tục và '
              'nhất quán với các checkpoint trước. '
              + (f"Trên toàn bộ 6.016 truy vấn validation, checkpoint cuối đạt macro Recall@10 là "
                 f"{fmt(b['Recall@10'])}% và Recall@50 là {fmt(b['Recall@50'])}%, "
                 f"thay đổi {fmt(b['Recall@10'] - a['Recall@10'])} và {fmt(b['Recall@50'] - a['Recall@50'])} "
                 'điểm phần trăm so với step 60.000 khi sử dụng score caption gallery. '
                 f"Nhóm chọn {selected['name']} theo mean macro Recall trong hai ứng viên đã đánh giá. " if final else
                 'Đánh giá Recall của checkpoint cuối và xác minh bản lưu best riêng đang chờ thực hiện. ')
              + ('Bản checkpoint tốt nhất đã được lưu riêng trên Drive, xác minh SHA-256 và nạp lại trên CPU theo receipt xuất từ Colab. ' if copy_ok else '')
              + ('Đã bổ sung kết quả attention-fusion vào bảng so sánh chung. ' if attention else
                 'Kết quả attention-fusion sẽ được bổ sung sau khi teammate hoàn tất run. ')
              + 'Các kết quả hiện có thuộc tập validation; chưa báo cáo test Recall do dữ liệu test không có nhãn target.', '']
    if imported:
        folder = imported['evidence_directory']
        lines += ['## 6. Minh chứng đã kiểm tra', '',
                  f"Nhận ZIP ngày {imported['received_on']}; SHA-256 `{imported['archive_sha256']}`. "
                  'Kiểm tra CRC ZIP, coverage/rank/hit và tính lại Recall của cả hai scoring từ CSV; '
                  'đối chiếu protocol, fingerprint, config, snapshot và receipt bản best. '
                  'Các bảng và quyết định chọn checkpoint tái tạo khớp bản xuất Colab.', '',
                  f"Lượt kiểm thử trước evaluation trong ZIP ghi nhận **{imported['exported_tests']['tests']} tests đạt**, "
                  'khác bộ 16 tests của training. Không có lỗi, thất bại hay test bỏ qua trong XML này.', '',
                  f'- [Raw evaluation status]({folder}/final_results/status.json)',
                  f'- [Kết quả có caption]({folder}/final_results/unifashion_image_caption_score.json)',
                  f'- [Kết quả không score caption]({folder}/final_results/unifashion_image_score.json)',
                  '- [Quyết định chọn checkpoint](checkpoint_selection.json)',
                  '- [Receipt sao chép checkpoint](checkpoint_copy.json)',
                  '- [Bảng so sánh CSV](model_comparison.csv)',
                  '- [Hash các file minh chứng](evidence_import.json)', '',
                  'File `training_status.json` vẫn ghi `retrieval_evaluated: false` vì nó mô tả bước training. '
                  'Evaluation chạy riêng; trạng thái hoàn tất nằm trong `final_results/status.json`. '
                  'Hai file `_partial.csv` trong ZIP đã trùng nội dung với `_per_query.csv`; giữ lại nguyên bản minh chứng.', '']
    (output / 'weekly_report.md').write_text('\n'.join(lines), encoding='utf-8')
    return state


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--final-evaluation', type=Path)
    p.add_argument('--attention-evaluation', type=Path)
    a = p.parse_args()
    print(json.dumps(generate(a.inputs, a.output, a.final_evaluation, a.attention_evaluation), indent=2))
