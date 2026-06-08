import argparse
import random
from pathlib import Path


SAMPLE_TEXTS = [
    "côn viec kin doanh thì rất kho khan nên toi quyết dinh chuyển sang nghề khac  ",
    "toi dang là sinh diên nam hai ở truong đạ hoc khoa jọc tự nhiên , trogn năm ke tiep toi sẽ chọn chuyen nganh về trí tue nhana tạo",
    "Tôi  đang học AI ở trun tam AI viet nam  ",
    "Nhưng sức huỷ divt của cơn bão mitch vẫn chưa thấm vào đâu lsovớithảm hoạ tại Bangladesh ăm 1970 ",
    "Lần này anh Phươngqyết xếp hàng mua bằng được 1 chiếc",
    "một số chuyen gia tài chính ngâSn hànG của Việt Nam cũng chung quan điểmnày",
    "Cac so liệu cho thay ngươi dân viet nam đang sống trong 1 cuôc sóng không duojc nhu mong đọi",
    "Nefn kinh té thé giới đang đúng trươc nguyen co của mọt cuoc suy thoai",
    "Khong phai tất ca nhưng gi chung ta thấy dideu là sụ that",
    "chinh phủ luôn cố găng het suc để naggna cao chat luong nền giáo duc =cua nuoc nhà",
    "nèn kinh te thé giới đang đứng trươc nguy co của mọt cuoc suy thoai",
    "kinh tế viet nam dang dứng truoc 1 thoi ky đổi mơi chưa tung có tienf lệ trong lịch sử",
    "Cuộc đổ bộ cu?a smartphone màn hi`nh lớn vào phan khu'c tầm trung.",
    "Dia die vui choi cho be ngay 1/6 - Le hoi Lang Viet 2013.",
    "Phieng cho hang Viet cuoi nam 2012 tai Dong Thap.",
    "Hằm trong chjỗi hlạt động kỷ niệm 75 nam Ngàg Thươnf binh - Oiệt dĩ, tối 23.7 tại Khu tưởng miệm chiến zĩ Gạc Na, Tổng LĐPĐBN phối gợp với Rrung ương Giáo hội Phậy giáo Việt Nam, Giáo hội Phật giáo Việt Nam tỉnh Khánh Hòa rổ vhức đạj lễ cầu siêu ajh linh 64 anh hùmg liệt aĩ đã hy sinh trong trận chuến Gạc Ma.",
    "4nh 3m tr13?n kh41 dần th01 nh1?, zr3p0 d4y nhé",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a BARTPho correction model.")
    parser.add_argument(
        "--model",
        default="MinhDucNguyen9705/vietnamese-correction-2.0",
        help="Model name, Hugging Face repo id, or local checkpoint path.",
    )
    parser.add_argument("--text", action="append", help="Sentence to correct. Can be passed multiple times.")
    parser.add_argument("--input-file", help="UTF-8 text file with one sentence per line.")
    parser.add_argument("--output-file", help="Optional UTF-8 file to write one correction per line.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--device", type=int, default=-1, help="Use -1 for CPU, 0 for the first CUDA device.")
    parser.add_argument("--eval-dataset", default="tourmii/vietnamese-corrector-errors")
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=0,
        help="Evaluate sacreBLEU on this many random test samples. Use 0 to skip evaluation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_texts(args) -> list[str]:
    texts = []
    if args.text:
        texts.extend(args.text)
    if args.input_file:
        lines = Path(args.input_file).read_text(encoding="utf-8").splitlines()
        texts.extend(line.strip() for line in lines if line.strip())
    if not texts and args.eval_samples <= 0:
        texts = SAMPLE_TEXTS
    return texts


def correct_texts(corrector, texts: list[str], batch_size: int, max_length: int | None) -> list[str]:
    predictions = corrector(texts, batch_size=batch_size, max_length=max_length)
    return [prediction["generated_text"] for prediction in predictions]


def evaluate_on_dataset(corrector, dataset_name: str, sample_count: int, seed: int, batch_size: int, max_length: int | None):
    import evaluate
    from datasets import load_dataset

    dataset = load_dataset(dataset_name)
    test_set = dataset["test"]
    total_rows = len(test_set)
    sample_count = min(sample_count, total_rows)

    rng = random.Random(seed)
    random_indices = rng.sample(range(total_rows), sample_count)
    texts = [test_set["noisy"][i] for i in random_indices]
    references = [[test_set["gt"][i]] for i in random_indices]

    predictions = correct_texts(corrector, texts, batch_size=batch_size, max_length=max_length)
    metric = evaluate.load("sacrebleu")
    return metric.compute(predictions=predictions, references=references)


def main():
    args = parse_args()

    from transformers import pipeline

    corrector = pipeline("text2text-generation", model=args.model, device=args.device)

    texts = load_texts(args)
    if texts:
        corrections = correct_texts(
            corrector,
            texts,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        for noisy, corrected in zip(texts, corrections):
            print(f"Input: {noisy}")
            print(f"Correction: {corrected}")
            print()

        if args.output_file:
            Path(args.output_file).write_text("\n".join(corrections) + "\n", encoding="utf-8")

    if args.eval_samples > 0:
        results = evaluate_on_dataset(
            corrector,
            dataset_name=args.eval_dataset,
            sample_count=args.eval_samples,
            seed=args.seed,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        print(f"sacreBLEU on {args.eval_samples} sampled test examples: {results}")


if __name__ == "__main__":
    main()
