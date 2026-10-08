from src.config import load_config, ensure_dirs
from src.predict import train_full_and_predict

if __name__ == "__main__":
    cfg = load_config()
    ensure_dirs(cfg)
    cand, final = train_full_and_predict(cfg)
    print("="*70)
    print("下一期概率候选 Top-N")
    print("="*70)
    print(cand.to_string(index=False))
