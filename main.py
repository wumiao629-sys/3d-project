from src.config import load_config, ensure_dirs
from src.data_loader import load_raw_data
from src.validator import validate_data
from src.features import make_features
from src.walk_forward import run_backtest
from src.reporting import save_metrics, save_feature_importance

def main():
    cfg = load_config()
    ensure_dirs(cfg)

    print("="*70)
    print("3D 开奖数据机器学习实验系统")
    print("="*70)

    df = load_raw_data(cfg)
    print(f"原始数据：{len(df)} 行")

    df, errors, warnings = validate_data(df)
    if errors:
        print("\n数据检查失败：")
        for e in errors:
            print(" -", e)
        raise SystemExit(1)

    for w in warnings:
        print("警告：", w)

    X, y, meta = make_features(df, cfg)
    print(f"可用特征行：{len(X)}")
    print(f"特征数量：{X.shape[1]}")

    train_min = cfg["walk_forward"]["min_train_rows"]
    val_min = cfg["data"]["min_validation_rows"]
    test_min = cfg["data"]["min_test_rows"]
    train_n = max(train_min, int(len(X)*cfg["walk_forward"]["initial_train_ratio"]))
    val_n = max(val_min, int(len(X)*cfg["walk_forward"]["validation_ratio"]))
    required = train_n + val_n + test_min

    if len(X) < required:
        print("\n当前数据不足以执行严格的训练/验证/盲测。")
        print(f"当前可用特征行：{len(X)}")
        print(f"当前配置至少需要约：{required}")
        print("程序不会用极少数据制造虚假的模型成绩。")
        return

    metrics, pred_df, model, weights = run_backtest(X, y, meta, cfg)
    save_metrics(metrics, cfg)
    save_feature_importance(model, cfg)

    print("\n===== 严格 Walk-Forward 盲测结果 =====")
    for pos in ["h","t","u"]:
        print(pos, metrics[pos])
    print(
        f"完整号码 Top-{metrics['full_number_top_n']} 命中率："
        f"{metrics['full_number_top_n_hit_rate']:.4f}"
    )
    print(f"测试预测期数：{metrics['test_predictions']}")
    print("\n结果目录：", cfg["paths"]["report_dir"])
    print("回测预测：", cfg["paths"]["backtest_dir"])

if __name__ == "__main__":
    main()
