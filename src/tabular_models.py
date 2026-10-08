import numpy as np
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

class MultiPositionModels:
    def __init__(self, cfg):
        seed = cfg["models"]["random_seed"]
        self.cfg = cfg
        p = cfg["models"]
        self.lgb = {}
        self.xgb = {}
        self.cat = {}
        for pos in ["h", "t", "u"]:
            self.lgb[pos] = LGBMClassifier(
                objective="multiclass", num_class=10, random_state=seed,
                n_estimators=p["lgbm"]["n_estimators"],
                learning_rate=p["lgbm"]["learning_rate"],
                num_leaves=p["lgbm"]["num_leaves"],
                max_depth=p["lgbm"]["max_depth"],
                subsample=p["lgbm"]["subsample"],
                colsample_bytree=p["lgbm"]["colsample_bytree"],
                reg_lambda=p["lgbm"]["reg_lambda"],
                reg_alpha=p["lgbm"]["reg_alpha"],
                verbosity=-1
            )
            self.xgb[pos] = XGBClassifier(
                objective="multi:softprob", num_class=10, random_state=seed,
                n_estimators=p["xgb"]["n_estimators"],
                learning_rate=p["xgb"]["learning_rate"],
                max_depth=p["xgb"]["max_depth"],
                min_child_weight=p["xgb"]["min_child_weight"],
                subsample=p["xgb"]["subsample"],
                colsample_bytree=p["xgb"]["colsample_bytree"],
                reg_lambda=p["xgb"]["reg_lambda"],
                reg_alpha=p["xgb"]["reg_alpha"],
                eval_metric="mlogloss",
                tree_method="hist"
            )
            self.cat[pos] = CatBoostClassifier(
                loss_function="MultiClass", random_seed=seed,
                iterations=p["catboost"]["iterations"],
                learning_rate=p["catboost"]["learning_rate"],
                depth=p["catboost"]["depth"],
                l2_leaf_reg=p["catboost"]["l2_leaf_reg"],
                verbose=p["catboost"]["verbose"]
            )

    def fit(self, X, y):
        for pos, target in [("h", "target_h"), ("t", "target_t"), ("u", "target_u")]:
            self.lgb[pos].fit(X, y[target])
            self.xgb[pos].fit(X, y[target])
            self.cat[pos].fit(X, y[target])
        return self

    def predict_proba(self, X):
        result = {}
        for model_name, models in [("lgb", self.lgb), ("xgb", self.xgb), ("cat", self.cat)]:
            result[model_name] = {}
            for pos in ["h", "t", "u"]:
                result[model_name][pos] = models[pos].predict_proba(X)
        return result

    def feature_importance(self):
        rows = []
        for pos in ["h", "t", "u"]:
            imp = self.lgb[pos].feature_importances_
            names = self.lgb[pos].feature_name_
            for n, v in zip(names, imp):
                rows.append({"model": "LightGBM", "position": pos, "feature": n, "importance": float(v)})
        return rows
