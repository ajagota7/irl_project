auc_roc = metrics['auc_roc']
pearson = metrics['pearson_correlation']
auc_roc_str = f"{float(auc_roc):.4f}" if not isinstance(auc_roc, str) else auc_roc
pearson_str = f"{float(pearson):.4f}" if not isinstance(pearson, str) else pearson
print(f"  AUC-ROC: {auc_roc_str}, Pearson: {pearson_str}") 