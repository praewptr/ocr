import evaluate

# Load metrics once to avoid repeated loading
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def compute_metrics(pred, processor):

    if isinstance(pred.predictions, tuple):
        pred_ids = pred.predictions[0]
    else:
        pred_ids = pred.predictions

    label_ids = pred.label_ids

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_ids = pred_ids.astype("int64")
    label_ids = label_ids.astype("int64")

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "cer": cer}

def create_compute_metrics_function(processor):

    def compute_metrics_bound(pred):
        return compute_metrics(pred, processor)
    
    return compute_metrics_bound