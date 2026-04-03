# Training Improvement Plan for HyperSpectral CFINet (172 channels)

## Critical Issues Identified

1. **Validation AP = 0.000** - Model is not detecting anything
2. **Contrastive loss = 0.0000** - Contrastive learning component not active
3. **Backbone conv1 initialization** - First layer (172→64) not properly initialized
4. **Training loss very low** - May indicate underfitting or data issues

---

## Step-by-Step Diagnostic Plan

### Phase 1: Data & Annotation Verification (HIGH PRIORITY)

#### Step 1.1: Verify Dataset Loading
- [ ] Check if annotations are being loaded correctly
- [ ] Verify `train.jsonl`, `validate.jsonl` contain valid bbox annotations
- [ ] Check annotation format matches expected COCO format
- [ ] Count number of ground truth boxes per image in train/val sets
- [ ] Verify image paths are correct and images are accessible

**Action**: Add debug prints or inspect dataset:
```python
# Check in dataset class
print(f"Number of images: {len(self)}")
print(f"Sample annotation: {self.data_infos[0]}")
print(f"GT bboxes shape: {gt_bboxes.shape}")
```

#### Step 1.2: Verify Normalization
- [ ] Check if mean/std values are appropriate for your hyperspectral data range
- [ ] Verify normalization is not zeroing out important information
- [ ] Check if data values are in expected range after normalization

**Action**: Inspect a sample image:
```python
# After normalization, check:
print(f"Image min: {img.min()}, max: {img.max()}, mean: {img.mean()}")
print(f"Image shape: {img.shape}")  # Should be [172, H, W]
```

#### Step 1.3: Visualize Training Data
- [ ] Visualize a few training images with GT bboxes overlaid
- [ ] Check if bboxes are reasonable sizes and positions
- [ ] Verify bbox coordinates are in correct format (x1, y1, x2, y2)

---

### Phase 2: Model Architecture Issues (HIGH PRIORITY)

#### Step 2.1: Backbone Initialization Problem
**Issue**: Line 1167 in log shows `backbone.conv1.weight` is NOT initialized from pretrained weights (expected, since pretrained has 3 channels, you have 172). However, it may not be properly initialized.

**Action**: 
- [ ] Check if conv1 is using Kaiming/He initialization (default for ResNet)
- [ ] Consider using a better initialization strategy for 172-channel input
- [ ] Option: Initialize conv1 with channel-wise averaging or PCA-based initialization

**Potential Fix**: Add custom initialization for first layer:
```python
# In config or model initialization
# Option 1: Initialize 172 channels by repeating 3-channel pretrained weights
# Option 2: Use Kaiming init with proper gain
# Option 3: Use spectral normalization or other advanced init
```

#### Step 2.2: Verify Model Forward Pass
- [ ] Check if model outputs valid predictions (not all zeros/NaNs)
- [ ] Verify RPN is generating proposals
- [ ] Check if RoI head is receiving valid proposals
- [ ] Monitor proposal counts during training

**Action**: Add debug hooks:
```python
# In training loop, log:
# - Number of RPN proposals per image
# - Proposal scores distribution
# - RoI features statistics
```

---

### Phase 3: Contrastive Loss Investigation (MEDIUM PRIORITY)

#### Step 3.1: Why Contrastive Loss is Zero
**Root Cause Analysis**:
- Contrastive loss requires high-quality GT instances (hq_score >= 0.65, pro_counts >= 8)
- Early in training, no instances meet these criteria
- Queue directory might be empty initially

**Actions**:
- [ ] Check if `con_queue_dir` exists and is writable
- [ ] Monitor `hq_inds` count during training (should increase over time)
- [ ] Lower `hq_score` threshold temporarily (0.65 → 0.3) to see if loss activates
- [ ] Lower `hq_pro_counts_thr` (8 → 2) for initial training
- [ ] Check if contrastive queue is being populated

**Potential Fix**: Adjust thresholds in config:
```python
ins_quality_assess_cfg=dict(
    cls_score=0.05,
    hq_score=0.30,  # Lower from 0.65
    lq_score=0.15,  # Lower from 0.25
    hq_pro_counts_thr=2),  # Lower from 8
```

#### Step 3.2: Contrastive Loss Warmup
- [ ] Consider disabling contrastive loss for first few epochs
- [ ] Gradually increase contrastive loss weight during training
- [ ] Use curriculum learning approach

---

### Phase 4: Training Configuration (MEDIUM PRIORITY)

#### Step 4.1: Learning Rate Schedule
**Current**: Step schedule with warmup
- [ ] Verify learning rate is appropriate for your data size
- [ ] Consider using cosine annealing instead of step
- [ ] Check if LR decay at epoch 8/11 is too aggressive

**Potential Fix**:
```python
lr_config = dict(
    policy='cosine',  # Instead of 'step'
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    min_lr=0.0001)
```

#### Step 4.2: Loss Weights
- [ ] RPN weight = 0.9 might be too low
- [ ] Contrastive loss weight = 0.5 might need adjustment
- [ ] Consider rebalancing loss components

#### Step 4.3: Training Epochs
- [ ] 12 epochs might be insufficient
- [ ] Consider training longer (24-36 epochs)
- [ ] Add more frequent validation (every 3 epochs instead of 12)

---

### Phase 5: Detection-Specific Issues (HIGH PRIORITY)

#### Step 5.1: Score Threshold
**Current**: `score_thr=0.05` in test_cfg
- [ ] This might be filtering out all predictions
- [ ] Try lowering to 0.01 or 0.001 for debugging
- [ ] Check prediction score distribution

**Action**: Temporarily lower threshold:
```python
test_cfg=dict(
    rpn=dict(max_per_img=300, nms=dict(iou_threshold=0.5)),
    rcnn=dict(score_thr=0.001))  # Lower from 0.05
```

#### Step 5.2: NMS Threshold
- [ ] Check if NMS is too aggressive (iou_threshold=0.5)
- [ ] Verify proposals are being generated
- [ ] Check if proposals pass score threshold

#### Step 5.3: Anchor Configuration
- [ ] Verify anchor scales/ratios are appropriate for your object sizes
- [ ] Check if anchors match your GT bbox sizes
- [ ] Consider adjusting anchor generator settings

---

### Phase 6: Debugging Tools

#### Step 6.1: Add Comprehensive Logging
- [ ] Log number of GT boxes per image
- [ ] Log number of positive/negative anchors
- [ ] Log RPN proposal counts and scores
- [ ] Log RoI head predictions
- [ ] Log contrastive loss components

#### Step 6.2: Visualization Tools
- [ ] Visualize RPN proposals on validation images
- [ ] Visualize final predictions
- [ ] Compare predictions vs ground truth
- [ ] Plot loss curves for all components

#### Step 6.3: Model Inspection
- [ ] Check model parameter statistics (mean, std, min, max)
- [ ] Monitor gradient flow
- [ ] Check for vanishing/exploding gradients
- [ ] Verify batch normalization statistics

---

## Recommended Action Order

### Immediate (Do First):
1. **Verify dataset annotations** - Check if GT boxes exist and are valid
2. **Lower score threshold** - Change test_cfg score_thr to 0.001
3. **Add validation logging** - Log proposal counts and prediction scores
4. **Check backbone initialization** - Verify conv1 is properly initialized

### Short-term (Next Steps):
5. **Adjust contrastive loss thresholds** - Lower hq_score and hq_pro_counts_thr
6. **Fix backbone initialization** - Implement proper init for 172-channel input
7. **Increase validation frequency** - Evaluate every 3 epochs instead of 12
8. **Add debug visualizations** - Visualize predictions on validation set

### Medium-term (After Initial Fixes):
9. **Tune learning rate schedule** - Consider cosine annealing
10. **Adjust loss weights** - Rebalance RPN vs RCNN vs contrastive
11. **Extend training** - Train for more epochs
12. **Hyperparameter tuning** - Optimize anchor scales, NMS thresholds

---

## Expected Outcomes After Fixes

1. **Validation AP should increase** from 0.000 to > 0.1 (even 0.1 is progress)
2. **Contrastive loss should activate** after a few epochs (non-zero values)
3. **Training loss should stabilize** and show clear convergence
4. **Proposal counts should be reasonable** (not zero, not too many)

---

## Monitoring Checklist

During next training run, monitor:
- [ ] Number of GT boxes per batch
- [ ] Number of RPN proposals per image
- [ ] Proposal score distribution (should not be all zeros)
- [ ] RoI head prediction scores
- [ ] Contrastive loss activation (should become non-zero after epoch 2-3)
- [ ] Validation AP progression
- [ ] Gradient norms (should be stable, not vanishing/exploding)

---

## Notes

- **DO NOT modify code yet** - First complete diagnostic steps
- Document findings at each step
- Make one change at a time to isolate issues
- Keep backups of working configs
- Use version control for experiments
