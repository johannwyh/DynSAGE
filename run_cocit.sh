python -um src.main \
--gpu-id 7 \
--dataSet cocit \
--epochs 200 \
--b_sz 60 \
--cuda \
--learn_method sup \
--conf ./src/experiments.conf \
--name cocit_raw_expand \
--DynamicSAGE 
#--expand-train-set --thres-top1 0.99 --warm-up 50 --expand-freq 20 \
#--graph-revise --revise-topk 10 --plot
