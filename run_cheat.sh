python -um src.main \
--dataSet pubmed \
--epochs 200 \
--b_sz 60 \
--cuda \
--learn_method sup \
--name pubmed_trans20_cheat \
--transductive --node_per_class 20 \
--cheat \
--conf ./src/experiments.conf