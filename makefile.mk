EPOCHS=50
LR=0.001

venv/bin/python3:
	python3 -m venv venv
	python3 -m pip install -r requirements.txt

define make-targets
data/$1/$2/done: venv/bin/python3
	mkdir -p data/$1/$2
	PYTHONPATH="src" venv/bin/python3 src/train.py  --epochs $(EPOCHS) --learning-rate $(LR) --dir data/$1/$2 --model $1 --seed $2
	touch data/$1/$2/done

clean-$1-$2:
	rm -rf data/$1/$2
endef

MODELS=Transformer HalveTransformer
SEEDS=0 1 2

$(foreach model,$(MODELS),$(foreach seed,$(SEEDS),$(eval $(call make-targets,$(model),$(seed)))))

full-transformer: $(foreach seed,$(SEEDS),data/Transformer/$(seed)/done)
halve-transformer: $(foreach seed,$(SEEDS),data/HalveTransformer/$(seed)/done)
full-transformer-clean: $(foreach seed,$(SEEDS),clean-Transformer-$(seed))
halve-transformer-clean: $(foreach seed,$(SEEDS),clean-HalveTransformer-$(seed))
clean: full-transformer-clean halve-transformer-clean
all: full-transformer halve-transformer

figs/gpumem.png: full-transformer halve-transformer
	jet init --shape 1 1 \
    jet line --input-path data/Transformer/2/train.jsonl      --input-path data/Transformer/1/train.jsonl      --input-path data/Transformer/0/train.jsonl      \
		--x epoch --y gpumem --where "epoch,>,4" --color 1 .6 0    --label full-transforler \
    jet line --input-path data/HalveTransformer/2/train.jsonl --input-path data/HalveTransformer/1/train.jsonl --input-path data/HalveTransformer/0/train.jsonl \
		--x epoch --y gpumem --where "epoch,>,4" --color .5 0.2 .7 --label halve-transformer \
	jet mod --top-spine False --right-spine False --x-label epoch --y-label "GPU Mem. (MB)" \
    jet plot --show False --output-path figs/gpumem.png

figs/time.png: full-transformer halve-transformer
	jet init --shape 1 1 \
    jet line --input-path data/Transformer/2/train.jsonl      --input-path data/Transformer/1/train.jsonl      --input-path data/Transformer/0/train.jsonl      \
		--x epoch --y time --where "epoch,>,4" --color 1 .6 0    --label full-transforler \
    jet line --input-path data/HalveTransformer/2/train.jsonl --input-path data/HalveTransformer/1/train.jsonl --input-path data/HalveTransformer/0/train.jsonl \
		--x epoch --y time --where "epoch,>,4" --color .5 0.2 .7 --label halve-transformer \
	jet mod --top-spine False --right-spine False --x-label epoch --y-label "time (s)" \
    jet plot --show False --output-path figs/time.png

figs/accuracy.png: full-transformer halve-transformer
	jet init --shape 1 1 \
	jet line --input-path data/Transformer/2/train.jsonl      --input-path data/Transformer/1/train.jsonl      --input-path data/Transformer/0/train.jsonl      \
		--x epoch --y accuracy --color 1 .6 0    --label train-full-transforler \
	jet line --input-path data/HalveTransformer/2/train.jsonl --input-path data/HalveTransformer/1/train.jsonl --input-path data/HalveTransformer/0/train.jsonl \
		--x epoch --y accuracy --color .5 0.2 .7 --label train-halve-transformer \
	jet line --input-path data/Transformer/2/valid.jsonl      --input-path data/Transformer/1/valid.jsonl      --input-path data/Transformer/0/valid.jsonl      \
		--x epoch --y accuracy --color 1 .6 0    --label valid-full-transforler  --linestyle "--" \
	jet line --input-path data/HalveTransformer/2/valid.jsonl --input-path data/HalveTransformer/1/valid.jsonl --input-path data/HalveTransformer/0/valid.jsonl \
		--x epoch --y accuracy --color .5 0.2 .7 --label valid-halve-transformer --linestyle "--" \
	jet mod --y-lim 0 1 --top-spine False --right-spine False --x-label epoch --y-label "accuracy" \
	jet plot --show False --output-path figs/accuracy.png

figs/loss.png: full-transformer halve-transformer
	jet init --shape 1 1 \
	jet line --input-path data/Transformer/2/train.jsonl      --input-path data/Transformer/1/train.jsonl      --input-path data/Transformer/0/train.jsonl      \
		--x epoch --y loss --color 1 .6 0    --label train-full-transforler \
	jet line --input-path data/HalveTransformer/2/train.jsonl --input-path data/HalveTransformer/1/train.jsonl --input-path data/HalveTransformer/0/train.jsonl \
		--x epoch --y loss --color .5 0.2 .7 --label train-halve-transformer \
	jet line --input-path data/Transformer/2/valid.jsonl      --input-path data/Transformer/1/valid.jsonl      --input-path data/Transformer/0/valid.jsonl      \
		--x epoch --y loss --color 1 .6 0    --label valid-full-transforler  --linestyle "--" \
	jet line --input-path data/HalveTransformer/2/valid.jsonl --input-path data/HalveTransformer/1/valid.jsonl --input-path data/HalveTransformer/0/valid.jsonl \
		--x epoch --y loss --color .5 0.2 .7 --label valid-halve-transformer --linestyle "--" \
	jet mod --y-lim 0 1 --top-spine False --right-spine False --x-label epoch --y-label "loss" \
	jet plot --show False --output-path figs/loss.png

figs: figs/gpumem.png figs/time.png figs/accuracy.png figs/loss.png
clean-figs: 
	rm -rf figs/*
