poetry run python $PYTHONPATH/experiments/train.py \
-m model.pretrained=roberta-large \
train.lr=3e-5 \
train.batch_size=32 \
train.mode=cv \
preprocess.max_length=32 \
train.epochs=5 \
train.start_factor=0.5 \
train.save_model=True


poetry run python $PYTHONPATH/experiments/train.py \
-m model.pretrained=bert-base-uncased  \
train.lr=4e-5 \
train.batch_size=16 \
train.mode=cv \
preprocess.max_length=64 \
train.epochs=5 \
train.start_factor=0.5 \
train.save_model=True

poetry run python $PYTHONPATH/experiments/train.py \
-m model.pretrained=bert-large-uncased  \
train.lr=3e-5 \
train.batch_size=32 \
train.mode=cv \
preprocess.max_length=32 \
train.epochs=5 \
train.start_factor=0.5 \
train.save_model=True

poetry run python $PYTHONPATH/experiments/train.py \
-m model.pretrained=roberta-base  \
train.lr=3e-5 \
train.batch_size=128 \
train.mode=cv \
preprocess.max_length=32 \
train.epochs=5 \
train.start_factor=0.5 \
train.save_model=True

poetry run python $PYTHONPATH/experiments/train.py \
-m model.pretrained=albert-base-v2  \
train.lr=3e-5 \
train.batch_size=16 \
train.mode=cv \
preprocess.max_length=32 \
train.epochs=5 \
train.start_factor=0.5 \
train.save_model=True

poetry run python $PYTHONPATH/experiments/train.py \
-m model.pretrained=xlnet-large-cased  \
train.lr=2e-5 \
train.batch_size=32 \
train.mode=cv \
preprocess.max_length=32 \
train.epochs=5 \
train.start_factor=0.5 \
train.save_model=True

poetry run python $PYTHONPATH/experiments/train.py \
-m model.pretrained=xlnet-base-cased  \
train.lr=4e-5 \
train.batch_size=16 \
train.mode=cv \
preprocess.max_length=32 \
train.epochs=5 \
train.start_factor=0.5 \
train.save_model=True

poetry run python $PYTHONPATH/experiments/train.py \
-m model.pretrained=albert-large-v2  \
train.lr=5e-5 \
train.batch_size=32 \
train.mode=cv \
preprocess.max_length=32 \
train.epochs=5 \
train.start_factor=0.5 \
train.save_model=True

